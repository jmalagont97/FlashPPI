import hashlib
import os
from collections import OrderedDict
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm


def sequence_md5(seq: str) -> str:
    """Computes MD5 hash for a biological sequence (stripped and uppercased)."""
    return hashlib.md5(seq.strip().upper().encode("utf-8")).hexdigest()


def default_cache_path(fasta_path: str) -> str:
    """Returns default HDF5 cache path for a given FASTA file path."""
    base, _ = os.path.splitext(fasta_path)
    return f"{base}.flashppi.h5"


def validate_compatible(
    h5_path: str,
    model_name: str,
    max_len: int,
    model_revision: Optional[str] = None,
    md5s: Optional[List[str]] = None,
) -> List[str]:
    """
    Validates whether an existing HDF5 embedding file matches the expected parameters.
    Returns a list of error descriptions (empty list if completely compatible).
    """
    errors = []
    if not os.path.exists(h5_path):
        return ["File does not exist."]

    try:
        with h5py.File(h5_path, "r") as h5:
            fmt_ver = h5.attrs.get("format_version", 1)
            if fmt_ver != 1:
                errors.append(f"Incompatible format_version: expected 1, found {fmt_ver}")

            stored_model = h5.attrs.get("model_name", "")
            if stored_model != model_name:
                errors.append(f"Model mismatch: expected '{model_name}', found '{stored_model}'")

            stored_max_len = h5.attrs.get("max_len", -1)
            if int(stored_max_len) != int(max_len):
                errors.append(f"max_len mismatch: expected {max_len}, found {stored_max_len}")

            if md5s is not None:
                if "seq_md5" not in h5:
                    errors.append("Missing 'seq_md5' dataset in cache.")
                else:
                    stored_md5s = list(h5["seq_md5"].asstr()[:])
                    if len(stored_md5s) != len(md5s):
                        errors.append(f"Sequence count mismatch: expected {len(md5s)}, found {len(stored_md5s)}")
                    elif stored_md5s != md5s:
                        errors.append("Sequence hashes or order do not match input FASTA.")
    except Exception as e:
        errors.append(f"Error reading HDF5 file: {str(e)}")

    return errors


def write_embedding_store(
    h5_path: str,
    ids: List[str],
    sequences: List[str],
    model: torch.nn.Module,
    tokenizer,
    device: Union[str, torch.device],
    batch_size: int = 64,
    max_len: int = 1024,
    model_name: str = "tattabio/flashppi",
    model_revision: Optional[str] = None,
    overwrite: bool = False,
    desc: str = "Encoding proteins",
) -> None:
    """
    Encodes proteins in batches and stores residue-level and projection embeddings in HDF5 (fp16).
    Supports resuming interrupted runs and idempotent execution.
    """
    n_seqs = len(sequences)
    if n_seqs == 0:
        raise ValueError("Cannot write embedding store for empty sequences list.")

    md5s = [sequence_md5(s) for s in sequences]

    if os.path.exists(h5_path):
        if overwrite:
            os.remove(h5_path)
        else:
            errors = validate_compatible(h5_path, model_name=model_name, max_len=max_len, md5s=md5s)
            if errors:
                raise ValueError(
                    f"Existing cache '{h5_path}' is incompatible:\n - "
                    + "\n - ".join(errors)
                    + "\nUse --overwrite_embeddings to regenerate it."
                )
            with h5py.File(h5_path, "r") as h5:
                if h5.attrs.get("complete", False):
                    print(f"Embedding cache '{h5_path}' is complete. Skipping encoding.")
                    return

    # Compute sequence token lengths using tokenizer in CPU
    print(f"Pre-tokenizing {n_seqs} sequences to determine exact tensor dimensions...")
    all_lengths = []
    tok_batch_size = 1024
    for i in range(0, n_seqs, tok_batch_size):
        sub_seqs = sequences[i : i + tok_batch_size]
        tok_out = tokenizer(sub_seqs, padding=False, truncation=True, max_length=max_len)
        all_lengths.extend([len(tok_ids) for tok_ids in tok_out["input_ids"]])

    lengths = np.array(all_lengths, dtype=np.int32)
    offsets = np.zeros(n_seqs + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    total_residues = int(offsets[-1])

    # Infer embedding dimensions if not already created
    if not os.path.exists(h5_path):
        first_seq = [sequences[0]]
        inp = tokenizer(first_seq, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        if hasattr(inp, "to"):
            inp = inp.to(device)
        else:
            inp = {k: v.to(device) if hasattr(v, "to") else v for k, v in inp.items()}

        autocast_device = str(device).split(":")[0] if "cuda" in str(device) else "cpu"
        autocast_dtype = torch.float16 if "cuda" in str(device) else torch.bfloat16

        with torch.no_grad(), torch.autocast(autocast_device, dtype=autocast_dtype):
            res_dummy = model.encode_protein(inp["input_ids"], inp["attention_mask"])
            q_dummy = model.head_q(res_dummy, inp["attention_mask"])
            plm_dim = int(res_dummy.shape[-1])
            clip_dim = int(q_dummy.shape[-1])

        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(h5_path, "w") as h5:
            h5.attrs["format_version"] = 1
            h5.attrs["model_name"] = str(model_name)
            h5.attrs["model_revision"] = str(model_revision or "unknown")
            h5.attrs["max_len"] = int(max_len)
            h5.attrs["residue_dtype"] = "float16"
            h5.attrs["plm_dim"] = plm_dim
            h5.attrs["clip_dim"] = clip_dim
            h5.attrs["complete"] = False
            h5.attrs["created_at"] = datetime.now(timezone.utc).isoformat()

            h5.create_dataset("ids", data=ids, dtype=str_dtype)
            h5.create_dataset("sequences", data=sequences, dtype=str_dtype)
            h5.create_dataset("seq_md5", data=md5s, dtype=str_dtype)
            h5.create_dataset("lengths", data=lengths, dtype=np.int32)
            h5.create_dataset("offsets", data=offsets, dtype=np.int64)
            h5.create_dataset("done", data=np.zeros(n_seqs, dtype=bool), dtype=bool)

            chunk_len = min(4096, max(1, total_residues))
            h5.create_dataset(
                "residues",
                shape=(total_residues, plm_dim),
                dtype=np.float16,
                chunks=(chunk_len, plm_dim),
            )
            h5.create_dataset("q_embeds", shape=(n_seqs, clip_dim), dtype=np.float32)
            h5.create_dataset("k_embeds", shape=(n_seqs, clip_dim), dtype=np.float32)

    # Open file in read-write mode to process pending sequences
    with h5py.File(h5_path, "r+") as h5:
        done_mask = np.array(h5["done"][:], dtype=bool)
        pending_indices = np.where(~done_mask)[0].tolist()

        if not pending_indices:
            h5.attrs["complete"] = True
            print(f"All {n_seqs} sequences already encoded in '{h5_path}'.")
            return

        print(f"{desc}: {len(pending_indices)} of {n_seqs} sequences pending.")
        # Sort pending indices by length descending to fail fast on OOM and minimize padding waste
        pending_indices.sort(key=lambda idx: lengths[idx], reverse=True)

        autocast_device = str(device).split(":")[0] if "cuda" in str(device) else "cpu"
        autocast_dtype = torch.float16 if "cuda" in str(device) else torch.bfloat16

        with torch.no_grad():
            for b_start in tqdm(range(0, len(pending_indices), batch_size), desc=desc, unit="batch"):
                batch_idxs = pending_indices[b_start : b_start + batch_size]
                batch_seqs = [sequences[idx] for idx in batch_idxs]

                inputs = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )
                if hasattr(inputs, "to"):
                    inputs = inputs.to(device)
                else:
                    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

                with torch.autocast(autocast_device, dtype=autocast_dtype):
                    res_embed = model.encode_protein(inputs["input_ids"], inputs["attention_mask"])
                    q = model.head_q(res_embed, inputs["attention_mask"]).cpu().float().numpy()
                    k = model.head_k(res_embed, inputs["attention_mask"]).cpu().float().numpy()

                batch_token_lens = inputs["attention_mask"].sum(dim=1).tolist()

                for j, orig_idx in enumerate(batch_idxs):
                    seq_len = int(batch_token_lens[j])
                    off = offsets[orig_idx]
                    # Convert to fp16 CPU numpy array
                    res_slice = res_embed[j, :seq_len, :].cpu().half().numpy()
                    h5["residues"][off : off + seq_len, :] = res_slice
                    h5["q_embeds"][orig_idx] = q[j]
                    h5["k_embeds"][orig_idx] = k[j]
                    h5["done"][orig_idx] = True

                h5.flush()
                del res_embed, q, k, inputs

        h5.attrs["complete"] = True
        h5.flush()
        print(f"Successfully finished embedding {n_seqs} sequences into '{h5_path}'.")


class EmbeddingStore:
    """
    Memory-efficient reader for FlashPPI HDF5 embedding files.
    Loads projection embeddings (q/k) and sequence metadata into RAM,
    while serving residue embeddings on-demand via LRU cache or full preload.
    """

    def __init__(
        self,
        h5_path: str,
        cache_gb: float = 4.0,
        preload: Optional[bool] = None,
    ):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Embedding store file '{h5_path}' does not exist.")

        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, "r")

        if not self.h5.attrs.get("complete", False):
            raise ValueError(f"Embedding cache '{h5_path}' is incomplete or was interrupted.")

        self.attrs = dict(self.h5.attrs)
        self.ids = list(self.h5["ids"].asstr()[:])
        self.lengths = np.array(self.h5["lengths"][:], dtype=np.int32)
        self.offsets = np.array(self.h5["offsets"][:], dtype=np.int64)
        self.q_embeds = np.array(self.h5["q_embeds"][:], dtype=np.float32)
        self.k_embeds = np.array(self.h5["k_embeds"][:], dtype=np.float32)
        self.plm_dim = int(self.h5["residues"].shape[1])
        self.clip_dim = int(self.q_embeds.shape[1])

        total_residue_bytes = self.h5["residues"].size * self.h5["residues"].dtype.itemsize
        max_cache_bytes = int(cache_gb * (1024**3))

        if preload is True or (preload is None and total_residue_bytes <= max_cache_bytes):
            # Preload entire residue dataset into RAM
            self._full_residues = np.array(self.h5["residues"][:], dtype=np.float16)
            self._cache = None
            self._cache_bytes = 0
            self._max_cache_bytes = max_cache_bytes
        else:
            self._full_residues = None
            self._cache = OrderedDict()
            self._cache_bytes = 0
            self._max_cache_bytes = max_cache_bytes

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, idx: int) -> torch.Tensor:
        """Returns residue embeddings for protein idx as CPU fp16 Tensor of shape (L_i, D)."""
        off = self.offsets[idx]
        seq_len = int(self.lengths[idx])
        if seq_len == 0:
            return torch.empty((0, self.plm_dim), dtype=torch.float16)

        if self._full_residues is not None:
            slice_np = self._full_residues[off : off + seq_len, :]
            return torch.from_numpy(slice_np)

        if idx in self._cache:
            self._cache.move_to_end(idx)
            return torch.from_numpy(self._cache[idx])

        slice_np = np.array(self.h5["residues"][off : off + seq_len, :], dtype=np.float16)
        item_bytes = slice_np.nbytes

        while (self._cache_bytes + item_bytes > self._max_cache_bytes) and len(self._cache) > 0:
            _, evicted_val = self._cache.popitem(last=False)
            self._cache_bytes -= evicted_val.nbytes

        self._cache[idx] = slice_np
        self._cache_bytes += item_bytes
        return torch.from_numpy(slice_np)

    def get_padded(
        self, idxs: Sequence[int], device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gathers residue embeddings for a batch of protein indices,
        pads them to the maximum sequence length of the batch,
        and constructs the exact attention mask from sequence lengths.

        Returns:
            pad_tensor: (B, max_L, D) on device (float16)
            mask_tensor: (B, max_L) on device (torch.long, 1 for valid tokens, 0 for padding)
        """
        B = len(idxs)
        if B == 0:
            return (
                torch.empty((0, 0, self.plm_dim), dtype=torch.float16, device=device),
                torch.empty((0, 0), dtype=torch.long, device=device),
            )

        batch_lens = [int(self.lengths[i]) for i in idxs]
        max_L = max(batch_lens) if batch_lens else 0

        pad_tensor = torch.zeros((B, max_L, self.plm_dim), dtype=torch.float16)
        mask_tensor = torch.zeros((B, max_L), dtype=torch.long)

        for row, i in enumerate(idxs):
            seq_len = batch_lens[row]
            if seq_len > 0:
                res_i = self.get(i)
                pad_tensor[row, :seq_len, :] = res_i
                mask_tensor[row, :seq_len] = 1

        device_str = str(device)
        non_blocking = "cuda" in device_str
        return pad_tensor.to(device, non_blocking=non_blocking), mask_tensor.to(
            device, non_blocking=non_blocking
        )

    def close(self) -> None:
        if hasattr(self, "h5") and self.h5:
            try:
                self.h5.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
