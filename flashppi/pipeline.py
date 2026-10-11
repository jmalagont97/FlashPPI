import os
from typing import List, Optional, Sequence, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .embed_store import (
    EmbeddingStore,
    default_cache_path,
    sequence_md5,
    validate_compatible,
    write_embedding_store,
)


def load_model(
    model_name: str = "tattabio/flashppi",
    device: Union[str, torch.device] = "cuda",
) -> Tuple[AutoModel, AutoTokenizer]:
    """Loads tokenizer and model weights for FlashPPI."""
    print(f"Loading model and tokenizer ({model_name}) to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    return model, tokenizer


def read_fasta(fasta_path: str) -> Tuple[List[str], List[str]]:
    """Reads a FASTA file and returns (ids, sequences)."""
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise ValueError(f"FASTA file '{fasta_path}' contains no sequences.")

    ids = [r.id for r in records]
    sequences = [str(r.seq) for r in records]

    # Warn if duplicate IDs exist
    if len(ids) != len(set(ids)):
        duplicates = len(ids) - len(set(ids))
        print(f"WARNING: FASTA contains {duplicates} duplicate IDs. Identifiers may clash in output.")

    return ids, sequences


def ensure_embeddings(
    fasta_path: Optional[str] = None,
    h5_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tokenizer=None,
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 64,
    max_len: int = 1024,
    model_name: str = "tattabio/flashppi",
    overwrite: bool = False,
    cache_gb: float = 4.0,
    desc: str = "Encoding proteome",
) -> EmbeddingStore:
    """
    Ensures that protein embeddings exist in an HDF5 cache, creating or resuming them if needed,
    and returns an open EmbeddingStore instance.
    """
    if not h5_path:
        if not fasta_path:
            raise ValueError("Either fasta_path or h5_path must be provided.")
        h5_path = default_cache_path(fasta_path)

    # If the file exists and is already complete, open directly unless overwrite is requested
    if os.path.exists(h5_path) and not overwrite:
        errors = validate_compatible(h5_path, model_name=model_name, max_len=max_len)
        if not errors:
            try:
                store = EmbeddingStore(h5_path, cache_gb=cache_gb)
                print(f"Loaded existing embedding cache: {h5_path} ({len(store)} proteins)")
                return store
            except ValueError:
                # File is present but incomplete, proceed to resume
                print(f"Cache '{h5_path}' exists but is incomplete. Resuming encoding...")
        else:
            print(f"Existing cache '{h5_path}' is incompatible: {errors}. Recomputing...")

    # If we need to write or resume, fasta_path and model/tokenizer are required
    if not fasta_path:
        raise ValueError(f"Cache file '{h5_path}' is missing or incomplete, but no FASTA file was provided to generate it.")
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be provided to compute protein embeddings.")

    ids, sequences = read_fasta(fasta_path)
    write_embedding_store(
        h5_path=h5_path,
        ids=ids,
        sequences=sequences,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_len=max_len,
        model_name=model_name,
        overwrite=overwrite,
        desc=desc,
    )

    return EmbeddingStore(h5_path, cache_gb=cache_gb)


def faiss_topk(
    query_embeds: np.ndarray,
    key_embeds: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs maximum inner-product nearest-neighbor search with FAISS."""
    dim = key_embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(key_embeds, dtype=np.float32))

    search_k = min(k, len(key_embeds))
    distances, indices = index.search(
        np.ascontiguousarray(query_embeds, dtype=np.float32), k=search_k
    )
    return distances, indices


def score_pairs(
    model: torch.nn.Module,
    q_store: EmbeddingStore,
    c_store: EmbeddingStore,
    tasks: Sequence[Tuple[int, int]],
    contact_batch_size: int = 64,
    device: Union[str, torch.device] = "cuda",
    desc: str = "Contact prediction",
    raw_scores_path: Optional[str] = None,
    is_host_list: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """
    Computes residue-level contact scores for a list of candidate pairs (q_idx, c_idx).
    Returns a 1D numpy array of contact scores aligned with `tasks`.
    Optionally streams raw candidate scores to a Parquet file.
    """
    n_tasks = len(tasks)
    if n_tasks == 0:
        return np.empty(0, dtype=np.float32)

    scores_out = np.zeros(n_tasks, dtype=np.float32)
    autocast_device = str(device).split(":")[0] if "cuda" in str(device) else "cpu"
    autocast_dtype = torch.float16 if "cuda" in str(device) else torch.bfloat16

    parquet_rows = [] if raw_scores_path else None

    with torch.no_grad():
        for i in tqdm(range(0, n_tasks, contact_batch_size), desc=desc, unit="batch"):
            batch = tasks[i : i + contact_batch_size]
            q_idxs = [q for q, _ in batch]
            c_idxs = [c for _, c in batch]

            pad_q, mask_q = q_store.get_padded(q_idxs, device)
            pad_c, mask_c = c_store.get_padded(c_idxs, device)

            with torch.autocast(autocast_device, dtype=autocast_dtype):
                logits, valid_mask = model.predict_contacts(pad_q, pad_c, mask_q, mask_c)
                # Exclude padded cells before taking max, identical to FlashPPI forward
                logits = logits.float().masked_fill(~valid_mask, float("-inf"))
                batch_scores = torch.sigmoid(logits.flatten(1).max(dim=-1).values).cpu().numpy()

            scores_out[i : i + len(batch)] = batch_scores

            if parquet_rows is not None:
                for k, (q, c) in enumerate(batch):
                    row_data = {
                        "q_idx": q,
                        "c_idx": c,
                        "query_id": q_store.ids[q],
                        "match_id": c_store.ids[c],
                        "contact_score": float(batch_scores[k]),
                    }
                    if is_host_list is not None:
                        row_data["is_host"] = bool(is_host_list[i + k])
                    parquet_rows.append(row_data)

            del pad_q, pad_c, mask_q, mask_c, logits, valid_mask

    if raw_scores_path and parquet_rows:
        os.makedirs(os.path.dirname(os.path.abspath(raw_scores_path)), exist_ok=True)
        raw_df = pd.DataFrame(parquet_rows)
        raw_df.to_parquet(raw_scores_path, index=False)
        print(f"Saved {len(raw_df)} candidate raw scores to {raw_scores_path}")

    return scores_out
