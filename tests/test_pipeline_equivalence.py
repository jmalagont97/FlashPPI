import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from flashppi.embed_store import EmbeddingStore, write_embedding_store
from flashppi.pipeline import faiss_topk, score_pairs


class DummyEquivalenceModel(nn.Module):
    """
    Deterministic model implementing encode_protein, head_q, head_k, and predict_contacts.
    """
    def __init__(self, plm_dim=32, clip_dim=16):
        super().__init__()
        self.plm_dim = plm_dim
        self.clip_dim = clip_dim
        # Fixed linear projections for contacts
        torch.manual_seed(42)
        self.q_proj = nn.Linear(plm_dim, plm_dim, bias=False)
        self.k_proj = nn.Linear(plm_dim, plm_dim, bias=False)

    def encode_protein(self, input_ids, attention_mask):
        B, L = input_ids.shape
        # Distinct pattern per sequence based on token values
        base = input_ids.unsqueeze(-1).float() * 0.05
        return base.expand(B, L, self.plm_dim)

    def head_q(self, res_embed, attention_mask):
        pooled = (res_embed * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1)
        return pooled[:, : self.clip_dim]

    def head_k(self, res_embed, attention_mask):
        return self.head_q(res_embed, attention_mask) * 0.8

    def predict_contacts(self, embed1, embed2, mask1, mask2):
        # Multi-head dot product simulation
        q = self.q_proj(embed1)
        k = self.k_proj(embed2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.plm_dim ** -0.5)
        valid_mask = (mask1.unsqueeze(2) * mask2.unsqueeze(1)).bool()
        return logits, valid_mask


class DummyEquivalenceTokenizer:
    def __call__(self, seqs, padding=True, truncation=True, max_length=1024, return_tensors=None):
        encoded = []
        for s in seqs:
            tokens = [ord(c) % 25 + 1 for c in s[:max_length]]
            encoded.append(tokens)

        if not padding:
            return {"input_ids": encoded}

        max_l = max(len(toks) for toks in encoded) if encoded else 0
        input_ids = []
        attention_mask = []
        for toks in encoded:
            pad_len = max_l - len(toks)
            input_ids.append(toks + [0] * pad_len)
            attention_mask.append([1] * len(toks) + [0] * pad_len)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def legacy_pipeline_score(model, tokenizer, sequences, device, batch_size=2, top_k=3):
    """Reference implementation matching original predict_proteome.py logic."""
    query_embeds, key_embeds, residue_embeds_list, seq_lengths = [], [], [], []
    autocast_device = str(device).split(":")[0] if "cuda" in str(device) else "cpu"
    autocast_dtype = torch.float16 if "cuda" in str(device) else torch.bfloat16

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.autocast(autocast_device, dtype=autocast_dtype):
                res_embed = model.encode_protein(inputs["input_ids"], inputs["attention_mask"])
                query_embeds.append(model.head_q(res_embed, inputs["attention_mask"]).cpu().float().numpy())
                key_embeds.append(model.head_k(res_embed, inputs["attention_mask"]).cpu().float().numpy())

            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for j, seq_len in enumerate(lengths):
                residue_embeds_list.append(res_embed[j, : int(seq_len), :])
                seq_lengths.append(int(seq_len))

    query_embeds = np.concatenate(query_embeds, axis=0)
    key_embeds = np.concatenate(key_embeds, axis=0)
    residue_embeds = torch.cat(residue_embeds_list, dim=0)

    cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.long, device=device)
    cu_seqlens[1:] = torch.tensor(seq_lengths, dtype=torch.long, device=device).cumsum(0)

    _, I = faiss_topk(query_embeds, key_embeds, min(top_k + 1, len(sequences)))
    inference_tasks = [(q, int(c)) for q in range(len(sequences)) for c in I[q] if c >= 0 and c != q]

    legacy_scores = []
    with torch.no_grad():
        for i in range(0, len(inference_tasks), batch_size):
            batch = inference_tasks[i : i + batch_size]
            q_batch = [residue_embeds[cu_seqlens[q] : cu_seqlens[q + 1]] for q, _ in batch]
            c_batch = [residue_embeds[cu_seqlens[c] : cu_seqlens[c + 1]] for _, c in batch]

            pad_q = pad_sequence(q_batch, batch_first=True)
            pad_c = pad_sequence(c_batch, batch_first=True)
            mask_q = (pad_q.abs().sum(dim=-1) != 0).long()
            mask_c = (pad_c.abs().sum(dim=-1) != 0).long()

            with torch.autocast(autocast_device, dtype=autocast_dtype):
                logits, valid_mask = model.predict_contacts(pad_q, pad_c, mask_q, mask_c)
                logits = logits.float().masked_fill(~valid_mask, float("-inf"))
                batch_scores = torch.sigmoid(logits.flatten(1).max(dim=-1).values).cpu().numpy()

            legacy_scores.extend(batch_scores.tolist())

    return inference_tasks, np.array(legacy_scores, dtype=np.float32)


class TestPipelineEquivalence(unittest.TestCase):
    def test_staged_vs_legacy_equivalence(self):
        device = "cpu"
        model = DummyEquivalenceModel(plm_dim=32, clip_dim=16).to(device).eval()
        tokenizer = DummyEquivalenceTokenizer()

        sequences = [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR",
            "MSKQPSDVSSECDREGRQLQPAERPPQLRPGAPTSLQTEPQGNPEGNHGGEGDSCPHGSPQGPLAPPASPGPFATRSP",
            "MAEQVALSRTQVCGILREELFQGDAFHQSDTHIFIIMGASGDLAKKKIYPTIWWLFRDGLLPENTFIVGYARSRLTV",
            "ACDEFGHIKLMNPQRSTVWY",
            "MKLLV",
            "LSTVWYACDEFG",
        ]
        ids = [f"P_{i}" for i in range(len(sequences))]

        # 1. Run legacy pipeline
        legacy_tasks, legacy_scores = legacy_pipeline_score(
            model=model,
            tokenizer=tokenizer,
            sequences=sequences,
            device=device,
            batch_size=2,
            top_k=3,
        )

        # 2. Run new staged pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "equiv.flashppi.h5")
            write_embedding_store(
                h5_path=h5_path,
                ids=ids,
                sequences=sequences,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=2,
                max_len=100,
                model_name="dummy_equiv",
            )

            with EmbeddingStore(h5_path) as store:
                _, I = faiss_topk(store.q_embeds, store.k_embeds, min(3 + 1, len(sequences)))
                staged_tasks = [(q, int(c)) for q in range(len(sequences)) for c in I[q] if c >= 0 and c != q]

                # Tasks must match exactly in order and contents
                self.assertEqual(legacy_tasks, staged_tasks)

                staged_scores = score_pairs(
                    model=model,
                    q_store=store,
                    c_store=store,
                    tasks=staged_tasks,
                    contact_batch_size=2,
                    device=device,
                )

                # Scores must agree closely (max diff < 1e-3)
                max_diff = np.max(np.abs(legacy_scores - staged_scores))
                print(f"Max absolute score difference between legacy and staged: {max_diff:.6f}")
                self.assertLess(max_diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
