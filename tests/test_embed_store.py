import os
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

from flashppi.embed_store import (
    EmbeddingStore,
    default_cache_path,
    sequence_md5,
    validate_compatible,
    write_embedding_store,
)


class DummyTokenizer:
    def __call__(self, seqs, padding=True, truncation=True, max_length=1024, return_tensors=None):
        # Maps each amino acid char to char code + 1
        encoded = []
        for s in seqs:
            tokens = [ord(c) % 50 + 1 for c in s[:max_length]]
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


class DummyModel(nn.Module):
    def __init__(self, plm_dim=64, clip_dim=32):
        super().__init__()
        self.plm_dim = plm_dim
        self.clip_dim = clip_dim

    def encode_protein(self, input_ids, attention_mask):
        B, L = input_ids.shape
        # Deterministic embeddings based on token id
        res = (input_ids.unsqueeze(-1).float() * 0.1).expand(B, L, self.plm_dim)
        return res

    def head_q(self, res_embed, attention_mask):
        B = res_embed.shape[0]
        # Pool and project to clip_dim
        pooled = (res_embed * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1)
        return pooled[:, : self.clip_dim]

    def head_k(self, res_embed, attention_mask):
        return self.head_q(res_embed, attention_mask) * 0.5


class InterruptedDummyModel(DummyModel):
    def __init__(self, plm_dim=64, clip_dim=32, fail_after_calls=1):
        super().__init__(plm_dim, clip_dim)
        self.calls = 0
        self.fail_after_calls = fail_after_calls

    def encode_protein(self, input_ids, attention_mask):
        self.calls += 1
        if self.calls > self.fail_after_calls:
            raise RuntimeError("Simulated interruption during encoding batch")
        return super().encode_protein(input_ids, attention_mask)


class TestEmbedStore(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.tokenizer = DummyTokenizer()
        self.model = DummyModel(plm_dim=64, clip_dim=32)
        self.seqs = [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR",
            "MSKQPSDVSSECDREGRQLQPAERPPQLRPGAPTSLQTEPQGNPEGNHGGEGDSCPHGSPQGPLAPPASPGPFATRSP",
            "MAEQVALSRTQVCGILREELFQGDAFHQSDTHIFIIMGASGDLAKKKIYPTIWWLFRDGLLPENTFIVGYARSRLTV",
            "ACDEFGHIKLMNPQRSTVWY",
            "MKLLV",
        ]
        self.ids = [f"prot_{i}" for i in range(len(self.seqs))]

    def test_roundtrip_and_padded_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "test.flashppi.h5")

            write_embedding_store(
                h5_path=h5_path,
                ids=self.ids,
                sequences=self.seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                batch_size=2,
                max_len=100,
                model_name="dummy_test_model",
            )

            self.assertTrue(os.path.exists(h5_path))

            with EmbeddingStore(h5_path, cache_gb=1.0) as store:
                self.assertEqual(len(store), len(self.seqs))
                self.assertEqual(store.ids, self.ids)

                for i, seq in enumerate(self.seqs):
                    res = store.get(i)
                    self.assertEqual(res.dtype, torch.float16)
                    self.assertEqual(res.shape[0], len(seq))
                    self.assertEqual(res.shape[1], 64)

                # Test get_padded
                batch_idxs = [0, 4, 3]  # lens: 78, 5, 20
                pad, mask = store.get_padded(batch_idxs, self.device)
                self.assertEqual(pad.shape, (3, 78, 64))
                self.assertEqual(mask.shape, (3, 78))
                self.assertEqual(mask[0].sum().item(), 78)
                self.assertEqual(mask[1].sum().item(), 5)
                self.assertEqual(mask[2].sum().item(), 20)
                # Verify padded region is zero
                self.assertTrue(torch.all(pad[1, 5:] == 0.0))

    def test_resuming_interrupted_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "resume.flashppi.h5")

            failing_model = InterruptedDummyModel(plm_dim=64, clip_dim=32, fail_after_calls=1)

            # First run should fail halfway
            with self.assertRaises(RuntimeError):
                write_embedding_store(
                    h5_path=h5_path,
                    ids=self.ids,
                    sequences=self.seqs,
                    model=failing_model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    batch_size=2,
                    max_len=100,
                    model_name="dummy_test_model",
                )

            # At this point, complete attribute should be False
            with self.assertRaises(ValueError):
                EmbeddingStore(h5_path)

            # Resume with normal model
            write_embedding_store(
                h5_path=h5_path,
                ids=self.ids,
                sequences=self.seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                batch_size=2,
                max_len=100,
                model_name="dummy_test_model",
            )

            # Now store should be valid and complete
            with EmbeddingStore(h5_path) as store:
                self.assertEqual(len(store), len(self.seqs))
                self.assertTrue(store.attrs["complete"])

    def test_validation_and_compatibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "compat.flashppi.h5")

            write_embedding_store(
                h5_path=h5_path,
                ids=self.ids,
                sequences=self.seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                batch_size=2,
                max_len=100,
                model_name="dummy_test_model",
            )

            # Model mismatch
            errs = validate_compatible(h5_path, model_name="other_model", max_len=100)
            self.assertTrue(any("Model mismatch" in e for e in errs))

            # max_len mismatch
            errs = validate_compatible(h5_path, model_name="dummy_test_model", max_len=512)
            self.assertTrue(any("max_len mismatch" in e for e in errs))

            # Sequence hash mismatch
            altered_md5s = [sequence_md5(s) for s in self.seqs]
            altered_md5s[0] = "wrong_md5"
            errs = validate_compatible(
                h5_path, model_name="dummy_test_model", max_len=100, md5s=altered_md5s
            )
            self.assertTrue(any("Sequence hashes" in e for e in errs))

    def test_lru_cache_without_preload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "lru.flashppi.h5")

            write_embedding_store(
                h5_path=h5_path,
                ids=self.ids,
                sequences=self.seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                batch_size=2,
                max_len=100,
                model_name="dummy_test_model",
            )

            # Force preload=False and very small cache
            store = EmbeddingStore(h5_path, cache_gb=0.00001, preload=False)
            self.assertIsNone(store._full_residues)

            for i in range(len(self.seqs)):
                res = store.get(i)
                self.assertEqual(res.shape[0], len(self.seqs[i]))

            self.assertLessEqual(store._cache_bytes, store._max_cache_bytes + 10000)
            store.close()


if __name__ == "__main__":
    unittest.main()
