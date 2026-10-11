#!/usr/bin/env python3
"""
FlashPPI: Cross-Proteome PPI Screening (Staged with HDF5 caching)

Predicts protein-protein interactions between two proteomes (host and viral).
Precomputes or reuses HDF5 caches for host and viral proteomes, allowing the host
proteome to be encoded once and reused across many viral screening runs with constant VRAM.

Usage:
  python predict_cross_proteome.py \
    --host_fasta host.fasta \
    --viral_fasta virus.fasta \
    --output cross_predictions.csv
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from flashppi.pipeline import ensure_embeddings, faiss_topk, load_model


def score_cross_pairs(
    model: torch.nn.Module,
    viral_store,
    host_store,
    inference_tasks,
    n_host: int,
    contact_batch_size: int = 64,
    device: str = "cuda",
    desc: str = "Contact prediction",
    raw_scores_path: str = None,
):
    """
    Computes contact scores for cross-proteome candidate pairs.
    Candidate target index `c` maps to host_store if c < n_host, or viral_store if c >= n_host.
    """
    n_tasks = len(inference_tasks)
    if n_tasks == 0:
        return np.empty(0, dtype=np.float32)

    scores_out = np.zeros(n_tasks, dtype=np.float32)
    autocast_device = str(device).split(":")[0] if "cuda" in str(device) else "cpu"
    autocast_dtype = torch.float16 if "cuda" in str(device) else torch.bfloat16

    parquet_rows = [] if raw_scores_path else None
    plm_dim = viral_store.plm_dim

    with torch.no_grad():
        for i in tqdm(range(0, n_tasks, contact_batch_size), desc=desc, unit="batch"):
            batch = inference_tasks[i : i + contact_batch_size]
            B = len(batch)

            # Query is always viral
            q_idxs = [q for q, _, _ in batch]
            pad_q, mask_q = viral_store.get_padded(q_idxs, device)

            # Candidate c can be from host or viral
            c_res_list = []
            c_lens = []
            for _, c, is_host in batch:
                if is_host:
                    res = host_store.get(c)
                    c_lens.append(int(host_store.lengths[c]))
                else:
                    v_idx = c - n_host
                    res = viral_store.get(v_idx)
                    c_lens.append(int(viral_store.lengths[v_idx]))
                c_res_list.append(res)

            max_L_c = max(c_lens) if c_lens else 0
            pad_c = torch.zeros((B, max_L_c, plm_dim), dtype=torch.float16)
            mask_c = torch.zeros((B, max_L_c), dtype=torch.long)
            for row in range(B):
                L = c_lens[row]
                if L > 0:
                    pad_c[row, :L, :] = c_res_list[row]
                    mask_c[row, :L] = 1

            non_blocking = "cuda" in str(device)
            pad_c = pad_c.to(device, non_blocking=non_blocking)
            mask_c = mask_c.to(device, non_blocking=non_blocking)

            with torch.autocast(autocast_device, dtype=autocast_dtype):
                logits, valid_mask = model.predict_contacts(pad_q, pad_c, mask_q, mask_c)
                logits = logits.float().masked_fill(~valid_mask, float("-inf"))
                batch_scores = torch.sigmoid(logits.flatten(1).max(dim=-1).values).cpu().numpy()

            scores_out[i : i + B] = batch_scores

            if parquet_rows is not None:
                for k, (q, c, is_host) in enumerate(batch):
                    match_id = host_store.ids[c] if is_host else viral_store.ids[c - n_host]
                    parquet_rows.append({
                        "q_idx": q,
                        "c_idx": c,
                        "viral_id": viral_store.ids[q],
                        "match_id": match_id,
                        "contact_score": float(batch_scores[k]),
                        "is_host": bool(is_host),
                    })

            del pad_q, pad_c, mask_q, mask_c, logits, valid_mask

    if raw_scores_path and parquet_rows:
        os.makedirs(os.path.dirname(os.path.abspath(raw_scores_path)), exist_ok=True)
        raw_df = pd.DataFrame(parquet_rows)
        raw_df.to_parquet(raw_scores_path, index=False)
        print(f"Saved {len(raw_df)} candidate raw scores to {raw_scores_path}")

    return scores_out


def main():
    parser = argparse.ArgumentParser(
        description="FlashPPI: Cross-Proteome PPI Screening between two organisms (with HDF5 caching)"
    )
    parser.add_argument(
        "--host_fasta",
        type=str,
        default=None,
        help="Path to the host proteome FASTA file (keys).",
    )
    parser.add_argument(
        "--viral_fasta",
        type=str,
        default=None,
        help="Path to the viral/pathogen proteome FASTA file (queries).",
    )
    parser.add_argument(
        "--host_embeddings",
        type=str,
        default=None,
        help="Path to existing or target host HDF5 embedding file.",
    )
    parser.add_argument(
        "--viral_embeddings",
        type=str,
        default=None,
        help="Path to existing or target viral HDF5 embedding file.",
    )
    parser.add_argument(
        "--overwrite_embeddings",
        action="store_true",
        help="Force recomputing and overwrite the HDF5 embedding caches.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cross_predictions.csv",
        help="Path to save the output CSV. Default: cross_predictions.csv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tattabio/flashppi",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--stage1_top_k",
        type=int,
        default=100,
        help="Number of nearest neighbors to retrieve per viral protein in stage 1.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Contact score threshold to keep predictions.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for protein encoding (stage 1).",
    )
    parser.add_argument(
        "--contact_batch_size",
        type=int,
        default=None,
        help="Batch size for contact prediction (stage 2). Default: same as --batch_size.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--cache_gb",
        type=float,
        default=4.0,
        help="RAM cache budget in GB for residue embeddings.",
    )
    parser.add_argument(
        "--raw_scores",
        type=str,
        default=None,
        help="Optional path to save all raw candidate pair scores as a Parquet file.",
    )
    parser.add_argument(
        "--embed_only",
        action="store_true",
        help="Only run stage 1 (compute and cache embeddings into HDF5), then exit.",
    )
    parser.add_argument(
        "--delete_embeddings",
        action="store_true",
        help="Delete intermediate HDF5 embedding files after completion.",
    )

    args = parser.parse_args()

    if not args.host_fasta and not args.host_embeddings:
        parser.error("You must specify either --host_fasta or --host_embeddings.")
    if not args.viral_fasta and not args.viral_embeddings:
        parser.error("You must specify either --viral_fasta or --viral_embeddings.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.model_name, device)

    # Stage 1: Ensure both proteome embeddings in HDF5 caches
    print("\nStage 1: Ensuring host proteome embeddings...")
    host_store = ensure_embeddings(
        fasta_path=args.host_fasta,
        h5_path=args.host_embeddings,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        model_name=args.model_name,
        overwrite=args.overwrite_embeddings,
        cache_gb=max(1.0, args.cache_gb * 0.75),
        desc="Stage 1: Encoding host",
    )

    print("\nStage 1: Ensuring viral proteome embeddings...")
    viral_store = ensure_embeddings(
        fasta_path=args.viral_fasta,
        h5_path=args.viral_embeddings,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        model_name=args.model_name,
        overwrite=args.overwrite_embeddings,
        cache_gb=max(1.0, args.cache_gb * 0.25),
        desc="Stage 1: Encoding viral",
    )

    if args.embed_only:
        print(f"Host embeddings saved to '{host_store.h5_path}'.")
        print(f"Viral embeddings saved to '{viral_store.h5_path}'.")
        print("Exiting (--embed_only).")
        host_store.close()
        viral_store.close()
        return

    n_host = len(host_store)
    n_viral = len(viral_store)
    print(f"\nHost proteome: {n_host} proteins | Viral proteome: {n_viral} proteins")

    # Build combined key space with host and viral keys
    combined_k_embeds = np.concatenate([host_store.k_embeds, viral_store.k_embeds], axis=0)

    print(
        f"\nFAISS Retrieval (combined key space: {n_host} host + {n_viral} viral = {len(combined_k_embeds)})..."
    )
    search_k = min(args.stage1_top_k + 1, len(combined_k_embeds))
    _, I = faiss_topk(viral_store.q_embeds, combined_k_embeds, search_k)

    # Build inference tasks, skipping self-hits (viral protein q_idx → combined index n_host + q_idx)
    inference_tasks = []  # (viral_idx, combined_idx, is_host)
    for q_idx in range(n_viral):
        for c_idx in I[q_idx]:
            if c_idx < 0 or c_idx == n_host + q_idx:
                continue
            inference_tasks.append((q_idx, int(c_idx), bool(c_idx < n_host)))

    print(f"Found {len(inference_tasks)} candidate pairs. Starting contact prediction...")

    contact_bs = args.contact_batch_size if args.contact_batch_size is not None else args.batch_size
    scores = score_cross_pairs(
        model=model,
        viral_store=viral_store,
        host_store=host_store,
        inference_tasks=inference_tasks,
        n_host=n_host,
        contact_batch_size=contact_bs,
        device=device,
        desc="Contact prediction",
        raw_scores_path=args.raw_scores,
    )

    # Group by viral protein: pick best host match, annotate host_is_best_contact
    query_predictions = defaultdict(list)
    for k, (q_idx, c_idx, is_host) in enumerate(inference_tasks):
        query_predictions[q_idx].append((c_idx, float(scores[k]), is_host))

    results = []
    for q_idx, preds in query_predictions.items():
        host_preds = [(c, s) for c, s, h in preds if h]
        if not host_preds:
            continue
        best_host_c_idx, best_host_score = max(host_preds, key=lambda x: x[1])
        if best_host_score <= args.threshold:
            continue
        host_is_best_contact = max(preds, key=lambda x: x[1])[2]
        results.append({
            "viral_id": viral_store.ids[q_idx],
            "host_id": host_store.ids[best_host_c_idx],
            "contact_score": best_host_score,
            "host_is_best_contact": host_is_best_contact,
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("contact_score", ascending=False)
        df.to_csv(args.output, index=False)
        print(f"\nSuccess! Saved {len(df)} predictions to {args.output}")
    else:
        print(f"\nNo cross-proteome interactions found above the threshold of {args.threshold}.")

    h_path_del = host_store.h5_path if args.delete_embeddings else None
    v_path_del = viral_store.h5_path if args.delete_embeddings else None
    host_store.close()
    viral_store.close()

    if h_path_del and os.path.exists(h_path_del):
        os.remove(h_path_del)
    if v_path_del and os.path.exists(v_path_del):
        os.remove(v_path_del)


if __name__ == "__main__":
    main()
