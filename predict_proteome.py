#!/usr/bin/env python3
"""
FlashPPI: Fast Proteome-wide PPI Screening (Staged with HDF5 caching)

Predicts protein-protein interactions within a single proteome.
Computes and caches embeddings into HDF5 (fp16), maintaining minimal and constant VRAM usage.

Usage:
  python predict_proteome.py --fasta my_proteome.fasta --output predictions.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch

from flashppi.pipeline import ensure_embeddings, faiss_topk, load_model, score_pairs


def main():
    parser = argparse.ArgumentParser(
        description="FlashPPI: Fast Proteome-wide PPI Screening (All-vs-All with HDF5 caching)"
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default=None,
        help="Path to the input FASTA file containing the proteome.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to existing or target HDF5 embedding cache file (default: <fasta_stem>.flashppi.h5).",
    )
    parser.add_argument(
        "--overwrite_embeddings",
        action="store_true",
        help="Force recomputing and overwrite the HDF5 embedding cache.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save the output CSV. Default: predictions.csv",
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
        help="Number of nearest neighbors to retrieve per protein in stage 1.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
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
        help="Delete intermediate HDF5 embedding file after completion.",
    )

    args = parser.parse_args()

    if not args.fasta and not args.embeddings:
        parser.error("You must specify either --fasta or --embeddings.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.model_name, device)

    print("Stage 1: Ensuring protein embeddings in HDF5 cache...")
    store = ensure_embeddings(
        fasta_path=args.fasta,
        h5_path=args.embeddings,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        model_name=args.model_name,
        overwrite=args.overwrite_embeddings,
        cache_gb=args.cache_gb,
        desc="Stage 1: Encoding proteins",
    )

    if args.embed_only:
        print(f"Embeddings saved to '{store.h5_path}'. Exiting (--embed_only).")
        store.close()
        return

    n_proteins = len(store)
    if n_proteins < 2:
        store.close()
        raise ValueError("FASTA/embedding store must contain at least 2 sequences to predict interactions.")

    print(f"\nStage 1: FAISS Retrieval across {n_proteins} proteins...")
    search_k = min(args.stage1_top_k + 1, n_proteins)
    _, I = faiss_topk(store.q_embeds, store.k_embeds, search_k)

    inference_tasks = [
        (q, int(c)) for q in range(n_proteins) for c in I[q] if c >= 0 and c != q
    ]
    print(f"Found {len(inference_tasks)} candidate pairs. Starting fine-grained contact prediction...")

    contact_bs = args.contact_batch_size if args.contact_batch_size is not None else args.batch_size
    scores = score_pairs(
        model=model,
        q_store=store,
        c_store=store,
        tasks=inference_tasks,
        contact_batch_size=contact_bs,
        device=device,
        desc="Contact prediction",
        raw_scores_path=args.raw_scores,
    )

    results = []
    for k, (q_idx, c_idx) in enumerate(inference_tasks):
        if scores[k] > args.threshold:
            results.append({
                "query_id": store.ids[q_idx],
                "match_id": store.ids[c_idx],
                "contact_score": float(scores[k]),
            })

    # Deduplicate and save
    df = pd.DataFrame(results)
    if not df.empty:
        # Vectorized pair key calculation for high performance
        q_arr = df["query_id"].values
        m_arr = df["match_id"].values
        cond = q_arr < m_arr
        df["pair_key"] = list(zip(np.where(cond, q_arr, m_arr), np.where(cond, m_arr, q_arr)))

        # Keep the maximum score for each unique pair
        df = df.loc[df.groupby("pair_key")["contact_score"].idxmax()].drop(columns=["pair_key"])
        df = df.sort_values(by="contact_score", ascending=False)

        df.to_csv(args.output, index=False)
        print(f"\nSuccess! Saved {len(df)} high-confidence predictions to {args.output}")
    else:
        print(f"\nNo interactions found above the threshold of {args.threshold}.")

    h5_file_to_delete = store.h5_path if args.delete_embeddings else None
    store.close()

    if h5_file_to_delete and os.path.exists(h5_file_to_delete):
        os.remove(h5_file_to_delete)
        print(f"Cleaned up intermediate embeddings: {h5_file_to_delete}")


if __name__ == "__main__":
    main()