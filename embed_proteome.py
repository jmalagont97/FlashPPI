#!/usr/bin/env python3
"""
FlashPPI: Standalone Proteome Embedding Tool

Encodes protein sequences from a FASTA file using FlashPPI's protein language model backbone
and saves residue embeddings (fp16) along with query and key projection vectors (fp32)
into an optimized, resumable HDF5 file.

Usage:
  python embed_proteome.py --fasta proteome.fasta [--output proteome.flashppi.h5]
"""
import argparse
import os
import torch

from flashppi.embed_store import default_cache_path, write_embedding_store
from flashppi.pipeline import load_model, read_fasta


def main():
    parser = argparse.ArgumentParser(
        description="FlashPPI: Precompute and cache proteome embeddings in HDF5 (fp16)"
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to the input FASTA file containing protein sequences.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output HDF5 file. Default: <fasta_stem>.flashppi.h5",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tattabio/flashppi",
        help="HuggingFace model ID or local directory path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for protein encoding inference.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force recomputing and overwrite existing cache file.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h5_output = args.output if args.output else default_cache_path(args.fasta)

    print(f"Input FASTA: {args.fasta}")
    print(f"Output HDF5: {h5_output}")

    ids, sequences = read_fasta(args.fasta)
    print(f"Loaded {len(sequences)} sequences from FASTA.")

    model, tokenizer = load_model(args.model_name, device)

    write_embedding_store(
        h5_path=h5_output,
        ids=ids,
        sequences=sequences,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        model_name=args.model_name,
        overwrite=args.overwrite,
        desc=f"Embedding {os.path.basename(args.fasta)}",
    )

    file_size_mb = os.path.getsize(h5_output) / (1024 * 1024)
    print(f"Done! Embedding store saved to: {h5_output} ({file_size_mb:.2f} MB)")
    if torch.cuda.is_available():
        max_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak VRAM allocated: {max_vram:.2f} MB")


if __name__ == "__main__":
    main()
