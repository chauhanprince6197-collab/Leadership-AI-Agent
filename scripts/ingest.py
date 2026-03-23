#!/usr/bin/env python3
"""
scripts/ingest.py — CLI document ingestion tool

Usage:
    python scripts/ingest.py --folder ./documents
    python scripts/ingest.py --file ./reports/q4_2024.pdf
    python scripts/ingest.py --stats
    python scripts/ingest.py --delete annual_report_2024.txt
"""

import argparse, sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("ANTHROPIC_API_KEY", "not-needed-for-ingestion")

from app.ingestion.pipeline import ingest_file, ingest_folder, LOADER_MAP
from app.retrieval.store import HybridVectorStore
from config.settings import settings


def get_store():
    return HybridVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
    )


def show_stats(store):
    s = store.get_stats()
    print(f"\n── Vector Store ──────────────────────────────────")
    print(f"  Chunks:     {s['total_chunks']}")
    print(f"  Documents:  {s['total_documents']}")
    print(f"  Doc types:  {', '.join(s['doc_types']) or 'none'}")
    print(f"  Persist:    {settings.chroma_persist_dir}")
    for doc in sorted(s['documents']):
        print(f"    • {doc}")
    print()


def main():
    p = argparse.ArgumentParser(description="Leadership Agent — Ingestion CLI")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--folder", type=Path, metavar="DIR")
    g.add_argument("--file",   type=Path, metavar="FILE")
    g.add_argument("--stats",  action="store_true")
    g.add_argument("--delete", type=str, metavar="SOURCE_NAME")
    args = p.parse_args()

    store = get_store()

    if args.stats:
        show_stats(store)

    elif args.delete:
        n = store.delete_by_source(args.delete)
        print(f"Deleted {n} chunks for '{args.delete}'" if n else "Not found.")

    elif args.file:
        if not args.file.exists():
            print(f"File not found: {args.file}"); sys.exit(1)
        t = time.time()
        chunks = ingest_file(args.file, chunk_size=settings.chunk_size,
                             chunk_overlap=settings.chunk_overlap)
        store.add_chunks(chunks)
        print(f"✓ {len(chunks)} chunks in {time.time()-t:.1f}s")
        show_stats(store)

    elif args.folder:
        supported = set(LOADER_MAP.keys())
        files = [f for f in args.folder.rglob("*")
                 if f.is_file() and f.suffix.lower() in supported]
        print(f"Found {len(files)} files in {args.folder}")
        total = 0
        t0 = time.time()
        for fp in files:
            try:
                chunks = ingest_file(fp, chunk_size=settings.chunk_size,
                                     chunk_overlap=settings.chunk_overlap)
                store.add_chunks(chunks)
                print(f"  ✓ {fp.name:<50} {len(chunks):>4} chunks")
                total += len(chunks)
            except Exception as e:
                print(f"  ✗ {fp.name:<50} FAILED: {e}")
        print(f"\nDone — {total} chunks in {time.time()-t0:.1f}s")
        show_stats(store)


if __name__ == "__main__":
    main()
