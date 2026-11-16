"""Suggested pipeline entry point; TODO: implement the orchestration."""
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from .ingestion import load_pdf
from .chunking import fixed_size, by_paragraph
from .embeddings import MyEmbedder
from .store import PgVectorStore
from .evaluation import evaluate_query, summarize


def main() -> None:
    """TODO:

    1. Load the corpus and benchmark; validate at least 30 content pages.
    2. Build strategies A (fixed), B (paragraph) and C (B + metadata filter).
    3. Embed with the same model and store vectors/metadata per strategy.
    4. Run the same dev/test questions and save full top-3 traces.
    5. Report retrieval metrics and prepare samples for manual assessment.

    Follow the five dimensions in ../../BENCHMARK.md.
    """
    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    
    embedder = MyEmbedder("embeddinggemma-2:270m")
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()

    base_dir = Path(__file__).resolve().parent.parent
    pdf_path = base_dir / "data" / "corpus.pdf"
    if pdf_path.exists():
        pages = load_pdf(pdf_path, "fpt_bctn_2025", "corpus_pdf")
        if len(pages) < 30:
            print(f"Warning: Only {len(pages)} pages loaded. Expected at least 30.")

        chunks_a = []
        chunks_b = []
        for page in pages:
            chunks_a.extend(fixed_size(page, size=500, overlap=50))
            chunks_b.extend(by_paragraph(page, max_size=500))

        print(f"[*] Số lượng chunk Cấu hình A: {len(chunks_a)}")
        print(f"[*] Số lượng chunk Cấu hình B & C: {len(chunks_b)}")

        if chunks_a:
            start_a = time.time()
            vecs_a = embedder.embed_documents([getattr(c, 'text', '') for c in chunks_a])
            store.upsert("A", chunks_a, vecs_a)
            end_a = time.time()
            print(f"[*] Thời gian chạy Cấu hình A: {end_a - start_a:.2f} giây")

        if chunks_b:
            start_b = time.time()
            vecs_b = embedder.embed_documents([getattr(c, 'text', '') for c in chunks_b])
            store.upsert("B", chunks_b, vecs_b)
            store.upsert("C", chunks_b, vecs_b)
            end_b = time.time()
            print(f"[*] Thời gian chạy Cấu hình B & C: {end_b - start_b:.2f} giây")

    # benchmark_path = base_dir / "benchmark" / "benchmark.json"
    # if not benchmark_path.exists():
    #     print(f"Benchmark file not found at {benchmark_path}")
    #     return

    # with open(benchmark_path, "r", encoding="utf-8") as f:
    #     questions = json.load(f)

    # all_traces = []
    # for q in questions:
    #     q_vec = embedder.embed_query(q.get("question", ""))
        
    #     for strategy in ["A", "B", "C"]:
    #         filters = q.get("filter") if strategy == "C" else None
    #         hits = store.search(strategy, q_vec, top_k=3, filters=filters)
            
    #         eval_q = dict(q)
    #         eval_q["strategy"] = strategy
    #         trace = evaluate_query(eval_q, hits)
    #         all_traces.append(trace)

    # summary_metrics = summarize(all_traces)
    
    # out_dir = base_dir / "results"
    # out_dir.mkdir(exist_ok=True)
    
    # with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
    #     json.dump(all_traces, f, ensure_ascii=False, indent=2)
        
    # with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
    #     json.dump(summary_metrics, f, ensure_ascii=False, indent=2)
        
    # print(json.dumps(summary_metrics, indent=2))


if __name__ == "__main__":
    main()