"""
Supervisor + Workers Pattern cho RAG Retrieval.

Architecture:
                    ┌─────────────┐
        query ─────►│  SUPERVISOR │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Worker A   │ │  Worker B   │ │  Worker C   │
    │  Semantic   │ │  Lexical    │ │    HyDE     │
    │  (ChromaDB) │ │ (BM25+TFIDF)│ │  (LLM+Emb) │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           └───────────────┼───────────────┘
                           ▼
                    ┌─────────────┐
                    │  RRF Fusion │
                    │  + Rerank   │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  PageIndex  │ (fallback nếu score thấp)
                    │  Fallback   │
                    └──────┬──────┘
                           ▼
                       top_k results

Ưu điểm so với sequential pipeline:
- Workers chạy PARALLEL → giảm latency ~60%
- Dễ thêm/bỏ worker mà không đụng logic fusion
- Supervisor tự quyết định fallback dựa trên kết quả tổng hợp
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Callable

from .task7_reranking import rerank_rrf, rerank


# =============================================================================
# WORKER DEFINITION
# =============================================================================

@dataclass
class Worker:
    """Một retrieval worker với tên và hàm search."""
    name: str
    search_fn: Callable[[str, int], list[dict]]
    enabled: bool = True
    weight: float = 1.0          # Trọng số khi fuse (dùng cho logging)
    _results: list[dict] = field(default_factory=list, repr=False)
    _elapsed: float = 0.0


# =============================================================================
# SUPERVISOR
# =============================================================================

class RetrievalSupervisor:
    """
    Supervisor điều phối các retrieval workers.

    Supervisor nhận query, phân phối đến tất cả workers chạy song song
    (ThreadPoolExecutor), thu kết quả, apply RRF fusion + rerank,
    và quyết định có cần fallback sang PageIndex không.
    """

    def __init__(
        self,
        score_threshold: float = 0.005,
        rerank_method: str = "rrf",
        use_hyde: bool = True,
        use_tfidf: bool = True,
        verbose: bool = False,
    ):
        self.score_threshold = score_threshold
        self.rerank_method = rerank_method
        self.use_hyde = use_hyde
        self.use_tfidf = use_tfidf
        self.verbose = verbose
        self._workers: list[Worker] = []
        self._build_workers()

    def _build_workers(self):
        """Khởi tạo các workers mặc định."""
        from .task5_semantic_search import semantic_search, hyde_search
        from .task6_lexical_search import lexical_search, tfidf_search

        self._workers = [
            Worker(
                name="semantic",
                search_fn=semantic_search,
                enabled=True,
                weight=1.0,
            ),
            Worker(
                name="bm25",
                search_fn=lexical_search,
                enabled=True,
                weight=1.0,
            ),
            Worker(
                name="tfidf",
                search_fn=tfidf_search,
                enabled=self.use_tfidf,
                weight=0.8,
            ),
            Worker(
                name="hyde",
                search_fn=hyde_search,
                enabled=self.use_hyde,
                weight=1.2,
            ),
        ]

    def add_worker(self, worker: Worker):
        """Thêm custom worker vào supervisor."""
        self._workers.append(worker)

    def _run_worker(self, worker: Worker, query: str, fetch_k: int) -> Worker:
        """Chạy một worker và lưu kết quả + elapsed time."""
        t0 = time.perf_counter()
        try:
            worker._results = worker.search_fn(query, fetch_k)
        except Exception as e:
            if self.verbose:
                print(f"  [Worker:{worker.name}] ERROR: {e}")
            worker._results = []
        worker._elapsed = time.perf_counter() - t0
        return worker

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Supervisor điều phối toàn bộ retrieval pipeline.

        1. Dispatch query đến tất cả enabled workers (parallel)
        2. Thu kết quả từ workers
        3. RRF fusion tất cả result lists
        4. Rerank
        5. Fallback PageIndex nếu best score < threshold

        Args:
            query: Câu truy vấn
            top_k: Số kết quả trả về

        Returns:
            List of {'content', 'score', 'metadata', 'source'}
        """
        fetch_k = top_k * 4
        active_workers = [w for w in self._workers if w.enabled]

        if self.verbose:
            print(f"\n[Supervisor] Query: {query[:60]}...")
            print(f"[Supervisor] Dispatching to {len(active_workers)} workers in parallel...")

        # ── STEP 1: Chạy workers song song ──────────────────────────────────
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(active_workers)) as executor:
            futures: dict[Future, Worker] = {
                executor.submit(self._run_worker, w, query, fetch_k): w
                for w in active_workers
            }
            for future in as_completed(futures):
                future.result()  # exceptions đã được bắt trong _run_worker

        t_parallel = time.perf_counter() - t_start

        if self.verbose:
            for w in active_workers:
                print(f"  [{w.name:10s}] {len(w._results):3d} results | {w._elapsed*1000:.0f}ms")
            print(f"  [parallel wall time] {t_parallel*1000:.0f}ms")

        # ── STEP 2: RRF Fusion ───────────────────────────────────────────────
        result_lists = [w._results for w in active_workers if w._results]
        if not result_lists:
            return self._fallback(query, top_k)

        merged = rerank_rrf(result_lists, top_k=fetch_k)
        for item in merged:
            item["source"] = "hybrid"

        # ── STEP 3: Rerank ───────────────────────────────────────────────────
        final = rerank(query, merged, top_k=top_k, method=self.rerank_method)

        # ── STEP 4: Fallback nếu cần ─────────────────────────────────────────
        best_score = final[0]["score"] if final else 0.0
        if self.verbose:
            print(f"[Supervisor] Best score after fusion: {best_score:.4f} (threshold: {self.score_threshold})")

        if best_score < self.score_threshold:
            if self.verbose:
                print("[Supervisor] Score too low → PageIndex fallback")
            return self._fallback(query, top_k) or final

        return final[:top_k]

    def _fallback(self, query: str, top_k: int) -> list[dict]:
        """Fallback sang PageIndex vectorless search."""
        try:
            from .task8_pageindex_vectorless import pageindex_search
            results = pageindex_search(query, top_k=top_k)
            if results:
                if self.verbose:
                    print(f"[Supervisor] PageIndex returned {len(results)} results")
                return results
        except Exception as e:
            if self.verbose:
                print(f"[Supervisor] PageIndex failed: {e}")
        return []


# =============================================================================
# SINGLETON — dùng chung trong pipeline
# =============================================================================

_supervisor: RetrievalSupervisor | None = None


def get_supervisor(use_hyde: bool = False, verbose: bool = False) -> RetrievalSupervisor:
    """
    Lấy supervisor singleton.
    use_hyde=False mặc định vì HyDE cần API call (tốn credit).
    """
    global _supervisor
    if _supervisor is None:
        _supervisor = RetrievalSupervisor(use_hyde=use_hyde, verbose=verbose)
    return _supervisor


def retrieve_with_supervisor(
    query: str,
    top_k: int = 5,
    use_hyde: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """
    Drop-in replacement cho retrieve() trong task9, dùng supervisor pattern.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả
        use_hyde: Có dùng HyDE worker không (cần API key)
        verbose: In log chi tiết

    Returns:
        List of {'content', 'score', 'metadata', 'source'}
    """
    sup = RetrievalSupervisor(use_hyde=use_hyde, verbose=verbose)
    return sup.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì liên quan tới ma tuý",
        "Quy trình cai nghiện bắt buộc theo pháp luật",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        results = retrieve_with_supervisor(q, top_k=3, verbose=True)
        print(f"\nTop {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] {r['content'][:80].replace(chr(10),' ')}...")
