import asyncio
import os
from multiprocessing import Process, Queue
from typing import List, Tuple, Dict, Any

from vespa.application import Vespa

from optimize import VespaRankingOptimizer
from utils import configure_logger, process_single_result, QueryResult, RankingMetrics
from repository import QueryRepository

class OptimizerWorkerPool:
    def __init__(self, vespa_url: str, vespa_port: int | None, repository: QueryRepository, n_workers: int):
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port
        self.repository = repository
        self.n_workers = n_workers
        self.workers: List[Tuple[Process, Queue, Queue]] = []
        self._create_workers()

    def _create_workers(self):
        total_queries = self.repository.get_query_count()
        if total_queries <= 0:
            raise ValueError("No queries found in the SQLite cache.")
        base_chunk = total_queries // self.n_workers
        remainder = total_queries % self.n_workers

        start = 0
        for i in range(self.n_workers):
            chunk_size = base_chunk + (1 if i < remainder else 0)
            in_q = Queue()
            out_q = Queue()
            vespa_app = Vespa(url=self.vespa_url, port=self.vespa_port)
            repo = QueryRepository(sqlite_url=self.repository.sqlite_url)
            proc = Process(
                target=self.worker_function,
                args=(start, chunk_size, self.vespa_url, self.vespa_port, self.repository.sqlite_url, in_q, out_q),
            )
            proc.daemon = True
            proc.start()
            self.workers.append((proc, in_q, out_q))
            start += chunk_size

    @staticmethod
    def worker_function(offset, limit,
                        vespa_url, vespa_port, sqlite_url,
                        in_queue: Queue, out_queue: Queue):
        
        vespa_app = Vespa(url=vespa_url, port=vespa_port)
        repo = QueryRepository(sqlite_url=sqlite_url)
    
        worker_id = f"Worker-{offset}"
        proc_logger = configure_logger(optimizer_id=worker_id)
        proc_logger.info("Starting worker: offset=%d, limit=%d", offset, limit)
        # Each worker creates its own repository instance
        try:
            queries = repo.fetch_queries(offset, limit)
        except Exception as e:
            proc_logger.error("Error fetching queries: %s", e)
            out_queue.put(e)
            return

        proc_logger.info("Fetched %d queries", len(queries))
        # Create an instance of the optimizer in async (single-process mode)
        optimizer_instance = VespaRankingOptimizer(vespa_app=vespa_app, repo=repo, n_parallel=1)

        while True:
            message = in_queue.get()
            if message is None:
                proc_logger.info("Shutdown signal received. Exiting worker.")
                break
            weights = message.get("weights")
            trial_number = message.get("trial_number", -1)
            proc_logger.info("Received weights for trial #%d: %s", trial_number, weights)
            try:
                metrics = asyncio.run(optimizer_instance.evaluate_queries_async(queries, weights, optimizer_id=offset, process_logger= proc_logger))
            except Exception as e:
                proc_logger.error("Error during evaluation for trial #%d: %s", trial_number, e)
                out_queue.put(e)
            else:
                proc_logger.info("Completed evaluation for trial #%d: %s", trial_number, metrics)
                out_queue.put(metrics)

    def run_trial(self, weights: Dict[str, float], trial_number: int) -> List[RankingMetrics]:
        for (_, in_q, _) in self.workers:
            in_q.put({"weights": weights, "trial_number": trial_number})
        results = []
        for (_, _, out_q) in self.workers:
            result = out_q.get()  # Blocking call.
            if isinstance(result, Exception):
                raise result
            results.append(result)
        return results

    def shutdown(self):
        for (proc, in_q, _) in self.workers:
            in_q.put(None)  # Send shutdown signal.
        for (proc, _, _) in self.workers:
            proc.join()
