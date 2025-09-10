import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, Any
from tqdm import tqdm  # Import tqdm if not already imported.


import httpx
import numpy as np
import optuna
from vespa.application import Vespa, VespaAsync

from config import VESPA_APP_NAME, VESPA_PORT, VESPA_URL, STUDY_NAME, STUDY_STORAGE_POSTGRES_URL, SQLITE_URL
from repository import QueryRepository, QueryData
from utils import configure_logger, process_single_result, QueryResult, RankingMetrics
import logging
logger = configure_logger()

@dataclass
class StudyResults:
    best_weights: Dict[str, Any]
    best_metrics: Dict[str, Any]
    study: optuna.study.Study


class VespaRankingOptimizer:
    """
    Evaluates queries on a Vespa app. When running in parallel (n_parallel > 1),
    it delegates per-trial evaluation to persistent workers.
    
    This class implements a context manager interface so that all persistent
    worker processes are properly shutdown, even in the event of an exception.
    
    It also creates the Optuna study internally in its optimize() method.
    
    """
    def __init__(self, vespa_app: Vespa, repo: QueryRepository, n_parallel: int = 1):
        self.vespa_app = vespa_app
        self.repo = repo  # Dependency injected repository
        self.n_parallel = n_parallel
        self.search_fields = [
            "doc_title_embedding",
            "doc_act_type_embedding",
            "chunk_sentences_embeddings",
            "chunk_titles_embeddings"
        ]
        self.cos_sim_formulas = [
            f"({{targetHits:20}}nearestNeighbor({field},q))" for field in self.search_fields
        ]
        self.query_condition = " or ".join(self.cos_sim_formulas)
        self.worker_pool = None
        if self.n_parallel > 1:
            from worker_pool import OptimizerWorkerPool
            # Use the provided repository instance
            self.worker_pool = OptimizerWorkerPool(
                vespa_url=self.vespa_app.url,
                vespa_port=self.vespa_app.port,
                repository=self.repo,
                n_workers=self.n_parallel
            )

    def __enter__(self):
        # Allow use in a with-statement.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Ensure proper shutdown even if an exception occurs.
        self.shutdown()
        return False  # Do not suppress exceptions

    def build_query_body(self, embedding: str, weights: Dict[str, float]) -> Dict:
        return {
            "yql": f"select * from {VESPA_APP_NAME} where {self.query_condition}",
            "input.query(q)": embedding,
            "ranking": "semantic",
            "ranking.features.query(title_weight)": weights["title_weight"],
            "ranking.features.query(act_type_weight)": weights["act_type_weight"],
            "ranking.features.query(chunk_title_weight)": weights["chunk_title_weight"],
            "ranking.features.query(sentence_weight)": weights["sentence_weight"],
            "ranking.features.query(sp_fp_weight)": weights["sp_fp_weight"],
            "ranking.features.query(sp_th_weight)": weights["sp_th_weight"],
            "ranking.features.query(sp_max_sentence_weight)": weights["sp_max_sentence_weight"],
            # Page Rank
            "ranking.features.query(pagerank_weight)": weights["pagerank_weight"],
            "ranking.features.query(enable_pagerank)": 1,
            "ranking.features.query(freshness_weight)":weights["freshness_weight"], 
            "presentation.format.tensors": "short-value",
            "timeout": 40,
            "hits": 20,
            "summary": "minimal"
        }

    async def _async_worker(self, queue: asyncio.Queue,
                              weights: Dict[str, float],
                              session: VespaAsync,
                              progress_bar,
                              results: list[RankingMetrics],
                              process_logger: logging.Logger) -> None:
        while not queue.empty():
            try:
                query: QueryData = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                # Using the QueryData fields for building the query
                query_body = self.build_query_body(query.embedding, weights)
                response = await session.query(query_body)
                q_result = QueryResult(
                    query=query.query,
                    full_ids = query.full_ids,
                    scores = query.scores,
                    hits=response.hits,
                )
                metric = await asyncio.to_thread(process_single_result, q_result)
                # process_logger.debug(f"metrics is {metric}" )
                if metric:
                    results.append(metric)
                
            except Exception as e:
                process_logger.error("Error processing query %s: %s", query, e)
            finally:
                progress_bar.update(1)
                queue.task_done()

    async def evaluate_queries_async(self,
                                     queries: list[QueryData],
                                     weights: Dict[str, float],
                                     max_workers: int = 50,
                                     optimizer_id: int = 0,
                                     process_logger=logger) -> RankingMetrics:
        timeout = httpx.Timeout(connect=60.0, read=60.0, write=60.0, pool=60.0)
        results: list[RankingMetrics] = []
        process_logger.debug("Processing %d queries with %d async workers", len(queries), max_workers)

        query_queue: asyncio.Queue[QueryData] = asyncio.Queue()
        for q in queries:
            query_queue.put_nowait(q)

        async with self.vespa_app.asyncio(connections=10, timeout=timeout) as session:
            progress = tqdm(total=len(queries), desc=f"[PID:{os.getpid()}] Processing")
            workers = [
                asyncio.create_task(self._async_worker(query_queue, weights, session, progress, results, process_logger))
                for _ in range(max_workers)
            ]
            await query_queue.join()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            progress.close()

        if results:
            return RankingMetrics(
                mrr=np.mean([m.mrr for m in results], dtype=float),
                recall_at_1=np.mean([m.recall_at_1 for m in results], dtype=float),
                recall_at_3=np.mean([m.recall_at_3 for m in results], dtype=float),
                recall_at_5=np.mean([m.recall_at_5 for m in results], dtype=float),
                ndcg_at_5=np.mean([m.ndcg_at_5 for m in results], dtype=float),
                ndcg_at_10=np.mean([m.ndcg_at_10 for m in results], dtype=float)
            )
        else:
            return RankingMetrics(0, 0, 0, 0, 0, 0)

    def objective(self, trial: optuna.Trial) -> float:
        trial_start = time.perf_counter()
        logger.debug("Starting trial #%d", trial.number)

        raw_weights = {
            "title_weight": trial.suggest_float("title_weight",  0.0, 1, step=0.001),
            "act_type_weight": trial.suggest_float("act_type_weight",  0.0, 1, step=0.001),
            "chunk_title_weight": trial.suggest_float("chunk_title_weight",  0.0, 1, step=0.001),
            "sentence_weight": trial.suggest_float("sentence_weight", 0.0, 1, step=0.001),
            "sp_fp_weight": trial.suggest_float("sp_fp_weight", 0.0, 1, step=0.001),
            "sp_th_weight": trial.suggest_float("sp_th_weight", 0.0, 1, step=0.001),
            "sp_max_sentence_weight": trial.suggest_float("sp_max_sentence_weight", 0.0, 1, step=0.001),
            "pagerank_weight": trial.suggest_float("pagerank_weight", 0.0, 1, step=0.001),
            "freshness_weight": trial.suggest_float("freshness_weight", 0.0, 1, step=0.001)
        }
        total = sum(raw_weights.values())
        weights = {k: v / total for k, v in raw_weights.items()}
        logger.debug("Normalized weights: %s", weights)

        if self.worker_pool:
            chunk_metrics = self.worker_pool.run_trial(weights, trial.number)
        else:
            total_queries = self.repo.get_query_count()
            if total_queries <= 0:
                raise ValueError("No queries found in the database.")
            queries = self.repo.fetch_queries(0, total_queries)
            logger.info("Fetched %d queries for single-worker evaluation", len(queries))
            chunk_metrics = [asyncio.run(self.evaluate_queries_async(queries, weights, optimizer_id=0))]

        agg_metrics = RankingMetrics(
            mrr=np.mean([m.mrr for m in chunk_metrics], dtype=float),
            recall_at_1=np.mean([m.recall_at_1 for m in chunk_metrics], dtype=float),
            recall_at_3=np.mean([m.recall_at_3 for m in chunk_metrics], dtype=float),
            recall_at_5=np.mean([m.recall_at_5 for m in chunk_metrics], dtype=float),
            ndcg_at_5=np.mean([m.ndcg_at_5 for m in chunk_metrics], dtype=float),
            ndcg_at_10=np.mean([m.ndcg_at_10 for m in chunk_metrics], dtype=float)
        )
        for key in agg_metrics.__dataclass_fields__.keys():
            trial.set_user_attr(key, getattr(agg_metrics, key))

        objective_value = (
            0.4 * agg_metrics.mrr +
            0.3 * agg_metrics.recall_at_1 +
            0.2 * agg_metrics.ndcg_at_5 +
            0.1 * agg_metrics.recall_at_3
        )
        logger.debug("Trial #%d objective score: %.4f", trial.number, objective_value)
        logger.debug("Trial #%d completed in %.4f seconds", trial.number, time.perf_counter() - trial_start)
        return objective_value

    def optimize(self, n_trials: int = 100) -> StudyResults:

        def gamma(x: int):
            return min(x, 30)

            # min(int(np.ceil(0.1 * x)), 25)

        logger.debug("Creating study and starting optimization with %d trials", n_trials)
        study = optuna.create_study(
            study_name=STUDY_NAME,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=50,
                n_ei_candidates=48,
                gamma=gamma
                ),
            load_if_exists=True,
            storage=STUDY_STORAGE_POSTGRES_URL
        )
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=True,
        )
        logger.info("Best trial #%d with score %.4f", study.best_trial.number, study.best_trial.value)
        logger.info("Best parameters: %s", study.best_params)
        return StudyResults(
            best_weights=study.best_trial.params,
            best_metrics={k: study.best_trial.user_attrs[k] for k in RankingMetrics.__annotations__.keys()},
            study=study
        )

    def shutdown(self):
        if self.worker_pool:
            logger.info("Shutting down persistent worker pool.")
            self.worker_pool.shutdown()


# ===================== Main Entry Point =====================
def main():
    start_time = time.time()
    vespa_client = Vespa(url=VESPA_URL, port=VESPA_PORT)
    parallel_processes = os.cpu_count() or 1

    # Create the repository externally (dependency injection)
    repository = QueryRepository(SQLITE_URL, STUDY_STORAGE_POSTGRES_URL)

    # Use the optimizer as a context manager so that shutdown is automatically called.
    with VespaRankingOptimizer(vespa_app=vespa_client, repo=repository, n_parallel=parallel_processes) as optimizer:
        study_results = optimizer.optimize(n_trials=100)
        study = study_results.study
        best_trial = study.best_trial
        logger.info("Best Trial Details:")
        logger.info("  Value: %s", best_trial.value)
        for key, value in best_trial.params.items():
            logger.info("    %s: %s", key, value)

    elapsed = time.time() - start_time
    logger.debug("Total elapsed time: %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
