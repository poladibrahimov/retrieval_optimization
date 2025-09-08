import csv
import logging
import os
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple

import psycopg2
import sqlite3


@dataclass
class QueryResult:
    query: str
    expected_doc_id: str
    expected_chunk_id: str
    hits: List[Dict]


@dataclass
class RankingMetrics:
    mrr: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    ndcg_at_5: float
    ndcg_at_10: float


def _calculate_ndcg(relevance_scores: List[float], k: Optional[int] = None) -> float:
    """Calculate NDCG for a list of relevance scores"""
    if k is None:
        k = len(relevance_scores)

    dcg = sum((2**rel - 1) / np.log2(i + 2)
              for i, rel in enumerate(relevance_scores[:k]))
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = sum((2**rel - 1) / np.log2(i + 2)
               for i, rel in enumerate(ideal_relevance[:k]))

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg


def process_single_result(result: QueryResult) -> RankingMetrics:
    """
    Process a single QueryResult to compute individual metrics.

    Returns:
        A tuple containing (mrr, recall_1, recall_3, recall_5, ndcg_5, ndcg_10)
    """
    # Calculate position of correct document
    correct_pos = None
    for i, hit in enumerate(result.hits, 1):
        if (str(hit["fields"]["doc_id"]) == str(result.expected_doc_id) and
                str(hit["fields"]["chunk_id"]) == str(result.expected_chunk_id)):
            correct_pos = i
            break

    # MRR
    mrr = 1.0 / correct_pos if correct_pos else 0.0

    # Recall@K
    recall_1 = 1.0 if correct_pos == 1 else 0.0
    recall_3 = 1.0 if correct_pos and correct_pos <= 3 else 0.0
    recall_5 = 1.0 if correct_pos and correct_pos <= 5 else 0.0

    # NDCG@K
    relevance_5 = [1.0 if i < 5 and (i + 1) == correct_pos else 0.0
                   for i in range(min(5, len(result.hits)))]
    relevance_10 = [1.0 if i < 10 and (i + 1) == correct_pos else 0.0
                    for i in range(min(10, len(result.hits)))]

    ndcg_5 = _calculate_ndcg(relevance_5)
    ndcg_10 = _calculate_ndcg(relevance_10)

    return RankingMetrics(mrr, recall_1, recall_3, recall_5, ndcg_5, ndcg_10)



def configure_logger(optimizer_id: Any | None = None) -> logging.Logger:
    # Ensure logs directory exists
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    if optimizer_id is not None:
        log_filename = os.path.join(logs_dir, f"optimizer_{optimizer_id}.log")
        logger_name = f"optimizer.{optimizer_id}"
    else:
        log_filename = os.path.join(logs_dir, "optimizer.log")
        logger_name = "optimizer"
        
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Remove inherited handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # Configure its own file handler so that it logs to a separate file.
    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s [%(process)d] [%(name)s] %(levelname)s: %(message)s",
                                  datefmt="%H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add a console handler to print logs to console
    if optimizer_id is None:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False

    return logger




def fetch_queries_from_db(url: str, offset: int, limit: int) -> List[Tuple[str, str, str, str]]:
    """
    Fetch queries from the PostgreSQL database filtering for query_length '4' or 'n'.
    Note: This function does not perform logging. Logging is deferred to higher-level functions.

    Args:
        offset (int): Starting offset.
        limit (int): Maximum number of rows to fetch.
    
    Returns:
        List of tuples: Each tuple contains (query, doc_id, chunk_id, embedding).
    """
    sql = """
        SELECT query, doc_id, chunk_id, embedding
        FROM openai_queries
        WHERE query_length IN ('4', 'n')
        ORDER BY id
        LIMIT %s OFFSET %s
    """
    try:
        conn = psycopg2.connect(url)
        with conn.cursor() as cur:
            cur.execute(sql, (limit, offset))
            rows = cur.fetchall()
            return rows
    except Exception as e:
        # Errors will be raised to the caller to decide on logging
        raise e
    finally:
        if 'conn' in locals():
            conn.close()



def count_filtered_queries(url: str) -> Optional[int]:
    """
    Count the number of queries with query_length '4' or 'n'.

    Returns:
        int: The count or None if an error occurs.
    """
    sql = """
        SELECT COUNT(*)
        FROM openai_queries
        WHERE query_length IN ('4', 'n')
    """
    try:
        conn = psycopg2.connect(url)
        with conn.cursor() as cur:
            cur.execute(sql)
            count = cur.fetchone()

            if count is None:
                return 0
            else:
                return count[0]
    except Exception as e:
        # Allow caller to handle/log the exception
        raise e
    finally:
        if 'conn' in locals():
            conn.close()



def generate_sql_pagination_ranges(count, num_chunks):
    """
    Divide a total count into SQL pagination ranges and yield (offset, limit) tuples.

    Each tuple represents:
      - offset: the starting row (1-based indexing),
      - limit: the number of rows for that chunk.
    """
    boundaries = np.linspace(1, count + 1, num_chunks + 1, dtype=int)
    for i in range(num_chunks):
        yield int(boundaries[i]), int(boundaries[i+1] - boundaries[i])

def load_queries_from_csv(csv_file: str) -> List[Tuple[str, str, str, str]]:
    """
    Load queries from a CSV file filtering by query_length.
    
    Raises:
        KeyError: If any required column is missing.
    
    Returns:
        List of tuples: (query, doc_id, chunk_id, embedding).
    """
    data: List[Tuple[str, str, str, str]] = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required_fields = ["query_length", "query", "doc_id", "chunk_id", "embedding"]
        for row in reader:
            # Verify that all required keys exist in the row
            for field in required_fields:
                if field not in row:
                    raise KeyError(f"Missing required column: '{field}' in CSV file.")
                    
            if row["query_length"] in {"4", "n"}:
                data.append((
                    row["query"],
                    row["doc_id"],
                    row["chunk_id"],
                    row["embedding"]
                ))
    return data
