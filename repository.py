import io
import logging
import math
from typing import List, Optional, Iterator
from contextlib import contextmanager
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, func
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION
from sqlalchemy.types import JSON

# Create a dedicated logger for this module.
repo_logger = logging.getLogger("QueryRepository")
repo_logger.setLevel(logging.DEBUG)  # Set to DEBUG level (or as desired)
if not repo_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    repo_logger.addHandler(handler)

# Using SQLAlchemy 2.0 style declarative base.
class Base(DeclarativeBase):
    pass

# Define the ORM model for the openai_queries table.
class OpenAIQuery(Base):
    __tablename__ = 'optuna_v1'
    id = Column(Integer, primary_key=True)
    query = Column(String, nullable=False)
    query_length = Column(String, nullable=False)  # e.g., "4", "n"
    full_ids = Column(JSON, nullable=True)  # Comma-separated list of full document IDs
    scores = Column(JSON, nullable=True)  # Comma-separated list of scores
    embedding = Column(JSON)

# Define a namedtuple for returning query data.
QueryData = namedtuple('QueryData', ['query', 'full_ids', 'scores', 'embedding'])

class QueryRepository:
    def __init__(self, sqlite_url: str, pg_url: Optional[str] = None, filter_query: bool = True, query_lengths: List[str] = ["4", "n"]) -> None:
        self.sqlite_url = sqlite_url
        self.pg_url = pg_url
        self.filter_query = filter_query
        self.query_lengths = query_lengths 
        self.engine = create_engine(sqlite_url)
        # Ensure that the tables exist. Comment out if using Alembic migrations.
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        repo_logger.debug("Initialized QueryRepository with sqlite_url: %s, pg_url: %s", sqlite_url, pg_url)

        if pg_url:
            self.copy_from_pg() # Copy new rows from PostgreSQL to SQLite on initialization.

        
    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Provide a transactional scope around a series of operations."""
        session: Session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            repo_logger.error("Session rollback due to error: %s", e)
            raise
        finally:
            session.close()
    
    def get_query_count(self) -> int:
        with self.session_scope() as session:
            query = session.query(OpenAIQuery)
            if self.filter_query and self.query_lengths:
                query = query.filter(OpenAIQuery.query_length.in_(self.query_lengths))
            count = query.count()
            repo_logger.debug("Local query count: %d", count)
            return count

    def get_max_id(self) -> int:
        """
        Returns the maximum stored id from the local SQLite database.
        Applies filtering on query_length if enabled.
        """
        with self.session_scope() as session:
            q = session.query(func.max(OpenAIQuery.id))
            if self.filter_query and self.query_lengths:
                q = q.filter(OpenAIQuery.query_length.in_(self.query_lengths))
            max_id = q.scalar()
            repo_logger.debug("Maximum local id: %s", max_id)
            return max_id if max_id is not None else 0
    
    def fetch_queries(self, offset: int, limit: int) -> List[QueryData]:
        with self.session_scope() as session:
            query = session.query(OpenAIQuery)
            if self.filter_query and self.query_lengths:
                query = query.filter(OpenAIQuery.query_length.in_(self.query_lengths))
            query = query.order_by(OpenAIQuery.id).offset(offset).limit(limit)
            rows = query.all()
            repo_logger.debug("Fetched %d queries from local database", len(rows))
            return [
                QueryData(query=row.query, full_ids=row.full_ids, scores = row.scores, embedding=row.embedding)
                for row in rows
            ]
    
    def copy_from_pg(self, chunk_size=1000) -> None:
        """
        Copies new queries from PostgreSQL to the local SQLite database in parallel.
        It first retrieves the highest stored id in the local database and then
        only copies rows with id > max_id. If filtering is enabled, it also applies
        the query_length filter.
        
        The PostgreSQL table is assumed to have columns: id, query, doc_id, chunk_id, embedding, query_length.
        Data is fetched in chunks (of 1000 rows by default) concurrently using a thread pool.
        """
        if self.pg_url is None:
            raise ValueError("PostgreSQL URL is not provided.")
        
        max_local_id = self.get_max_id()
        repo_logger.info("Starting copy_from_pg; local max id is %d", max_local_id)
        
        # Build the WHERE clause for new rows.
        where_clauses = [f"id > {max_local_id}"]
        if self.filter_query and self.query_lengths:
            values = ", ".join(f"'{val}'" for val in self.query_lengths)
            where_clauses.append(f"query_length IN ({values})")
        where_clause = "WHERE " + " AND ".join(where_clauses)
        
        # First, get the total new row count from PostgreSQL.
        total_new = 0
        conn_pg = psycopg2.connect(self.pg_url, connect_timeout=10)
        try:
            with conn_pg.cursor() as cur:
                count_query = f"SELECT COUNT(*) FROM optuna_v1 {where_clause};"
                repo_logger.debug("Executing count query: %s", count_query.strip())
                cur.execute(count_query)
                row = cur.fetchone()
                if not row:
                    raise ValueError("No count returned from PostgreSQL.")
                total_new = row[0]  # Extract the count from the tuple.
                repo_logger.info("Total new rows to fetch: %d", total_new)
        finally:
            conn_pg.close()

        if total_new == 0:
            repo_logger.info("No new rows to copy from PostgreSQL.")
            return
        
        
        num_chunks = math.ceil(total_new / chunk_size)
        repo_logger.info("Fetching data in %d chunks (chunk size = %d)", num_chunks, chunk_size)
    
        def fetch_chunk(chunk_index: int) -> List[tuple]:
            """Fetch a chunk of rows from PostgreSQL."""
            conn = psycopg2.connect(self.pg_url, connect_timeout=10)
            try:
                with conn.cursor() as cur:
                    offset = chunk_index * chunk_size
                    # Now include the 'id' column as the first column.
                    query = f"""
                        SELECT id, query, query_length, full_ids, scores, embedding
                        FROM optuna_v1
                        {where_clause}
                        ORDER BY id
                        LIMIT {chunk_size} OFFSET {offset};
                    """
                    repo_logger.debug("Fetching chunk %d: %s", chunk_index, query.strip())
                    cur.execute(query)
                    return cur.fetchall()
            finally:
                conn.close()
        
        all_rows: List[tuple] = []
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = [executor.submit(fetch_chunk, i) for i in range(num_chunks)]
            for future in as_completed(futures):
                chunk = future.result()
                repo_logger.debug("Fetched chunk with %d rows", len(chunk))
                all_rows.extend(chunk)
        
        repo_logger.info("Total rows fetched: %d", len(all_rows))
        
        # Insert fetched rows into SQLite.
        with self.session_scope() as session:
            count = 0
            for row in all_rows:
                # row[0] is the id, row[1] is query, row[2] is doc_id, row[3] is chunk_id, row[4] is embedding, row[5] is query_length.
                new_query = OpenAIQuery(
                    id=row[0],
                    query=row[1],
                    query_length=row[2],
                    full_ids=row[3],
                    scores=row[4],
                    embedding=row[5],
                )
                session.add(new_query)
                count += 1
            repo_logger.info("Imported %d new rows into SQLite.", count)
