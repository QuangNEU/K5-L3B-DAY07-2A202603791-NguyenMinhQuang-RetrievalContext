"""TODO: store and search vectors using PostgreSQL + pgvector."""
from typing import Any
import psycopg
import json
from .models import Chunk

class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        with psycopg.connect(self.database_url) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        strategy VARCHAR(50),
                        document_id VARCHAR(255),
                        page INT,
                        metadata JSONB,
                        text TEXT,
                        embedding vector({self.dimensions})
                    );
                """)
            conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks or not vectors or len(chunks) != len(vectors):
            return

        document_ids = list(set(getattr(c, 'document_id', '') for c in chunks))

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                for doc_id in document_ids:
                    cur.execute(
                        "DELETE FROM document_chunks WHERE strategy = %s AND document_id = %s",
                        (strategy, doc_id)
                    )

                for chunk, vector in zip(chunks, vectors):
                    vector_str = "[" + ",".join(str(v) for v in vector) + "]"
                    
                    cur.execute("""
                        INSERT INTO document_chunks 
                        (strategy, document_id, page, metadata, text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s::vector)
                    """, (
                        strategy,
                        getattr(chunk, 'document_id', ''),
                        getattr(chunk, 'page', 0),
                        json.dumps(getattr(chunk, 'metadata', {})),
                        getattr(chunk, 'text', ''),
                        vector_str
                    ))
            conn.commit()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        results = []
        
        vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
        
        query = """
            SELECT id, document_id, page, metadata, text, 
                   (embedding <=> %s::vector) AS cosine_distance
            FROM document_chunks
            WHERE strategy = %s
        """
        params = [vector_str, strategy]

        if filters:
            for k, v in filters.items():
                if k == 'document_id':
                    query += " AND document_id = %s"
                    params.append(str(v))
                else:
                    query += " AND metadata->>%s = %s"
                    params.extend([str(k), str(v)])

        query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([vector_str, top_k])

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "id": row[0],
                        "document_id": row[1],
                        "page": row[2],
                        "metadata": row[3] if isinstance(row[3], dict) else json.loads(row[3] or '{}'),
                        "text": row[4],
                        "score": 1.0 - float(row[5])
                    })
        return results

    def get_all(self, strategy: str, document_id: str = "fpt_bctn_2025") -> list[dict[str, Any]]:
        """Lấy toàn bộ chunks của một strategy để phục vụ Lexical Retrieval (BM25)"""
        results = []
        query = """
            SELECT id, document_id, page, metadata, text
            FROM document_chunks
            WHERE strategy = %s AND document_id = %s
        """
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (strategy, document_id))
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "id": row[0],
                        "document_id": row[1],
                        "page": row[2],
                        "metadata": row[3] if isinstance(row[3], dict) else json.loads(row[3] or '{}'),
                        "text": row[4]
                    })
        return results