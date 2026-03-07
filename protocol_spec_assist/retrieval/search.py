"""
Retrieval: BGE-M3 embeddings + Qdrant hybrid (dense + sparse) + reranker.
Fully local. No external API calls.
"""

from __future__ import annotations
import uuid
from typing import Optional
from dataclasses import dataclass


DENSE_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
COLLECTION_NAME = "protocol_chunks"


@dataclass
class RetrievedChunk:
    text: str
    heading: str
    source_type: str
    page: int
    protocol_id: str
    dense_score: float
    rerank_score: Optional[float] = None
    chunk_id: Optional[str] = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.dense_score


class EmbeddingModel:
    """BGE-M3: dense + sparse embeddings."""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
                self._model = BGEM3FlagModel(DENSE_MODEL, use_fp16=True)
                print(f"[Embeddings] Loaded {DENSE_MODEL}")
            except ImportError:
                raise ImportError(
                    "FlagEmbedding not installed.\n"
                    "Run: pip install FlagEmbedding"
                )

    def encode(self, texts: list[str]) -> dict:
        """Returns dense vectors + sparse weights for hybrid indexing."""
        self._load()
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": output["dense_vecs"].tolist(),
            "sparse": output["lexical_weights"],   # list of {token: weight} dicts
        }


class Reranker:
    """BGE reranker-v2-m3: lightweight multilingual cross-encoder."""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(RERANKER_MODEL, use_fp16=True)
                print(f"[Reranker] Loaded {RERANKER_MODEL}")
            except ImportError:
                raise ImportError(
                    "FlagEmbedding not installed.\n"
                    "Run: pip install FlagEmbedding"
                )

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 8) -> list[RetrievedChunk]:
        self._load()
        pairs = [[query, c.text] for c in chunks]
        scores = self._model.compute_score(pairs, normalize=True)
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]


class ProtocolIndex:
    """
    Qdrant-backed hybrid search index.
    One collection shared across all protocols.
    Filtered by protocol_id at query time.
    """

    def __init__(self, index_dir: str = "data/index"):
        self._client = None
        self._index_dir = index_dir
        self._embedder = EmbeddingModel()
        self._reranker = Reranker()
        self._dim: Optional[int] = None

    def _load_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(path=self._index_dir)
                self._ensure_collection()
            except ImportError:
                raise ImportError(
                    "qdrant-client not installed.\n"
                    "Run: pip install qdrant-client"
                )

    def _ensure_collection(self):
        from qdrant_client.models import (
            VectorParams, Distance, SparseVectorParams, SparseIndexParams
        )
        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            print(f"[Index] Created Qdrant collection: {COLLECTION_NAME}")

    def delete_protocol(self, protocol_id: str):
        """Delete all indexed chunks for a protocol before re-indexing."""
        self._load_client()
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="protocol_id", match=MatchValue(value=protocol_id))]
            ),
        )
        print(f"[Index] Deleted existing chunks for {protocol_id}")

    def index_protocol(self, chunks: list[dict], protocol_id: str):
        """Embed and upsert all chunks for a protocol."""
        self._load_client()
        from qdrant_client.models import PointStruct, SparseVector

        # Delete existing chunks to prevent duplicates on re-index
        self.delete_protocol(protocol_id)

        print(f"[Index] Indexing {len(chunks)} chunks for {protocol_id}...")
        texts = [c["text"] for c in chunks]

        # Batch encode
        batch_size = 32
        all_dense, all_sparse = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self._embedder.encode(batch)
            all_dense.extend(encoded["dense"])
            all_sparse.extend(encoded["sparse"])

        points = []
        for chunk, dense_vec, sparse_weights in zip(chunks, all_dense, all_sparse):
            # Use deterministic chunk_id if available, else generate UUID
            point_id = chunk.get("chunk_id", str(uuid.uuid4()))

            # Convert sparse weights to Qdrant format
            sparse_idx = [int(k) for k in sparse_weights.keys()] if isinstance(sparse_weights, dict) else []
            sparse_values = [float(v) for v in sparse_weights.values()] if isinstance(sparse_weights, dict) else []

            points.append(PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_idx,
                        values=sparse_values,
                    ) if sparse_idx else SparseVector(indices=[], values=[]),
                },
                payload={
                    "protocol_id": protocol_id,
                    "text": chunk["text"],
                    "heading": chunk.get("heading", ""),
                    "source_type": chunk.get("source_type", "narrative"),
                    "page": chunk.get("page_start", 0),
                    "is_table_row": chunk.get("is_table_row", False),
                    "chunk_id": point_id,
                }
            ))

        # Upsert in batches
        for i in range(0, len(points), 100):
            self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i+100],
            )

        print(f"[Index] Done. {len(points)} points indexed.")

    def search(
        self,
        query: str,
        protocol_id: str,
        concept_queries: Optional[list[str]] = None,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 8,
        include_tables: bool = True,
        source_type_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid dense+sparse search with RRF fusion and reranking.
        Uses Qdrant's native hybrid query with prefetch for true hybrid retrieval.
        """
        self._load_client()
        from qdrant_client.models import (
            Filter, FieldCondition, MatchValue,
            SparseVector, Prefetch, FusionQuery, Fusion
        )

        queries = [query] + (concept_queries or [])
        all_hits: dict[str, RetrievedChunk] = {}

        for q in queries:
            encoded = self._embedder.encode([q])
            dense_vec = encoded["dense"][0]
            sparse_weights = encoded["sparse"][0]
            sparse_idx = [int(k) for k in sparse_weights.keys()] if isinstance(sparse_weights, dict) else []
            sparse_vals = [float(v) for v in sparse_weights.values()] if isinstance(sparse_weights, dict) else []

            # Build filter
            must_conditions = [
                FieldCondition(key="protocol_id", match=MatchValue(value=protocol_id))
            ]
            if source_type_filter:
                must_conditions.append(
                    FieldCondition(key="source_type", match=MatchValue(value=source_type_filter))
                )
            if not include_tables:
                must_conditions.append(
                    FieldCondition(key="source_type", match=MatchValue(value="narrative"))
                )

            search_filter = Filter(must=must_conditions)

            # FIX: True hybrid retrieval using Qdrant's prefetch + RRF fusion
            hybrid_results = self._client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=SparseVector(indices=sparse_idx, values=sparse_vals),
                        using="sparse",
                        limit=top_k_retrieve,
                        filter=search_filter,
                    ),
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=top_k_retrieve,
                        filter=search_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k_retrieve,
            )

            for hit in hybrid_results.points:
                cid = hit.id
                if cid not in all_hits:
                    all_hits[cid] = RetrievedChunk(
                        text=hit.payload["text"],
                        heading=hit.payload.get("heading", ""),
                        source_type=hit.payload.get("source_type", "narrative"),
                        page=hit.payload.get("page", 0),
                        protocol_id=hit.payload["protocol_id"],
                        dense_score=hit.score,
                        chunk_id=hit.payload.get("chunk_id", str(cid)),
                    )

        chunks = list(all_hits.values())
        if not chunks:
            return []

        # Rerank
        return self._reranker.rerank(query, chunks, top_k=top_k_rerank)
