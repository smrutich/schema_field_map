"""
Step 3 — Embedding Model Configuration

Provides:
- EmbeddingManager: Local embedding model (all-MiniLM-L6-v2) for semantic similarity
- Hybrid scoring: 0.8 * cosine_similarity + 0.2 * lexical_similarity
- Destination fields embedded once at startup; source fields embedded on-demand

No API cost — runs entirely locally via sentence-transformers.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages local embedding model and similarity computations.

    - Embeds destination fields once and stores the matrix
    - Embeds source fields on-demand during candidate retrieval
    - Computes hybrid scores (embedding + lexical)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._destination_matrix: NDArray[np.float32] | None = None
        self._destination_paths: list[str] = []
        logger.info(f"Embedding model loaded | dim={self.model.get_embedding_dimension()}")

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self.model.get_embedding_dimension()

    # ------------------------------------------------------------------
    # Destination field embedding (once at startup)
    # ------------------------------------------------------------------

    def embed_destination_fields(
        self, text_reprs: list[str], paths: list[str]
    ) -> NDArray[np.float32]:
        """Embed all destination field text_repr strings and store the matrix.

        Args:
            text_reprs: List of destination field text_repr strings to embed
            paths: Parallel list of destination field dot-notation paths (index)

        Returns:
            Normalized embedding matrix (n_fields x embedding_dim)
        """
        if len(text_reprs) != len(paths):
            raise ValueError(
                f"text_reprs ({len(text_reprs)}) and paths ({len(paths)}) must be same length"
            )

        logger.info(f"Embedding {len(text_reprs)} destination fields")
        embeddings = self.model.encode(text_reprs, normalize_embeddings=True)
        self._destination_matrix = np.array(embeddings, dtype=np.float32)
        self._destination_paths = list(paths)

        logger.info(
            f"Destination matrix built | shape={self._destination_matrix.shape}"
        )
        return self._destination_matrix

    @property
    def destination_matrix(self) -> NDArray[np.float32]:
        """Return the pre-built destination embedding matrix."""
        if self._destination_matrix is None:
            raise RuntimeError(
                "Destination embeddings not built yet. "
                "Call embed_destination_fields() first."
            )
        return self._destination_matrix

    @property
    def destination_paths(self) -> list[str]:
        """Return the destination field paths in matrix row order."""
        return self._destination_paths

    # ------------------------------------------------------------------
    # Source field embedding (on-demand)
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed a single text string (e.g., a source field text_repr).

        Returns:
            Normalized embedding vector (1D array of embedding_dim)
        """
        embedding = self.model.encode([text], normalize_embeddings=True)
        return np.array(embedding[0], dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed multiple text strings in a batch.

        Returns:
            Normalized embedding matrix (n_texts x embedding_dim)
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def cosine_similarity(
        self, source_embedding: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute cosine similarity between a source embedding and all destinations.

        Since both source and destination are L2-normalized, cosine similarity
        is simply the dot product.

        Args:
            source_embedding: Normalized source field embedding (1D)

        Returns:
            Array of similarity scores (one per destination field)
        """
        return self.destination_matrix @ source_embedding

    @staticmethod
    def lexical_similarity(source_name: str, destination_path: str) -> float:
        """Compute lexical similarity using RapidFuzz token sort ratio.

        Handles abbreviated names by comparing token-sorted versions:
        - "f_name" vs "fullName.firstName" → tokenized comparison

        Args:
            source_name: Raw source field name (e.g., "f_name")
            destination_path: Dot-notation destination path (e.g., "fullName.firstName")

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Normalize: replace dots/underscores with spaces for fair comparison
        src_normalized = source_name.replace("_", " ").lower()
        dst_normalized = destination_path.replace(".", " ").replace("_", " ").lower()
        return fuzz.token_sort_ratio(src_normalized, dst_normalized) / 100.0

    def hybrid_score(
        self,
        embedding_sim: float,
        lexical_sim: float,
        embedding_weight: float = 0.8,
    ) -> float:
        """Compute weighted hybrid score.

        Default: 0.8 * embedding_similarity + 0.2 * lexical_similarity

        Args:
            embedding_sim: Cosine similarity score (0-1)
            lexical_sim: Lexical similarity score (0-1)
            embedding_weight: Weight for embedding component (default 0.8)

        Returns:
            Hybrid score between 0.0 and 1.0
        """
        lexical_weight = 1.0 - embedding_weight
        return embedding_weight * embedding_sim + lexical_weight * lexical_sim

    # ------------------------------------------------------------------
    # Full retrieval pipeline for a single source field
    # ------------------------------------------------------------------

    def retrieve_candidates(
        self,
        source_text_repr: str,
        source_field_name: str,
        collection_filter: str | None = None,
        collection_names: list[str] | None = None,
        top_k: int = 3,
        threshold: float = 0.40,
    ) -> list[dict]:
        """Retrieve top candidate matches for a source field.

        Combines embedding similarity and lexical similarity into hybrid scores,
        filters by collection if routing has been applied, and returns top-k
        above threshold.

        Args:
            source_text_repr: Enriched text_repr of the source field
            source_field_name: Raw field name (for lexical comparison)
            collection_filter: If set, only return candidates from this collection
            collection_names: Parallel list of collection names per destination path
                              (required if collection_filter is set)
            top_k: Number of top candidates to return
            threshold: Minimum hybrid score to include

        Returns:
            List of candidate dicts sorted by hybrid_score descending:
            [{"destination_field", "embedding_similarity", "lexical_similarity",
              "hybrid_score", "retrieval_confidence_prior"}, ...]
        """
        # Embed source field on-demand
        source_emb = self.embed_text(source_text_repr)

        # Cosine similarity against all destinations
        cos_scores = self.cosine_similarity(source_emb)

        candidates = []
        for i, dest_path in enumerate(self._destination_paths):
            # Apply collection filter if routing is active
            if collection_filter and collection_names:
                if collection_names[i] != collection_filter:
                    continue

            emb_sim = float(cos_scores[i])
            lex_sim = self.lexical_similarity(source_field_name, dest_path)
            h_score = self.hybrid_score(emb_sim, lex_sim)

            if h_score >= threshold:
                # Assign confidence prior
                if h_score > 0.90:
                    prior = "HIGH"
                elif h_score >= 0.75:
                    prior = "MEDIUM"
                else:
                    prior = "LOW"

                candidates.append({
                    "destination_field": dest_path,
                    "embedding_similarity": round(emb_sim, 4),
                    "lexical_similarity": round(lex_sim, 4),
                    "hybrid_score": round(h_score, 4),
                    "retrieval_confidence_prior": prior,
                })

        # Sort by hybrid score descending, take top-k
        candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return candidates[:top_k]
