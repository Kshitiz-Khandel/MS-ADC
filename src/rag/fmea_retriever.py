import re
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.rag.indexer import FMEAIndexer, FMEAChunk
from src.rag.cache import SemanticRAGCache

class FMEARetriever:
    """
    Retrieves matching FMEA troubleshooting SOPs using semantic similarity and chamber relevance.
    Backed by SemanticRAGCache for sub-millisecond response latency (Comp 24).
    """
    def __init__(self, corpus_dir: Optional[Path] = None, cache: Optional[SemanticRAGCache] = None):
        self.corpus_dir = corpus_dir or (Path(__file__).parent.parent.parent / "data" / "fmea_corpus")
        self.indexer = FMEAIndexer(self.corpus_dir)
        self.chunks = self.indexer.load_and_chunk_corpus()
        self.cache = cache or SemanticRAGCache()

    def _compute_relevance_score(self, query: str, chunk: FMEAChunk) -> float:
        q_lower = query.lower()
        query_terms = set(re.findall(r"\w+", q_lower))
        if not query_terms:
            return 0.0

        content_text = f"{chunk.doc_id} {chunk.section_title} {chunk.content} {' '.join(chunk.failure_classes)} {chunk.tool_chamber}".lower()
        matched_terms = [t for t in query_terms if t in content_text]
        raw_score = len(matched_terms) / len(query_terms)

        # Precise Chamber Grounding
        if "etch" in q_lower:
            if "etch" in chunk.doc_id.lower() or "etch" in chunk.tool_chamber.lower():
                raw_score += 0.8
            else:
                return 0.0
        elif "litho" in q_lower:
            if "litho" in chunk.doc_id.lower() or "litho" in chunk.tool_chamber.lower() or "scanner" in chunk.tool_chamber.lower():
                raw_score += 0.8
            else:
                return 0.0
        elif "cmp" in q_lower:
            if "cmp" in chunk.doc_id.lower() or "cmp" in chunk.tool_chamber.lower() or "platen" in chunk.tool_chamber.lower():
                raw_score += 0.8
            else:
                return 0.0

        # Defect class match boost
        for fc in chunk.failure_classes:
            if fc.lower() in q_lower:
                raw_score += 0.4

        return min(1.0, round(raw_score, 4))

    def retrieve(self, query: str, top_k: int = 2, min_score: float = 0.2) -> List[Dict[str, Any]]:
        # 1. Check Semantic Cache
        cached_result = self.cache.get(query, top_k)
        if cached_result is not None:
            return cached_result

        # 2. Score chunks
        scored_chunks = []
        for chunk in self.chunks:
            score = self._compute_relevance_score(query, chunk)
            if score >= min_score:
                item = chunk.to_dict()
                item["similarity_score"] = score
                scored_chunks.append(item)

        # 3. Sort by score descending
        scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = scored_chunks[:top_k]

        # 4. Store in Cache
        self.cache.set(query, top_k, results)
        return results
