import unittest
from pathlib import Path
from src.rag.indexer import FMEAIndexer
from src.rag.fmea_retriever import FMEARetriever
from src.rag.cache import SemanticRAGCache

CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "fmea_corpus"

class TestFMEARAGKnowledgeEngine(unittest.TestCase):

    def setUp(self):
        self.corpus_dir = CORPUS_DIR
        self.cache = SemanticRAGCache(max_size=50, ttl_seconds=60)
        self.retriever = FMEARetriever(corpus_dir=self.corpus_dir, cache=self.cache)

    def test_fmea_corpus_files_exist(self):
        expected_files = [
            "300mm_rie_plasma_etch_fmea.md",
            "300mm_photolithography_fmea.md",
            "300mm_cmp_planarization_fmea.md"
        ]
        for fname in expected_files:
            path = self.corpus_dir / fname
            self.assertTrue(path.exists(), f"Corpus file missing: {fname}")
            self.assertGreater(path.stat().st_size, 200, f"File {fname} too small")

    def test_indexer_chunks_documents(self):
        indexer = FMEAIndexer(self.corpus_dir)
        chunks = indexer.load_and_chunk_corpus()
        self.assertGreaterEqual(len(chunks), 6, "Should create at least 6 distinct section chunks")
        
        doc_ids = {c.doc_id for c in chunks}
        self.assertIn("FMEA-SOP-ETCH-300-CH3", doc_ids)
        self.assertIn("FMEA-SOP-LITHO-300-SC2", doc_ids)
        self.assertIn("FMEA-SOP-CMP-300-PL1", doc_ids)

    def test_retriever_etch_center_short_query(self):
        query = "Center defect pattern with micro bridging short in 300mm RIE Chamber 3"
        results = self.retriever.retrieve(query, top_k=2)
        
        self.assertGreater(len(results), 0, "Retriever should return matching results")
        top_match = results[0]
        self.assertEqual(top_match["doc_id"], "FMEA-SOP-ETCH-300-CH3")
        self.assertIn("Center", top_match["failure_classes"])
        self.assertGreaterEqual(top_match["similarity_score"], 0.5)

    def test_retriever_litho_scratch_query(self):
        query = "Linear scratch streaks across dies on scanner track"
        results = self.retriever.retrieve(query, top_k=2)
        
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match["doc_id"], "FMEA-SOP-LITHO-300-SC2")
        self.assertIn("Scratch", top_match["failure_classes"])

    def test_semantic_cache_hit_and_latency_reduction(self):
        query = "Backside Helium leakage and RF match calibration in Chamber 3"
        
        # 1st call: Cache Miss
        self.cache.clear()
        res1 = self.retriever.retrieve(query, top_k=1)
        self.assertEqual(self.cache.metrics["misses"], 1)
        self.assertEqual(self.cache.metrics["hits"], 0)

        # 2nd call: Cache Hit
        res2 = self.retriever.retrieve(query, top_k=1)
        self.assertEqual(self.cache.metrics["hits"], 1)
        self.assertEqual(self.cache.metrics["hit_rate"], 0.5)
        self.assertEqual(res1, res2)

    def test_cache_eviction_and_ttl(self):
        tiny_cache = SemanticRAGCache(max_size=2, ttl_seconds=1)
        tiny_retriever = FMEARetriever(corpus_dir=self.corpus_dir, cache=tiny_cache)
        
        tiny_retriever.retrieve("query 1", top_k=1)
        tiny_retriever.retrieve("query 2", top_k=1)
        self.assertEqual(tiny_cache.metrics["cache_size"], 2)
        
        # Adding 3rd should evict oldest
        tiny_retriever.retrieve("query 3", top_k=1)
        self.assertEqual(tiny_cache.metrics["cache_size"], 2)

if __name__ == "__main__":
    unittest.main()
