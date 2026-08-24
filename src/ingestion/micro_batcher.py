import time
from typing import List, Dict, Any, Generator

class DynamicMicroBatcher:
    """
    Groups high-velocity die inspection streams into optimal micro-batches (e.g. 16 or 32 items)
    with adaptive latency timeout thresholds to saturate GPU TensorRT inference engines.
    """
    def __init__(self, max_batch_size: int = 32, max_latency_ms: float = 20.0):
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms

    def batch_stream(
        self,
        item_stream: List[Dict[str, Any]]
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Yields dynamic micro-batches adhering to batch size ceilings and latency boundaries."""
        current_batch = []
        batch_start_time = time.time()
        
        for item in item_stream:
            current_batch.append(item)
            elapsed_ms = (time.time() - batch_start_time) * 1000.0
            
            if len(current_batch) >= self.max_batch_size or elapsed_ms >= self.max_latency_ms:
                yield current_batch
                current_batch = []
                batch_start_time = time.time()
                
        if current_batch:
            yield current_batch
