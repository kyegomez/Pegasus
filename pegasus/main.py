from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit
from joblib import Memory
import numpy as np

# from pegasus.Ocean import ImageBindEmbeddingFunctio
from pegasus.oceandb import ImageBindEmbeddingFunction

memory = Memory("PegasusStore", verbose=0)

@memory.cache
@njit
def optimized_embedding_function(modality, data):
    return ImageBindEmbeddingFunction(modality)(data)

class Pegasus:
    def __init__(self, modality, multi_process=False, n_processes=4):
        self.modality = modality
        self.multi_process = multi_process
        self.n_processes = n_processes if multi_process else 1
        
    def _embed_data(self, data):
        if self.modality not in {"text", "audio", "vision", "sensor", "heatmap"}:
            raise ValueError("Invalid modality")
        return optimized_embedding_function(self.modality, data)
    
    def embed_data(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if not self.multi_process:
            return self._embed_data(data)
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            future_to_data = {executor.submit(self._embed_data, d): d for d in data}
            return {future_to_data[future]: future.result() for future in as_completed(future_to_data)}