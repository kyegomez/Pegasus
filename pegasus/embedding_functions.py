from pegasus.types import Documents, EmbeddingFunction, Embeddings
from typing import Optional
import torch

from pegasus.ImageBind import imagebind_model
from pegasus.ImageBind import ModalityType
from pegasus.ImageBind import load_and_transform_text, load_and_transform_vision_data, load_and_transform_audio_data

from concurrent.futures import ThreadPoolExecutor


class MultiModalEmbeddingFunction(EmbeddingFunction):
    _model_cache = {}

    def __init__(
        self,
        modality: str = ModalityType,  # type: ignore
        model_path: str = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
        device: str = "cuda:0"
    ):
        self._modality = modality
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = imagebind_model.imagebind_huge(pretrained=True)
        self._model.eval()
        self._model.to(self.device)

    def __call__(self, *args: Documents) -> Embeddings:
        if self._modality == ModalityType.TEXT:
            inputs = {
                ModalityType.TEXT: load_and_transform_text(
                    args[0], self.device
                )
            }
            print("Inputs:", inputs)
        elif self._modality == ModalityType.VISION:
            inputs = {
                ModalityType.VISION: load_and_transform_vision_data(
                    args[0], self.device
                )
            }
        elif self._modality == ModalityType.AUDIO:
            inputs = {
                ModalityType.AUDIO: load_and_transform_audio_data(
                    args[0], self.device
                )
            }
        else:
            raise ValueError("Invalid modality specified")

        with torch.no_grad():
            embeddings = self._model(inputs)
        
        print("Embeddings:", embeddings)

        # Convert the embeddings tensor to a NumPy array and then to a list of lists (embeddings)
        embeddings_array = embeddings[self._modality].cpu().numpy()

        print("Embeddings array:", embeddings_array)
        # embeddings_list = embeddings_array.tolist()

        # return embeddings_list

        return [embedding.tolist() for embedding in embeddings_array]


"""
text_embedding_function = MultiModalEmbeddingFunction(modality=ModalityType.TEXT)
vision_embedding_function = MultiModalEmbeddingFunction(modality=ModalityType.VISION)
audio_embedding_function = MultiModalEmbeddingFunction(modality=ModalityType.AUDIO)


"""

#ouptu to parquet?

import logging
from concurrent.futures import ThreadPoolExecutor

class OptimizedMultiModalEmbeddingFunction(EmbeddingFunction):
    """
    Class to handle multi-modal embeddings with error handling and logging.
    """
    _model_cache = {}

    def __init__(self, modality: str = ModalityType, model_path: str = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth", device: str = "cuda:0"):
        """
        Initialize the embedding function with specified modality and device.
        Args:
            modality (str): The type of modality - 'TEXT', 'VISION', 'AUDIO'.
            model_path (str): Path to the model file.
            device (str): The device to run the model on - 'cpu' or 'cuda'.
        """
        self._modality = modality
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.model_path = model_path

    def _load_model(self):
        """
        Load model and store it in cache for reusability.
        """
        if (self._modality, self.device) not in self._model_cache:
            model = imagebind_model.imagebind_huge(pretrained=True)
            model.eval()
            model.to(self.device)
            self._model_cache[(self._modality, self.device)] = model

        return self._model_cache[(self._modality, self.device)]

    def __call__(self, *args: Documents) -> Embeddings:
        """
        Main function call to compute embeddings.
        Args:
            args (Documents): Text, video file path, or audio file path.
        Returns:
            Embeddings as a list of lists.
        """
        model = self._load_model()
        load_func = {
            ModalityType.TEXT: load_and_transform_text,
            ModalityType.VISION: load_and_transform_vision_data,
            ModalityType.AUDIO: load_and_transform_audio_data
        }.get(self._modality, None)

        if load_func is None:
            logging.error(f"Invalid modality: {self._modality}")
            raise ValueError("Invalid modality specified")

        # Error handling for input data
        try:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(load_func, args[0], self.device)
                inputs = {self._modality: future.result()}
        except Exception as e:
            logging.error(f"Failed to load input data: {str(e)}")
            raise

        try:
            with torch.no_grad():
                embeddings = model(inputs)
        except Exception as e:
            logging.error(f"Failed to compute embeddings: {str(e)}")
            raise

        del inputs  # Delete the input tensors to free up memory

        try:
            embeddings_array = embeddings[self._modality].cpu().numpy()
        except Exception as e:
            logging.error(f"Failed to convert embeddings to numpy array: {str(e)}")
            raise

        del embeddings  # Delete the output tensor to free up memory

        return [embedding.tolist() for embedding in embeddings_array]
