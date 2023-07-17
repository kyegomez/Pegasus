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