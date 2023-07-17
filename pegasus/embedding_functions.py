from pegasus.types import Documents, EmbeddingFunction, Embeddings
from typing import Optional
import torch

from pegasus.ImageBind import imagebind_model
from pegasus.ImageBind import ModalityType
from pegasus.ImageBind import load_and_transform_text, load_and_transform_vision_data, load_and_transform_audio_data


class MultiModalEmbeddingFunction(EmbeddingFunction):
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


class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self, api_key: Optional[str] = None, model_name: str = "text-embedding-ada-002"
    ):
        try:
            import openai
        except ImportError:
            raise ValueError(
                "The openai python package is not installed. Please install it with `pip install openai`"
            )

        if api_key is not None:
            openai.api_key = api_key
        # If the api key is still not set, raise an error
        elif openai.api_key is None:
            raise ValueError(
                "Please provide an OpenAI API key. You can get one at https://platform.openai.com/account/api-keys"
            )

        self._client = openai.Embedding
        self._model_name = model_name

    def __call__(self, texts: Documents) -> Embeddings:
        # replace newlines, which can negatively affect performance.
        texts = [t.replace("\n", " ") for t in texts]

        # Call the OpenAI Embedding API
        embeddings = self._client.create(input=texts, engine=self._model_name)["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])

        # Return just the embeddings
        return [result["embedding"] for result in sorted_embeddings]


class CohereEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "large"):
        try:
            import cohere
        except ImportError:
            raise ValueError(
                "The cohere python package is not installed. Please install it with `pip install cohere`"
            )

        self._client = cohere.Client(api_key)
        self._model_name = model_name

    def __call__(self, texts: Documents) -> Embeddings:
        # Call Cohere Embedding API for each document.
        return [
            embeddings
            for embeddings in self._client.embed(texts=texts, model=self._model_name)
        ]


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self, api_key: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        try:
            import requests
        except ImportError:
            raise ValueError(
                "The requests python package is not installed. Please install it with `pip install requests`"
            )
        self._api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def __call__(self, texts: Documents) -> Embeddings:
        # Call HuggingFace Embedding API for each document
        return self._session.post(
            self._api_url, json={"inputs": texts, "options": {"wait_for_model": True}}
        ).json()

