from typing import Optional, Union, Dict, Sequence, TypeVar, List
from typing_extensions import Literal, TypedDict, Protocol
import pegasus.errors as errors
from abc import ABC, abstractmethod
import torch.nn as nn

# from ..utils.ImageBind.imagebind_model import ModalityType
from pegasus.ImageBind import ModalityType

import numpy as np

# use better cosine
from sklearn.metrics.pairwise import cosine_similarity


ID = str
IDs = List[ID]

Number = Union[int, float]
Embedding = List[Number]
Embeddings = List[Embedding]


Metadata = Dict[str, Union[str, int, float]]
Metadatas = List[Metadata]

Document = str
Documents = List[Document]

Parameter = TypeVar("Parameter", Embedding, Document, Metadata, ID)
T = TypeVar("T")
OneOrMany = Union[T, List[T]]

Include = List[Literal["documents", "embeddings", "metadatas", "distances"]]

# Grammar for where expressions
LiteralValue = Union[str, int, float]
LogicalOperator = Literal["$and", "$or"]
WhereOperator = Literal["$gt", "$gte", "$lt", "$lte", "$ne", "$eq"]
OperatorExpression = Dict[Union[WhereOperator, LogicalOperator], LiteralValue]

Where = Dict[
    Union[str, LogicalOperator], Union[LiteralValue, OperatorExpression, List["Where"]]
]

WhereDocumentOperator = Literal["$contains", LogicalOperator]
WhereDocument = Dict[WhereDocumentOperator, Union[str, List["WhereDocument"]]]


class GetResult(TypedDict):
    ids: List[ID]
    embeddings: Optional[List[Embedding]]
    documents: Optional[List[Document]]
    metadatas: Optional[List[Metadata]]


class QueryResult(TypedDict):
    ids: List[IDs]
    embeddings: Optional[List[List[Embedding]]]
    documents: Optional[List[List[Document]]]
    metadatas: Optional[List[List[Metadata]]]
    distances: Optional[List[List[float]]]


class IndexMetadata(TypedDict):
    dimensionality: int
    elements: int
    time_created: float


class EmbeddingFunction(Protocol):
    def __call__(self, texts: Documents) -> Embeddings:
        ...


def maybe_cast_one_to_many(
    target: OneOrMany[Parameter],
) -> List[Parameter]:
    """Infers if target is Embedding, Metadata, or Document and casts it to a many object if its one"""

    if isinstance(target, Sequence):
        # One Document or ID
        if isinstance(target, str) and target is not None:
            return [target]  # type: ignore
        # One Embedding
        if isinstance(target[0], (int, float)):
            return [target]  # type: ignore
    # One Metadata dict
    if isinstance(target, dict):
        return [target]
    # Already a sequence
    return target  # type: ignore


def validate_ids(ids: IDs) -> IDs:
    """Validates ids to ensure it is a list of strings"""
    if not isinstance(ids, list):
        raise ValueError(f"Expected IDs to be a list, got {ids}")
    for id in ids:
        if not isinstance(id, str):
            raise ValueError(f"Expected ID to be a str, got {id}")
    if len(ids) != len(set(ids)):
        dups = set([x for x in ids if ids.count(x) > 1])
        raise errors.DuplicateIDError(
            f"Expected IDs to be unique, found duplicates for: {dups}"
        )
    return ids


def validate_metadata(metadata: Metadata) -> Metadata:
    """Validates metadata to ensure it is a dictionary of strings to strings, ints, or floats"""
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata to be a dict, got {metadata}")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError(f"Expected metadata key to be a str, got {key}")
        if not isinstance(value, (str, int, float)):
            raise ValueError(
                f"Expected metadata value to be a str, int, or float, got {value}"
            )
    return metadata


def validate_metadatas(metadatas: Metadatas) -> Metadatas:
    """Validates metadatas to ensure it is a list of dictionaries of strings to strings, ints, or floats"""
    if not isinstance(metadatas, list):
        raise ValueError(f"Expected metadatas to be a list, got {metadatas}")
    for metadata in metadatas:
        validate_metadata(metadata)
    return metadatas


def validate_where(where: Where) -> Where:
    """
    Validates where to ensure it is a dictionary of strings to strings, ints, floats or operator expressions,
    or in the case of $and and $or, a list of where expressions
    """
    if not isinstance(where, dict):
        raise ValueError(f"Expected where to be a dict, got {where}")
    for key, value in where.items():
        if not isinstance(key, str):
            raise ValueError(f"Expected where key to be a str, got {key}")
        if (
            key != "$and"
            and key != "$or"
            and not isinstance(value, (str, int, float, dict))
        ):
            raise ValueError(
                f"Expected where value to be a str, int, float, or operator expression, got {value}"
            )
        if key == "$and" or key == "$or":
            if not isinstance(value, list):
                raise ValueError(
                    f"Expected where value for $and or $or to be a list of where expressions, got {value}"
                )
            if len(value) <= 1:
                raise ValueError(
                    f"Expected where value for $and or $or to be a list with at least two where expressions, got {value}"
                )
            for where_expression in value:
                validate_where(where_expression)
        # Value is a operator expression
        if isinstance(value, dict):
            # Ensure there is only one operator
            if len(value) != 1:
                raise ValueError(
                    f"Expected operator expression to have exactly one operator, got {value}"
                )

            for operator, operand in value.items():
                # Only numbers can be compared with gt, gte, lt, lte
                if operator in ["$gt", "$gte", "$lt", "$lte"]:
                    if not isinstance(operand, (int, float)):
                        raise ValueError(
                            f"Expected operand value to be an int or a float for operator {operator}, got {operand}"
                        )

                if operator not in ["$gt", "$gte", "$lt", "$lte", "$ne", "$eq"]:
                    raise ValueError(
                        f"Expected where operator to be one of $gt, $gte, $lt, $lte, $ne, $eq, got {operator}"
                    )

                if not isinstance(operand, (str, int, float)):
                    raise ValueError(
                        f"Expected where operand value to be a str, int, or float, got {operand}"
                    )
    return where


def validate_where_document(where_document: WhereDocument) -> WhereDocument:
    """
    Validates where_document to ensure it is a dictionary of WhereDocumentOperator to strings, or in the case of $and and $or,
    a list of where_document expressions
    """
    if not isinstance(where_document, dict):
        raise ValueError(
            f"Expected where document to be a dictionary, got {where_document}"
        )
    if len(where_document) != 1:
        raise ValueError(
            f"Expected where document to have exactly one operator, got {where_document}"
        )
    for operator, operand in where_document.items():
        if operator not in ["$contains", "$and", "$or"]:
            raise ValueError(
                f"Expected where document operator to be one of $contains, $and, $or, got {operator}"
            )
        if operator == "$and" or operator == "$or":
            if not isinstance(operand, list):
                raise ValueError(
                    f"Expected document value for $and or $or to be a list of where document expressions, got {operand}"
                )
            if len(operand) <= 1:
                raise ValueError(
                    f"Expected document value for $and or $or to be a list with at least two where document expressions, got {operand}"
                )
            for where_document_expression in operand:
                validate_where_document(where_document_expression)
        # Value is a $contains operator
        elif not isinstance(operand, str):
            raise ValueError(
                f"Expected where document operand value for operator $contains to be a str, got {operand}"
            )
    return where_document


def validate_include(include: Include, allow_distances: bool) -> Include:
    """Validates include to ensure it is a list of strings. Since get does not allow distances, allow_distances is used
    to control if distances is allowed"""

    if not isinstance(include, list):
        raise ValueError(f"Expected include to be a list, got {include}")
    for item in include:
        if not isinstance(item, str):
            raise ValueError(f"Expected include item to be a str, got {item}")
        allowed_values = ["embeddings", "documents", "metadatas"]
        if allow_distances:
            allowed_values.append("distances")
        if item not in allowed_values:
            raise ValueError(
                f"Expected include item to be one of {', '.join(allowed_values)}, got {item}"
            )
    return include


class SearchFunction(ABC):
    @abstractmethod
    def search(
        self, query_embeddings: List[Embedding], index_data: dict
    ) -> List[List[ID]]:
        pass


class CrossModalRetrieval(SearchFunction):

    """
    Use the provided MultiModalEmbeddingFunction to compute embeddings for the query.
    Select the corresponding embeddings of the other modality.
    Perform similarity search using the computed embeddings.
    Return the results.

    """

    def __init__(self, modality: str):
        self.modality = modality

    def search(
        self, query_embeddings: List[Embedding], index_data: dict
    ) -> List[List[ID]]:
        other_modality = [m for m in ModalityType if m != self.modality][0]  # type: ignore

        # Get the embeddings for the other modality
        other_embeddings = index_data[other_modality]

        # Perform similarity search
        distances = cosine_similarity(query_embeddings, other_embeddings)  # type: ignore
        sorted_indices = np.argsort(distances, axis=1)[:, ::-1]

        # Get the result IDs
        result_ids = index_data[f"{other_modality}_ids"][sorted_indices]

        return result_ids.tolist()


class MultiModalFusion(SearchFunction):
    def __init__(self, fusion_type: str):
        self.fusion_type = fusion_type

    def search(
        self, query_embeddings: List[Embedding], index_data: dict
    ) -> List[List[ID]]:
        if self.fusion_type == "early":
            combined_query_embeddings = self.early_fusion(query_embeddings)  # type: ignore
            combined_index_embeddings = self.early_fusion(index_data)
        elif self.fusion_type == "late":
            return self.late_fusion(query_embeddings, index_data)
        else:
            raise ValueError("Invalid fusion_type specified")

        distances = cosine_similarity(
            combined_query_embeddings, combined_index_embeddings
        )
        sorted_indices = np.argsort(distances, axis=1)[:, ::-1]

        # Get the result IDs
        result_ids = index_data["ids"][sorted_indices]

        return result_ids.tolist()

    @staticmethod
    def early_fusion(embeddings: dict) -> np.ndarray:
        combined_embeddings = np.hstack(list(embeddings.values()))
        return combined_embeddings

    def late_fusion(
        self, query_embeddings: List[Embedding], index_data: dict
    ) -> List[List[ID]]:
        result_scores = []
        for modality, embeddings in query_embeddings.items():  # type: ignore
            distances = cosine_similarity(embeddings, index_data[modality])
            result_scores.append(distances)

        combined_scores = np.mean(result_scores, axis=0)
        sorted_indices = np.argsort(combined_scores, axis=1)[:, ::-1]

        # Get the result IDs
        result_ids = index_data["ids"][sorted_indices]

        return result_ids.tolist()


class ModalitySpecificSearching(SearchFunction):
    def __init__(self, modality: str):
        self.modality = modality

    def search(
        self, query_embeddings: List[Embedding], index_data: dict
    ) -> List[List[ID]]:
        # Get the embeddings for the target modality
        modality_embeddings = index_data[self.modality]

        # Perform similarity search
        distances = cosine_similarity(query_embeddings, modality_embeddings)  # type: ignore
        sorted_indices = np.argsort(distances, axis=1)[:, ::-1]

        # Get the result IDs
        result_ids = index_data[f"{self.modality}_ids"][sorted_indices]

        return result_ids.tolist()


# to do -> better cosine perhaps torch nn.CosineSimilarity + exception handling