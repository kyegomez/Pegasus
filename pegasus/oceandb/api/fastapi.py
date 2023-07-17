from typing import Callable, Dict, List, Optional
import pegasus.oceandb.api import API
import pegasus.oceandb.api.types import (
    CrossModalRetrieval,
    Documents,
    Embeddings,
    Embedding,
    IDs,
    Include,
    Metadatas,
    ModalitySpecificSearching,
    MultiModalFusion,
    Where,
    WhereDocument,
)
import pandas as pd
import requests
import json
from typing import Sequence, List, Any
import pegasus.oceandb.api.models.Collection import Collection
import pegasus.oceandb.telemetry import Telemetry
import pegasus.oceandb.errors as errors
from uuid import UUID


class FastAPI(API):
    def __init__(self, settings, telemetry_client: Telemetry):
        url_prefix = "https" if settings.ocean_server_ssl_enabled else "http"
        self._api_url = f"{url_prefix}://{settings.ocean_server_host}:{settings.ocean_server_http_port}/api/v1"
        self._telemetry_client = telemetry_client

    def heartbeat(self):
        """Returns the current server time in nanoseconds to check if the server is alive"""
        resp = requests.get(self._api_url)
        raise_ocean_error(resp)
        return int(resp.json()["nanosecond heartbeat"])

    def list_collections(self) -> Sequence[Collection]:
        """Returns a list of all collections"""
        resp = requests.get(self._api_url + "/collections")
        raise_ocean_error(resp)
        json_collections = resp.json()
        collections = []
        for json_collection in json_collections:
            collections.append(Collection(self, **json_collection))

        return collections

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict] = None,
        embedding_function: Optional[Callable] = None,
        get_or_create: bool = False,
    ) -> Collection:
        """Creates a collection"""
        resp = requests.post(
            self._api_url + "/collections",
            data=json.dumps(
                {"name": name, "metadata": metadata, "get_or_create": get_or_create}
            ),
        )
        raise_ocean_error(resp)
        resp_json = resp.json()
        return Collection(
            client=self,
            id=resp_json["id"],
            name=resp_json["name"],
            embedding_function=embedding_function,
            metadata=resp_json["metadata"],
        )

    def get_collection(
        self,
        name: str,
        embedding_function: Optional[Callable] = None,
    ) -> Collection:
        """Returns a collection"""
        resp = requests.get(self._api_url + "/collections/" + name)
        raise_ocean_error(resp)
        resp_json = resp.json()
        return Collection(
            client=self,
            name=resp_json["name"],
            id=resp_json["id"],
            embedding_function=embedding_function,
            metadata=resp_json["metadata"],
        )

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict] = None,
        embedding_function: Optional[Callable] = None,
    ) -> Collection:
        """Get a collection, or return it if it exists"""

        return self.create_collection(
            name, metadata, embedding_function, get_or_create=True
        )

    def _modify(self, id: UUID, new_name: str, new_metadata: Optional[Dict] = None):
        """Updates a collection"""
        resp = requests.put(
            self._api_url + "/collections/" + str(id),
            data=json.dumps({"new_metadata": new_metadata, "new_name": new_name}),
        )
        raise_ocean_error(resp)
        return resp.json()

    def delete_collection(self, name: str):
        """Deletes a collection"""
        resp = requests.delete(self._api_url + "/collections/" + name)
        raise_ocean_error(resp)

    def _count(self, collection_id: UUID):
        """Returns the number of embeddings in the database"""
        resp = requests.get(
            self._api_url + "/collections/" + str(collection_id) + "/count"
        )
        raise_ocean_error(resp)
        return resp.json()

    def _peek(self, collection_id, limit=10):
        return self._get(
            collection_id,
            limit=limit,
            include=["embeddings", "documents", "metadatas"],
        )

    def _get(
        self,
        collection_id: UUID,
        ids: Optional[IDs] = None,
        where: Optional[Where] = {},
        sort: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        where_document: Optional[WhereDocument] = {},
        include: Include = ["metadatas", "documents"],
    ):
        """Gets embeddings from the database"""
        if page and page_size:
            offset = (page - 1) * page_size
            limit = page_size

        resp = requests.post(
            self._api_url + "/collections/" + str(collection_id) + "/get",
            data=json.dumps(
                {
                    "ids": ids,
                    "where": where,
                    "sort": sort,
                    "limit": limit,
                    "offset": offset,
                    "where_document": where_document,
                    "include": include,
                }
            ),
        )

        raise_ocean_error(resp)
        return resp.json()

    def _delete(self, collection_id: UUID, ids=None, where={}, where_document={}):
        """Deletes embeddings from the database"""

        resp = requests.post(
            self._api_url + "/collections/" + str(collection_id) + "/delete",
            data=json.dumps(
                {"where": where, "ids": ids, "where_document": where_document}
            ),
        )

        raise_ocean_error(resp)
        return resp.json()

    def _add(
        self,
        ids,
        collection_id: UUID,
        embeddings,
        metadatas=None,
        documents=None,
        increment_index=True,
    ):
        """
        Adds a batch of embeddings to the database
        - pass in column oriented data lists
        - by default, the index is progressively built up as you add more data. If for ingestion performance reasons you want to disable this, set increment_index to False
        -     and then manually create the index yourself with collection.create_index()
        """
        resp = requests.post(
            self._api_url + "/collections/" + str(collection_id) + "/add",
            data=json.dumps(
                {
                    "ids": ids,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "documents": documents,
                    "increment_index": increment_index,
                }
            ),
        )

        raise_ocean_error(resp)
        return True

    def _update(
        self,
        collection_id: UUID,
        ids: IDs,
        embeddings: Optional[Embeddings] = None,
        metadatas: Optional[Metadatas] = None,
        documents: Optional[Documents] = None,
    ):
        """
        Updates a batch of embeddings in the database
        - pass in column oriented data lists
        """

        resp = requests.post(
            self._api_url + "/collections/" + str(collection_id) + "/update",
            data=json.dumps(
                {
                    "ids": ids,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "documents": documents,
                }
            ),
        )

        resp.raise_for_status()
        return True

    def _upsert(
        self,
        collection_id: UUID,
        ids: IDs,
        embeddings: Embeddings,
        metadatas: Optional[Metadatas] = None,
        documents: Optional[Documents] = None,
        increment_index: bool = True,
    ):
        """
        Updates a batch of embeddings in the database
        - pass in column oriented data lists
        """

        resp = requests.post(
            self._api_url + "/collections/" + str(collection_id) + "/upsert",
            data=json.dumps(
                {
                    "ids": ids,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "documents": documents,
                    "increment_index": increment_index,
                }
            ),
        )

        resp.raise_for_status()
        return True

    def _query(
        self,
        collection_id: UUID,
        query_embeddings,
        n_results=10,
        where={},
        where_document={},
        include: Include = ["metadatas", "documents", "distances"],
    ):
        """Gets the nearest neighbors of a single embedding"""

        resp = requests.post(
            self._api_url + "/collections/" + str(collection_id) + "/query",
            data=json.dumps(
                {
                    "query_embeddings": query_embeddings,
                    "n_results": n_results,
                    "where": where,
                    "where_document": where_document,
                    "include": include,
                }
            ),
        )

        raise_ocean_error(resp)
        body = resp.json()
        return body

    def search(
        self, search_function: str, query_embeddings: List[Embedding], index_data: Any
    ) -> List[str]:
        if search_function == "cross_modal_retrieval":
            search_instance = CrossModalRetrieval()  # type: ignore
        elif search_function == "multi_modal_fusion":
            search_instance = MultiModalFusion()  # type: ignore
        elif search_function == "modality_specific_searching":
            search_instance = ModalitySpecificSearching()  # type: ignore
        else:
            raise ValueError("Invalid search function specified")

        result_ids = search_instance.search(query_embeddings, index_data)
        return result_ids  # type: ignore

    def reset(self):
        """Resets the database"""
        resp = requests.post(self._api_url + "/reset")
        raise_ocean_error(resp)
        return resp.json

    def persist(self):
        """Persists the database"""
        resp = requests.post(self._api_url + "/persist")
        raise_ocean_error(resp)
        return resp.json

    def raw_sql(self, sql):
        """Runs a raw SQL query against the database"""
        resp = requests.post(
            self._api_url + "/raw_sql", data=json.dumps({"raw_sql": sql})
        )
        raise_ocean_error(resp)
        return pd.DataFrame.from_dict(resp.json())

    def create_index(self, collection_name: str):
        """Creates an index for the given space key"""
        resp = requests.post(
            self._api_url + "/collections/" + collection_name + "/create_index"
        )
        raise_ocean_error(resp)
        return resp.json()

    def get_version(self):
        """Returns the version of the server"""
        resp = requests.get(self._api_url + "/version")
        raise_ocean_error(resp)
        return resp.json()


def raise_ocean_error(resp):
    """Raises an error if the response is not ok, using a OceanError if possible"""
    if resp.ok:
        return

    ocean_error = None
    try:
        body = resp.json()
        if "error" in body:
            if body["error"] in errors.error_types:
                ocean_error = errors.error_types[body["error"]](body["message"])

    except BaseException:
        pass

    if ocean_error:
        raise ocean_error

    try:
        resp.raise_for_status()
    except requests.HTTPError:
        raise (Exception(resp.text))