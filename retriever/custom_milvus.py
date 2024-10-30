from llama_index.vector_stores.milvus import MilvusVectorStore
from typing import Any, Dict, List, Optional, Union
  
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from llama_index.core.schema import TextNode
from llama_index.legacy.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.legacy.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pymilvus import MilvusClient, Partition, connections

logger = logging.getLogger(__name__)

MILVUS_ID_FIELD = "id"
def _to_milvus_filter(standard_filters: MetadataFilters) -> List[str]:
    """Translate standard metadata filters to Milvus specific spec."""
    filters = []
    for filter in standard_filters.legacy_filters():
        if isinstance(filter.value, str):
            filters.append(str(filter.key) + " == " + '"' + str(filter.value) + '"')
        else:
            filters.append(str(filter.key) + " == " + str(filter.value))
    return filters

@dataclass
class CustomVectorStoreQuery(VectorStoreQuery):
    expr : Optional[List[str]] = None
    expr_prefix : Optional[str] = None
    expr_suffix : Optional[str] = None
    condition : Optional[str] = None
    output_fields : Optional[List[str]] = None

        
class CustomMilvusVectorStore(MilvusVectorStore):
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        collection_name: str = "llamalection",
        partition_name: Optional[str] = None,
        dim: Optional[int] = None,
        embedding_field: str = DEFAULT_EMBEDDING_KEY,
        doc_id_field: str = DEFAULT_DOC_ID_KEY,
        similarity_metric: str = "COSINE",
        consistency_level: str = "Strong",
        overwrite: bool = False,
        text_key: Optional[str] = None,
        index_config: Optional[dict] = None,
        search_config: Optional[dict] = None,
        metadata_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(uri, token, collection_name, dim, embedding_field, doc_id_field, similarity_metric, consistency_level, overwrite, text_key, index_config, search_config, **kwargs)
        self.metadata_key = metadata_key
        self.partition_name = partition_name
        if self.partition_name == None:
            self.partition_names = None
        else:
            self.partition_names = [partition_name]
        
        # set self.partition(list type of partition_name not yet supported)

    def query_chunk(self, query: CustomVectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            node_ids (Optional[List[str]]): list of node_ids to filter by
            output_fields (Optional[List[str]]): list of fields to return
            embedding_field (Optional[str]): name of embedding field
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        expr = [" ".join([f"{MILVUS_ID_FIELD}", "like", f'\"{each_expr}\"']).strip() for each_expr in query.expr]
        output_fields = ["*"] if query.output_fields is None else query.output_fields

        # Parse the filter
        if query.filters is not None:
            expr.extend(_to_milvus_filter(query.filters))

        # Parse any docs we are filtering on
        if query.doc_ids is not None and len(query.doc_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.doc_ids]
            expr.append(f"{self.doc_id_field} in [{','.join(expr_list)}]")

        # Parse any nodes we are filtering on
        if query.node_ids is not None and len(query.node_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.node_ids]
            expr.append(f"{MILVUS_ID_FIELD} in [{','.join(expr_list)}]")

        # Limit output fields
        if query.output_fields is not None:
            output_fields = query.output_fields

        # Convert to string expression
        string_expr = ""
        if len(expr) != 0:
            string_expr = f" {query.condition} ".join(expr)

        # Perform the search
        print(string_expr)
        res = self.milvusclient.search(
            collection_name=self.collection_name,
            partition_names=self.partition_names,
            data=[query.query_embedding],
            filter=string_expr,
            limit=query.similarity_top_k,
            output_fields=output_fields,
            search_params=self.search_config,
        )

        logger.debug(
            f"Successfully searched embedding in collection: {self.collection_name}"
            f" Num Results: {len(res[0])}"
        )

        nodes = []
        similarities = []
        ids = []

        # Parse the results
        for hit in res[0]:
            if not self.text_key:
                node = metadata_dict_to_node(
                    {
                        "_node_content": hit["entity"].get("_node_content", None),
                        "_node_type": hit["entity"].get("_node_type", None),
                    }
                )
            else:
                try:
                    text = hit["entity"].get(self.text_key)
                except Exception:
                    raise ValueError(
                        "The passed in text_key value does not exist "
                        "in the retrieved entity."
                    )
                node = TextNode(
                    text=text,
                )
                node.metadata = hit["entity"].get(self.metadata_key)
            nodes.append(node)
            similarities.append(hit["distance"])
            ids.append(hit["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
    
    def query(self, query: CustomVectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            node_ids (Optional[List[str]]): list of node_ids to filter by
            output_fields (Optional[List[str]]): list of fields to return
            embedding_field (Optional[str]): name of embedding field
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        expr = []
        output_fields = ["*"]

        # Parse the filter
        if query.filters is not None:
            expr.extend(_to_milvus_filter(query.filters))

        # Parse any docs we are filtering on
        if query.doc_ids is not None and len(query.doc_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.doc_ids]
            expr.append(f"{self.doc_id_field} in [{','.join(expr_list)}]")

        # Parse any nodes we are filtering on
        if query.node_ids is not None and len(query.node_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.node_ids]
            expr.append(f"{MILVUS_ID_FIELD} in [{','.join(expr_list)}]")

        # Limit output fields
        if query.output_fields is not None:
            output_fields = query.output_fields

        # Convert to string expression
        string_expr = ""
        if len(expr) != 0:
            string_expr = f" {query.filters.condition.value} ".join(expr)

        # Perform the search
        print(string_expr)
        res = self.milvusclient.search(
            collection_name=self.collection_name,
            partition_names=self.partition_names,
            data=[query.query_embedding],
            filter=string_expr,
            limit=query.similarity_top_k,
            output_fields=output_fields,
            search_params=self.search_config,
        )

        logger.debug(
            f"Successfully searched embedding in collection: {self.collection_name}"
            f" Num Results: {len(res[0])}"
        )

        nodes = []
        similarities = []
        ids = []

        # Parse the results
        for hit in res[0]:
            if not self.text_key:
                node = metadata_dict_to_node(
                    {
                        "_node_content": hit["entity"].get("_node_content", None),
                        "_node_type": hit["entity"].get("_node_type", None),
                    }
                )
            else:
                try:
                    text = hit["entity"].get(self.text_key)
                except Exception:
                    raise ValueError(
                        "The passed in text_key value does not exist "
                        "in the retrieved entity."
                    )
                node = TextNode(
                    text=text,
                )
                node.id_ = hit["entity"].get("id")
                node.metadata = hit["entity"].get(self.metadata_key)
            nodes.append(node)
            similarities.append(hit["distance"])
            ids.append(hit["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)