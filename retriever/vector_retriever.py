# create milvus
from retriever.custom_milvus import CustomMilvusVectorStore, CustomVectorStoreQuery

# retriever
from llama_index.core.indices.base_retriever import BaseRetriever
from typing import Optional, Any, List
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.schema import NodeWithScore

from language_model.embed_model import embed_helper

from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    QueryType,
    TextNode,
)

from llama_index.core.callbacks.schema import CBEventType, EventPayload

class VectorDBRetriever(BaseRetriever):
    """Retriever over a milvus vector store."""

    def __init__(
        self,
        vector_store: CustomMilvusVectorStore,
        embed_model: Any = None,
        query_mode: str = "default",
        similarity_top_k: int = 2,
        filters: Optional[MetadataFilters] = None,
        embed_helper: Optional[embed_helper] = None,
        expr: Optional[List[str]] = None,
        expr_prefix: Optional[str] = None,
        expr_suffix: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self._filters = filters
        self._embed_helper = embed_helper
        self._embed_helper.set_model()
        self._expr = expr
        self._expr_prefix = expr_prefix
        self._expr_suffix = expr_suffix
        self._condition = condition
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        if self._embed_helper is None:
            query_embedding = self._embed_model.get_query_embedding(
                query_bundle.query_str
            )
        else:
            query_embedding = self._embed_helper.get_embedding(query_bundle.query_str)
        vector_store_query = CustomVectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
            filters=self._filters,
            expr=self._expr,
            expr_prefix=self._expr_prefix,
            expr_suffix=self._expr_suffix,
            condition=self._condition,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
    # def retrieve_chunk(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
    #     """Retrieve nodes given query.

    #     Args:
    #         str_or_query_bundle (QueryType): Either a query string or
    #             a QueryBundle object.

    #     """
    #     self._check_callback_manager()

    #     if isinstance(str_or_query_bundle, str):
    #         query_bundle = QueryBundle(str_or_query_bundle)
    #     else:
    #         query_bundle = str_or_query_bundle
    #     with self.callback_manager.as_trace("query"):
    #         with self.callback_manager.event(
    #             CBEventType.RETRIEVE,
    #             payload={EventPayload.QUERY_STR: query_bundle.query_str},
    #         ) as retrieve_event:
    #             nodes = self._retrieve_chunk(query_bundle)
    #             nodes = self._handle_recursive_retrieval(query_bundle, nodes)
    #             retrieve_event.on_end(
    #                 payload={EventPayload.NODES: nodes},
    #             )

    #     return nodes
    
    # def _retrieve_chunk(self, query_bundle: str | QueryBundle) -> List[NodeWithScore]:
    #     """Retrieve."""
    #     if self._embed_helper is None:
    #         query_embedding = self._embed_model.get_query_embedding(
    #             query_bundle.query_str
    #         )
    #     else:
    #         query_embedding = self._embed_helper.get_embedding(query_bundle.query_str)
    #     expr_list = [each_expr+"%" for each_expr in self._expr] if self._expr else None
    #     vector_store_query = CustomVectorStoreQuery(
    #         query_embedding=query_embedding,
    #         similarity_top_k=self._similarity_top_k,
    #         mode=self._query_mode,
    #         filters=self._filters,
    #         expr=expr_list,
    #         expr_prefix=self._expr_prefix,
    #         expr_suffix=self._expr_suffix,
    #         condition=self._condition,
    #     )
    #     query_result = self._vector_store.query_chunk(vector_store_query)

    #     nodes_with_scores = []
    #     for index, node in enumerate(query_result.nodes):
    #         score: Optional[float] = None
    #         if query_result.similarities is not None:
    #             score = query_result.similarities[index]
            
    #         nodes_with_scores.append(NodeWithScore(node=node, score=score))

    #     return nodes_with_scores