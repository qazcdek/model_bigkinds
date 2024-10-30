from retriever.custom_milvus import CustomMilvusVectorStore
from retriever.vector_retriever import VectorDBRetriever
from language_model.embed_model import embed_helper
from config import MILVUS_URI, MILVUS_TOKEN
from pymilvus import Partition, connections

embed_class = embed_helper(dir_name="bge_base_onnx", embed_path='model', model_name="BAAI/bge-base-en-v1.5")
openai_embed_class = embed_helper(dir_name="bge_base_onnx", embed_path='model', model_name="openai")

QUERY_PREFIX = "Represent this sentence for searching relevant passages:"

class QueryBundle:
    def __init__(self, query_str: str):
        self.query_str = query_str

def using_text_query(search_query, return_ids=False):
    try:
        print("MILVUS TOKEN : ", MILVUS_TOKEN)
        doc_store = CustomMilvusVectorStore(uri=MILVUS_URI, token=MILVUS_TOKEN, collection_name='documents_FR', text_key="text", metadata_key="metadata")
        doc_store.collection.load() # (partition_names=agency_names)
        retriever_doc = VectorDBRetriever(
            doc_store, query_mode="default", similarity_top_k=60, embed_helper=embed_class
        )
        query_instruction = QUERY_PREFIX
        query_bundle = QueryBundle(query_str=query_instruction + search_query)
        retrieved_doc = retriever_doc._retrieve(query_bundle)
        documents_id = []
        for node in retrieved_doc:
            documents_id.append(node.node_id)
            
        if return_ids:
            return documents_id
            
        return retrieved_doc
    except Exception as e:
        print("ERROR: ", e)
        return []

def get_url_util(sorted_retrieved_doc):
    html_urls = []
    for node in sorted_retrieved_doc:
        try:
            if node.metadata['html_url'] not in html_urls:
                html_urls.append(node.metadata['html_url'])
        except:
            pass
    return html_urls
    
def get_context_retrieved_doc(retrieved_doc):
    if len(retrieved_doc) <= 0:
        return "", []
    sorted_retrieved_doc = sorted(retrieved_doc, key=lambda item: item.score)
    doc_ids = []
    context_list = []
    html_urls = get_url_util(sorted_retrieved_doc)
    
    for node in sorted_retrieved_doc:
        doc_id = node.node_id[:-4] if len(node.node_id)>20 else node.node_id[:-5]
        if doc_id in doc_ids:
            idx = doc_ids.index(doc_id)
            context_list[idx].append(node)
        else:
            doc_ids.append(doc_id)
            context_list.append([node])
    sorted_context_list = [sorted(doc_list, key=lambda item: item.node_id) for doc_list in context_list]
    
    context = []
    core_meta = ['citation', 'document_number', 'document_overall_information', 'html_url', 'title']
    
    for doc_list in sorted_context_list:
        first_doc = True
        for doc in doc_list:
            if first_doc:
                for meta_key in doc.metadata:
                    if meta_key in core_meta:
                        if meta_key == 'citation':
                            meta_key_str = 'citation_number'
                        else:
                            meta_key_str = meta_key
                        context.append(str(meta_key_str) + ": " + str(doc.metadata[meta_key]))
                context.append(doc.get_content(metadata_mode="none"))
                first_doc = False
            else:
                context.append(doc.get_content(metadata_mode="none"))
        context.append("\n--------------------\n")
    return "\n".join(context), html_urls, doc_ids

def extract_chunks_from_bookshelf(search_query, bookshelf_id, collection_name, similarity_top_k=10, milvus_uri=MILVUS_URI, milvus_token=MILVUS_TOKEN):
    try:
        partition_name = bookshelf_id
        print("extract_chunks_from_bookshelf : ", partition_name)
        doc_store_chunk = CustomMilvusVectorStore(uri=milvus_uri, token=milvus_token, collection_name=collection_name, partition_name=partition_name, text_key="text", metadata_key="metadata")
        connections.connect(
            uri=milvus_uri,
            token=milvus_token)
        Partition(collection=collection_name, name=partition_name).load()
        retriever_doc_chunk = VectorDBRetriever(
        doc_store_chunk, query_mode="default", similarity_top_k=similarity_top_k, embed_helper=openai_embed_class
        )
        print("RECIEVED SEARCH QUERY: ", search_query)
        print("RETRIEVER_DOC: ", retriever_doc_chunk)
        query_instruction = QUERY_PREFIX
        query_bundle = QueryBundle(query_str=search_query)
        retrieved_doc = retriever_doc_chunk._retrieve(query_bundle)

        print("retrieval done!")

        prompted_context, html_urls, doc_ids = get_context_retrieved_doc(retrieved_doc)
        
        return prompted_context, html_urls, doc_ids
    except Exception as e:
        print("ERROR: ", e)
        return "", [], []
