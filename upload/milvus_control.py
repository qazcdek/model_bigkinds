# (save)mongoDB => documentDB 
# (load)documentDB => mongoDB
import json
import os
import sys
import re

from config import MILVUS_DB_FR, MILVUS_USER, MILVUS_PASSWORD, MILVUS_URI, MILVUS_TOKEN, already_milvus

# create milvus
from pymilvus import connections, utility, MilvusClient, Partition, Collection
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from language_model.embed_model import embed_helper
from tqdm import tqdm
import time

milvus_db_fr = MILVUS_DB_FR
milvus_user = MILVUS_USER
milvus_password = MILVUS_PASSWORD
milvus_uri = MILVUS_URI
milvus_token = MILVUS_TOKEN

# documents_FR, chunks_FR
def creat_milvus_collection(collection_name, milvus_db=milvus_db_fr, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token, dim=768, overwrite=False, **kwargs):
    #assert collection_name in collection_list, f"collection name must be {collection_list}"
    assert "chunks" in collection_name or "documents" in collection_name, f"collection name must contain 'chunks' or 'documents'"
    # milvus args: collection name, uri, token
    milvus_settings = {"collection_name":collection_name, "milvus_uri":milvus_uri, "milvus_token":milvus_token}
    if not milvus_token:
        milvus_token = f"{milvus_user}:{milvus_password}"
    try:
        connections.connect(
            uri=milvus_uri,
            token=milvus_token)
        print(f"Connect to DB: Success")
    except:
        print(f"Failed to connect, please check MILVUS URI/TOKEN.")
        sys.exit()
    
    check_collection = utility.has_collection(collection_name)
    # if collection already exist, overwrite or quit
    
    if check_collection:
        if overwrite:
            drop_result = utility.drop_collection(collection_name)
        else:
            print(f"Collection named '{collection_name}' already exists, please set overwrite arg True or select different collection name.")
            connections.disconnect("default")
            sys.exit()

    print("start to create schema!")
    # create a collection with customized primary field: book_id_field
    if 'chunks' in collection_name:
        id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=32, description="customized primary id")
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim, description="vector embeddings for text")
        metadata_field = FieldSchema(name="metadata", dtype=DataType.JSON)
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, description="raw text", max_length=8192)

        schema = CollectionSchema(fields=[id_field, embedding_field, metadata_field, text_field], description="collection for Federal Register documents (chunks)")
        print(f"Creating example collection: {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        connections.disconnect("default")
    
    elif 'documents' in collection_name:
        id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=32, description="customized primary id")
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim, description="vector embeddings for text")
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, description="abstract of document", max_length=8192)
        metadata_field = FieldSchema(name="metadata", dtype=DataType.JSON)
        nodeids_field = FieldSchema(name="node_ids", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=4096, max_length=32, description="node ids for each chunk in document")
        
        schema = CollectionSchema(fields=[id_field, embedding_field, text_field, metadata_field, nodeids_field,],
                                description="collection for Federal Register documents (documents)")
        print(f"Creating example collection: {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        connections.disconnect("default")
    else:
        print("Collection name error!")

    return milvus_settings

def upload_documents_by_json(json_data=None, json_path=None, collection_name='documents_FR', milvus_db=milvus_db_fr, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token, overwrite=False, dim=768, create_collection=False, dir_name="bge_base_onnx", model_name="BAAI/bge-base-en-v1.5"):
    assert 'documents' in collection_name, "collection name must be 'documents_FR'"
    assert json_data != None or json_path != None, "json data or json path should be needed"

    if json_path is not None:
        if 'document' in collection_name and 'document' not in os.path.basename(json_path):
            print("data type of json file should match with collection type")
            return
        with open(json_path) as f:
            json_data = json.load(f)
    
    try:
        connections.connect(
            uri=milvus_uri,
            token=milvus_token)
    except:
        print(f"Failed to connect, please check MILVUS URI/TOKEN.")
        sys.exit()

    check_collection = utility.has_collection(collection_name)
    if not check_collection:
        if create_collection:
            connections.disconnect("default")
            creat_milvus_collection(collection_name, milvus_uri=milvus_uri, milvus_db=milvus_db, milvus_user=milvus_user, milvus_password=milvus_password, milvus_token=milvus_token, dim=dim, overwrite=overwrite)
            connections.connect(
                        uri=milvus_uri,
                        token=f"{milvus_user}:{milvus_password}")
        else:
            print(f"'{collection_name}' collection does not exist. Please create collection before upload")
            connections.disconnect("default")
            sys.exit()
    
    embed_class = embed_helper(dir_name="bge_base_onnx", embed_path='model', model_name="BAAI/bge-base-en-v1.5")
    embed_class.set_model()
    
    key_list = list(json_data.keys())
    start_idx = 0
    if "document" in collection_name:
        result_list = [[],[],[],[],[]]

    total_step = len(key_list) // 1024 + 1
    present_step = 1
    while start_idx < len(key_list):
        print(f"upload process: {present_step} / {total_step}")
        end_idx = min(start_idx + 1024, len(key_list))
        if "document" in collection_name:
            # id, embedding, text, metadata, node_ids
            result_list = [[],[],[],[],[]]
            for key in key_list[start_idx:end_idx]:
                text = json_data[key]['summary']
                truncated_text = embed_class.get_truncation(text)
                embedding = embed_class.get_embedding(truncated_text)
                result_list[0].append(json_data[key]['id'])
                result_list[1].append(embedding)
                result_list[2].append(json_data[key]['summary'])
                result_list[3].append(json_data[key]['metadata'])
                result_list[4].append(json_data[key]['nodeids'])
            upload_documents_fr(result_list, collection_name=collection_name, milvus_uri=milvus_uri, milvus_token=milvus_token, overwrite=False)
        present_step += 1
        start_idx = end_idx
    connections.disconnect("default")
    
def delete_chunks_by_json(document_list, collection_name, bookshelf_id, milvus_db=milvus_db_fr, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token, overwrite=False, dim=768):
    assert 'chunks' in collection_name, "collection name must contain 'chunks'"
    if not milvus_token:
        milvus_token = f"{milvus_user}:{milvus_password}"
    #connet to cluster
    try:
        connections.connect(
            uri=milvus_uri,
            token=milvus_token)
        print(f"Connect to DB: Success")
    except:
        print(f"Failed to connect, please check MILVUS URI/TOKEN.")
        sys.exit()
    # check collection
    check_collection = utility.has_collection(collection_name)
    if not check_collection:
        print(f"'{collection_name}' collection does not exist. Please check collection name before delete data")
        connections.disconnect("default")
        sys.exit()
    
    expr_list = [f"id like \"{doc_str}%\"" for doc_str in document_list]
    expr = " or ".join(expr_list)
    try:
        partition = Partition(collection=collection_name, name=bookshelf_id)
        partition.delete(
                    expr=expr,
                    partition_names=[bookshelf_id],
                )
    except:
        print("failed to delete documents list")

    connections.disconnect("default")
    return

def upload_documents_fr(node_list, collection_name='documents_FR', milvus_db=milvus_db_fr, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token, overwrite=False, dim=768, create_collection=True):
    assert ('documents' in collection_name), "only for uploading documents"
    append_milvus_list = []
    # connect to milvus cloud
    if not milvus_token:
        milvus_token = f"{milvus_user}:{milvus_password}"
    try:
        connections.connect(
            uri=milvus_uri,
            token=milvus_token)
        print(f"Connect to DB: Success")
    except:
        print(f"Failed to connect, please check MILVUS URI/TOKEN.")
        sys.exit()
    # check and overwrite or quit
    check_collection = utility.has_collection(collection_name)
    if not check_collection:
        if create_collection:
            connections.disconnect("default")
            creat_milvus_collection(collection_name, milvus_db=milvus_db, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token, dim=dim, overwrite=overwrite)
            connections.connect(
                        uri=milvus_uri,
                        token=f"{milvus_user}:{milvus_password}")
        else:
            print(f"'{collection_name}' collection does not exist. Please create collection before upload")
            connections.disconnect("default")
            sys.exit()
        
    print(f"Connect to DB: Success")
    
    collection = Collection(name=collection_name)
    print(f"Load '{collection_name}' Collection: Success!")
    
    ins_resp = collection.upsert(node_list)

    print("flush start!")
    collection.flush()

    # later consider: index type
    index_params = {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}}
    collection.create_index(field_name="embedding", index_params=index_params)
    print("flush done!")

    connections.disconnect("default")

    # already_milvus update
    try:
        with open(already_milvus, 'r') as f:
            exist_milvus = json.load(f) # // dict('chunks', 'documents')
    except:
        exist_milvus = {'chunks':[], 'documents':[]}
    with open(already_milvus, 'w') as f:
        exist_milvus['documents'] = exist_milvus['documents'] + append_milvus_list
        json.dump(exist_milvus, f)

def upload_chunks_by_json(json_data, collection_name, bookshelf_id, milvus_db=milvus_db_fr, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token, overwrite=False, dim=768, create_collection=True, dir_name="bge_base_onnx", model_name="openai"):
    assert 'chunks' in collection_name, "collection name must contain 'chunks'"
    partition_name=bookshelf_id
    append_milvus_list = []
    embed_class = embed_helper(dir_name=dir_name, embed_path='model', model_name=model_name)
    embed_class.set_model()

    key_list = list(json_data.keys())

    if not milvus_token:
        milvus_token = f"{milvus_user}:{milvus_password}"
    try:
        connections.connect(
            uri=milvus_uri,
            token=milvus_token)
        print(f"Connect to DB: Success")
    except:
        print(f"Failed to connect, please check MILVUS URI/TOKEN.")
        sys.exit()

    check_collection = utility.has_collection(collection_name)
    if not check_collection:
        if create_collection:
            creat_milvus_collection(collection_name, milvus_uri=milvus_uri, milvus_db=milvus_db, milvus_user=milvus_user, milvus_password=milvus_password, milvus_token=milvus_token, dim=dim, overwrite=overwrite)
        else:
            print(f"'{collection_name}' collection does not exist. Please create collection before upload")
            sys.exit()

    collection = Collection(name=collection_name)
    print(f"Load '{collection_name}' Collection: Success!")

    has_partition = utility.has_partition(collection_name=collection_name, partition_name=partition_name)
    if not has_partition:
        collection.create_partition(partition_name=partition_name)
    bookshelf = Partition(collection=collection_name, name=partition_name)
    bookshelf.load()

    start_idx = 0
    while start_idx < len(key_list):
        end_idx = min(start_idx + 1024, len(key_list))

        # id, embedding, text, metadata
        target_chunk_id=key_list[start_idx:end_idx]
        expr = f"id in {str(target_chunk_id)}"
        res = bookshelf.query(
            expr=expr,
            output_fields = ["id"]
        )
        exist_chunk_id = list(map(lambda k: k['id'], res))
        non_exist_chunk_id=list(set(target_chunk_id)-set(exist_chunk_id))
        if len(non_exist_chunk_id) < 1:
            pass
        else:
            result_list = [[],[],[],[]]
            for key in tqdm(non_exist_chunk_id):
                text = json_data[key]['text']
                truncated_text = embed_class.get_truncation(text)
                result_list[0].append(json_data[key]['_id'])
                result_list[1].append(truncated_text)
                result_list[2].append(json_data[key]['metadata'])
                result_list[3].append(json_data[key]['text'])
            tmp=[]
            for i in  range(1, (len(result_list[1])//2000)+2):
                tmp.extend(embed_class.get_embedding_test(result_list[1][(i-1)*2000:i*2000]))
            result_list[1] = tmp
            ins_resp = bookshelf.insert(result_list)
        start_idx = end_idx

    index_params = {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}}
    collection.create_index(field_name="embedding", index_params=index_params)
    print("flush start!")
    collection.flush()
    print("flush done!!!")
    connections.disconnect("default")

def find_chunkid_expr(collection_name, element_list, query_expr, milvus_db=milvus_db_fr, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token):
    assert "chunks" in collection_name, "Find the document just only for chunks collections"
    # eg. query_expr = r'metadata["document_number"] == "{ele}"'
    if not milvus_token:
        milvus_token = f"{milvus_user}:{milvus_password}"
    try:
        connections.connect(
            uri=milvus_uri,
            token=milvus_token)
        print(f"Connect to DB: Success")
    except:
        print(f"Failed to connect, please check MILVUS URI/TOKEN.")
        sys.exit()
    # check and overwrite or quit
    check_collection = utility.has_collection(collection_name)
    if not check_collection:
        print(f"{collection_name} collection does not exists")
        connections.disconnect("default")
        sys.exit()
    collection = Collection(name=collection_name)
    print("Load Collection: Success!")
    
    result_ids = []
    tried_doc = 0
    for ele in element_list:
        try:
            res = collection.query(
                expr=re.sub(r"{ele}", ele, query_expr),
                output_fields=['id'],
            )
            tried_doc += 1
        except:
            print(f"try number: {tried_doc}")
            print(ele)
            sys.exit()
        if len(res) > 0:
            result_ids.extend(res)
    result_ids = [result['id'] for result in result_ids]
    print("Done with verifying existing documents")
    #print(f"{exist_doc} already exist in milvus DB")
    return result_ids

def find_partitions(collection_name, milvus_db=milvus_db_fr, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token, stringify=True):
    connections.connect(
        uri=milvus_uri,
        token=milvus_token)
    check_collection = utility.has_collection(collection_name)
    if not check_collection:
        print(f"{collection_name} collection does not exists")
        connections.disconnect("default")
        sys.exit()
    collection = Collection(name=collection_name)
    if stringify:
        temp_conn = collection._get_connection()
        partitions = temp_conn.list_partitions(collection_name)
    else:
        partitions = collection.partitions
    return partitions

##240521
def document_vector_search(doc_id, collection_name='documents_FR', return_ids=False):
    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    check_collection = utility.has_collection(collection_name)
    if not check_collection:
        print(f"{collection_name} collection does not exists")
        connections.disconnect("default")
        sys.exit()
    collection = Collection(name=collection_name)
    collection.load()
    search_params = {"metric_type": "COSINE", 
                     "offset": 0, 
                     "ignore_growing": False, 
                     "params": {"level": 2}
                     }
    result_doc_vector = collection.query(
        expr=f'id == "{doc_id}"',
        limit=1,
        output_fields=["embedding"]
    )
    doc_vector = result_doc_vector[0]["embedding"]
    results = collection.search(
        data = [doc_vector],
        anns_field="embedding",
        param=search_params,
        limit=20,
        output_fields=["id"],
        )
    if return_ids:
        return results[0].ids
    return results

##240516
def drop_partition(collection_name, partition_name, milvus_db=milvus_db_fr, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token):
    connections.connect(
        uri=milvus_uri,
        token=milvus_token)
    collection = Collection(name=collection_name)
    if collection.has_partition(partition_name):
        partition = Partition(collection=collection_name, name=partition_name)
        partition.release()
        collection.drop_partition(partition_name)
    else:
        print(f"{partition_name} partition doesn't exist.")    

