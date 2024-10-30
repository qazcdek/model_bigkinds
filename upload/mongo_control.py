from db_connect import client

# 각 문서로부터 
# for each document => document node, [chunk node] \
# => insert CHUNKS data to mongoDB and get ids \
# => insert DOCUMENT data to mongoDB and get id \
# => insert data(docs, chunks) to milvusDB(ids, embeddings)
from config import collection_list
from pymongo import UpdateOne
import re
from language_model.embed_model import embed_helper
from upload.make_nodelist import CustomTokenTextSplitter
from datetime import datetime, timedelta
import pytz

def mongo_insert_chunks_fr(db_name, collection_name, node_list, client=client, node_type="json"):
    assert "chunks" in collection_name, "should be chunks DB"
    db = client[db_name]
    collection = db[collection_name]
    entities = {}
    chunk_ids = {}
    if node_type == "node":
    # k: document_number
        for k in node_list:
            entities[k] = []
            for node in node_list[k]:
                entities[k].append({
                    "_id":node.node_id,
                    "text":node.get_text(),
                    "metadata":node.metadata,
                    "doc_id":node.node_id[:-5],
                    #"embedding":node.embedding,
                    })
            upserts = [UpdateOne({"_id":x["_id"]}, {'$set':x}, upsert=True) for x in entities[k]]
            result = collection.bulk_write(upserts)
            #result = collection.insert_many(entities[k])
            chunk_ids[k] = result.upserted_ids
        # ids:dict => key: document_number, value: list of ids for chunks 
    elif node_type == "json":
        for k in node_list:
            entities[k] = []
            for node in node_list[k]:
                entities[k].append({
                    "_id":node["id"],
                    "text":node['text'],
                    "metadata":node['metadata'],
                    'doc_id':node["id"][:-5],
                    #"embedding":node.embedding,
                    })
            upserts = [UpdateOne({"_id":x["_id"]}, {'$set':x}, upsert=True) for x in entities[k]]
            result = collection.bulk_write(upserts)
            #result = collection.insert_many(entities[k])
            chunk_ids[k] = result.upserted_ids
        # ids:dict => key: document_number, value: list of ids for chunks 
    return chunk_ids

# find chunk data to send chat model => return format {id: "_id":id, "text":text, "metadata":metadata}
def mongo_find_chunks(db_name, collection_name, documents_list, client=client, node_type='json'):
    db = client[db_name]
    collection = db[collection_name]
    result_dict = {}

    for document in documents_list:
        rgx = re.compile(f'{document}.*', re.IGNORECASE)
        x = collection.find({"_id":rgx})
        x = list(x)
        if len(x) > 0 or x != None:
            for y in x:
                result_dict[y["_id"]] = y
    print(result_dict)
    return result_dict
        
# daily update documentDB
def mongo_insert_documents_fr(db_name, collection_name, node_list, client=client, node_type='json'):
    assert collection_name in collection_list, f"{collection_name} is not accepted"
    assert "documents" in collection_name, "should be documents DB"
    db = client[db_name]
    collection = db[collection_name]
    entities = []
    doc_ids = {}

    if node_type == "node":
        # id, title, citation_number, agency, publication_date, abstract, chunk_id,
        for k in node_list:
            node = node_list[k]
            try:
                if type(node.metadata["executive_order_number"]) == str:
                    citation_number = "eo"+node.metadata["executive_order_number"]
                elif type(node.metadata["presidential_document_number"]) == str:
                    citation_number = node.metadata["presidential_document_number"]
                else:
                    citation_number = node.metadata["citation"]
            except:
                citation_number = ""
            node.metadata["citation_number"] = citation_number
            
            entities.append({
                "_id":k,
                "title":node.metadata['title'],
                "document_type":node.metadata['type'],
                "agency_names":node.metadata['agency_names'],
                "publication_date":node.metadata['publication_date'],
                "summary":node.get_content(metadata_mode="none"),
                "metadata":node.metadata,
                "nodeids":node.metadata['nodeids'][k],
                #"embedding":node.embedding,
                })
        upserts = [UpdateOne({"_id":x["_id"]}, {'$set':x}, upsert=True) for x in entities]
        result = collection.bulk_write(upserts)
        doc_ids[k] = result.upserted_ids
        #collection_news.delete_many({'created_dt':{"$lt":oneday_before_str}})
        #result = collection.insert_many(entities)  
    elif node_type == "json":
        # id, title, citation_number, agency, publication_date, abstract, chunk_id,
        for k in node_list:
            node = node_list[k]
            try:
                if node['document_type'] == "PR":
                    citation_number = ""
                elif type(node["metadata"]["executive_order_number"]) == str:
                    citation_number = "eo"+node["metadata"]["executive_order_number"]
                elif type(node["metadata"]["presidential_document_number"]) == str:
                    citation_number = node["metadata"]["presidential_document_number"]
                else:
                    citation_number = node["metadata"]["citation"]
            except:
                citation_number = ""
            node["metadata"]["citation_number"] = citation_number
            
            entities.append({
                "_id":node["id"],
                "title":node['title'],
                "document_type":node['document_type'],
                "agency_names":node['agency_names'],
                "publication_date":node['publication_date'],
                "summary":node["summary"],
                "metadata":node["metadata"],
                "nodeids":node['nodeids'],
                #"embedding":node.embedding,
                })
        upserts = [UpdateOne({"_id":x["_id"]}, {'$set':x}, upsert=True) for x in entities]
        result = collection.bulk_write(upserts)
        doc_ids[k] = result.upserted_ids
        #result = collection.insert_many(entities)    
    return doc_ids

def mongo_insert_chunks_pr(db_name, collection_name, node_list, client=client, node_type="json"):
    assert "chunks" in collection_name, "should be chunks DB"
    embed_class = embed_helper(dir_name="bge_base_onnx", embed_path='model', model_name="BAAI/bge-base-en-v1.5")
    embed_class.set_model()

    parser = CustomTokenTextSplitter(chunk_size=512, chunk_overlap=32)
    parser._tokenizer = embed_class.tokenizer_
    
    db = client[db_name]
    collection = db[collection_name]
    entities = {}
    chunk_ids = {}
    if node_type == "json":
        for k in node_list:
            chunk_num = 0
            entities[k] = []
            for node in node_list[k]:
                text_list = parser.split_text(node['text'])
                for splitted_text in text_list:
                    entities[k].append({
                        "_id":node["id"][:-4]+str(chunk_num).zfill(4),
                        "text":splitted_text,
                        "metadata":node['metadata'],
                        "doc_id":node["id"][:-4],
                        #"embedding":node.embedding,
                        })
                    chunk_num += 1
            upserts = [UpdateOne({"_id":x["_id"]}, {'$set':x}, upsert=True) for x in entities[k]]
            result = collection.bulk_write(upserts)
            #result = collection.insert_many(entities[k])
            chunk_ids[k] = result.upserted_ids
        # ids:dict => key: document_number, value: list of ids for chunks 
    return chunk_ids
