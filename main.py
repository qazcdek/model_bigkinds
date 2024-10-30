import sys
sys.path.append('/home/ec2-user/chat-model')

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from config import MILVUS_URI

from upload.make_nodelist import make_nodelist, make_nodelist_new
from upload.mongo_control import mongo_insert_chunks_fr, mongo_insert_documents_fr, mongo_insert_chunks_pr
from upload.milvus_control import *
from retriever.extract import using_text_query, extract_chunks_from_bookshelf

from config import *
from db_connect import document_collection, public_collection, public_document_collection

class DocTextList(BaseModel):
    doc_textlist : Any

class PrInput(BaseModel):
    chunk : Any
    document : Any

class ChunkInput(BaseModel):
    json_chunk_list : Any
    bookshelf_id : Any
    collection_name : Any

class DeleteChunkInput(BaseModel):
    document_list : Any
    bookshelf_id : Any
    collection_name : Any

##240516
class DropBookshelfInput(BaseModel):
    bookshelf_id : Any
    collection_name : Any

class QueryModel(BaseModel):
    search_query: str
    collection_name: str = "documents_FR"
    similarity_top_k: int = 60

##240521
class VectorSearchModel(BaseModel):
    doc_id: str
    collection_name: str = "documents_FR"
    level: int = 2
    
class ExtractChunkInput(BaseModel):
    search_query : Any
    bookshelf_id : Any
    collection_name : Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_upload_fr(doc_textlist):
    try:
        node_list_chunk, node_list_document, json_list_chunk, json_list_document = make_nodelist_new(doc_textlist)
        upload_documents_fr(node_list=node_list_document)
        mongo_insert_chunks_fr(db_name=MONGODB_DB_NAME, collection_name='chunks_FR', node_list=json_list_chunk, node_type="json")
        mongo_insert_documents_fr(db_name=MONGODB_DB_NAME, collection_name='documents_FR', node_list=json_list_document, node_type="json")
        print('Data upload done')

    except Exception as e:
        print(f'An error occurred during upload FR: {e}')
        raise e

@app.post("/upload_fr/")
async def upload_fr_endpoint(item: DocTextList, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_upload_fr, item.doc_textlist)
    return {"message": "Upload fr started"}


### pi
##############################
def process_upload_pi(doc_textlist):
    try:
        node_list_chunk, node_list_document, json_list_chunk, json_list_document = make_nodelist_new(doc_textlist, is_pi=True)
        ### mongo_insert_chunks_fr(db_name=MONGODB_DB_NAME, collection_name='chunks_PI', node_list=json_list_chunk, node_type="json")
        ### result_dict[y["_id"]] = y
        ####
        result_dict = {} # {"2023-12345":[{"id":"2023-12345-0001", "metadata":{}, "text":"dkjfdfj"},{...}]}
        for k in json_list_chunk:
            for chunk_dict in json_list_chunk[k]:
                try:
                    result_dict[chunk_dict['id']] = {"_id":chunk_dict['id'], "metadata":chunk_dict['metadata'], "text":chunk_dict['text']}
                except:
                    pass
            #result_dict[json_list_chunk[k]['id']] = json_list_chunk[k]
            #result_dict[json_list_chunk[k]['id']]["_id"] = result_dict[json_list_chunk[k]['id']]["id"]
        print('Data upload done')

    except Exception as e:
        print(f'An error occurred during upload PI(result dict): {e}')
        raise e
    public_document_collection.drop()
    mongo_insert_documents_fr(db_name=MONGODB_DB_NAME, collection_name='documents_PI', node_list=json_list_document, node_type="json")
    bookshelf = public_collection.find_one({"id": "public_inspection"}, {"_id": 0})
    time_now = datetime.now()
    if bookshelf == None:
        data_dict = {"id":"public_inspection","name":"public_inspection","list_mode":'list_only',"doc_list":[],"regist_dt":time_now,"update_dt":time_now}
        public_collection.insert_one(data_dict)
        
    new_doc_list = list(result_dict.keys())
    public_collection.update_one({"id":"public_inspection"}, {"$set":{"doc_list":new_doc_list, "update_dt":time_now}})
    
    # chunk_public의 파티션을 드랍하지 않고 안에 있는 데이터를 업데이트 할 경우 
    """doc_list = bookshelf.get("doc_list")
    remove_doc_list = [doc for doc in doc_list if doc not in new_doc_list]
    delete_chunks_by_json(remove_doc_list, collection_name="chunks1", bookshelf_id="public_inspection", milvus_db=milvus_db_fr, milvus_user=milvus_user, milvus_password=milvus_password, milvus_uri=milvus_uri, milvus_token=milvus_token, overwrite=False, dim=768)"""
    
    try:
        drop_partition(partition_name="public_inspection", collection_name="chunks_public")
        upload_chunks_by_json(json_data=result_dict, bookshelf_id="public_inspection", collection_name="chunks_public")
        print('Data upload done')
        return {"message": "Chunk upload done"}
    except Exception as e:
        print(f'An error occurred during upload PI(milvus): {e}')
        raise HTTPException(status_code=500, detail="An error occurred during upload")

@app.post("/upload_pi/")
async def upload_pi_endpoint(item: DocTextList, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_upload_pi, item.doc_textlist)
    return {"message": "Upload pi started"}

#################################

def process_upload_pr(json_chunk, json_doc):
    try:
        upload_documents_by_json(json_data=json_doc)
        mongo_insert_chunks_pr(db_name=MONGODB_DB_NAME, collection_name='chunks_FR', node_list=json_chunk, node_type="json")
        mongo_insert_documents_fr(db_name=MONGODB_DB_NAME, collection_name='documents_FR', node_list=json_doc, node_type="json")
        print('Data upload done')

    except Exception as e:
        print(f'An error occurred during upload PR: {e}')
        raise e

@app.post("/upload_pr/")
async def upload_fr_endpoint(item: PrInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_upload_pr, item.chunk, item.document)
    return {"message": "Upload pr started"}

@app.post("/upload_chunk/")
async def upload_chunk_endpoint(item: ChunkInput):
    try:
        upload_chunks_by_json(json_data=item.json_chunk_list, bookshelf_id=item.bookshelf_id, collection_name=item.collection_name)
        print('Data upload done')
        return {"message": "Chunk upload done"}
    except Exception as e:
        print(f'An error occurred during upload CHUNK: {e}')
        raise HTTPException(status_code=500, detail="An error occurred during upload")
    
@app.post("/delete_chunk/")
async def upload_fr_endpoint(item: DeleteChunkInput):
    try:
        delete_chunks_by_json([item.document_list], bookshelf_id=item.bookshelf_id, collection_name=item.collection_name) ### doc_id가 str이라 list로 감싼다
        print('Data delete done')
        return {"message": "Delete upload done"}
    except Exception as e:
        print(f'An error occurred during delete CHUNK: {e}')
        raise HTTPException(status_code=500, detail="An error occurred during delete")
    
@app.post("/drop_bookshelf/")
async def drop_bookshelf_endpoint(item: DropBookshelfInput):
    try:
        drop_partition(partition_name=item.bookshelf_id, collection_name=item.collection_name) 
        print('Drop partition done')
        return {"message": "Drop partition done"}
    except Exception as e:
        print(f'An error occurred during drop PARTITION: {e}')
        raise HTTPException(status_code=500, detail="An error occurred during drop")

@app.post("/search/")
async def search_documents(query_model: QueryModel):
    try:
        doc_id_list = using_text_query(search_query=query_model.search_query, return_ids=True)
        return {"document_ids": doc_id_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_chunk/")
async def api_extract_chunks_from_bookshelf(item: ExtractChunkInput):
    print("MILBUS URI : ", MILVUS_URI)
    try:
        prompted_context, html_urls, doc_ids=extract_chunks_from_bookshelf(search_query=item.search_query, bookshelf_id=item.bookshelf_id, collection_name=item.collection_name)

        return {"prompted_context":prompted_context, "html_urls":html_urls, "doc_ids":doc_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##240521
@app.post("/search_related_doc/")
async def search_documents(vector_search_model: VectorSearchModel):
    try:
        doc_id_list = document_vector_search(doc_id=vector_search_model.doc_id, return_ids=True)
        return {"document_ids": doc_id_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/test")
async def test():
    cnt=document_collection.count_documents({})
    print(cnt)
    return {"cnt": cnt}

@app.get("/test3")
async def test():
    print('Hello')
    return 'hello'
