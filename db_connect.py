from pymongo import MongoClient
from config import MONGODB_URL, MONGODB_DB_NAME

client = MongoClient(MONGODB_URL)
db = client[MONGODB_DB_NAME]
document_collection = db["documents_FR"]
chunk_collection = db["chunks_FR"]
public_collection = db['public']
public_document_collection = db['documents_PI']