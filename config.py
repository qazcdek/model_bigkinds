import os
# from vault_utils import fetch_vault_secrets

# secrets = fetch_vault_secrets()

# for key, value in secrets.items():
#     os.environ[key] = value


MONGODB_URL = str(os.getenv("MONGODB_URL"))
MONGODB_DB_NAME = str(os.getenv("MONGODB_DB_NAME"))
OPENAI_TOKEN = str(os.getenv("OPENAI_TOKEN"))
data_path = './data/'
collection_list = ['documents_FR', 'chunks_FR', 'documents_PI']
already_milvus = "already_exist_milvus.json"
MILVUS_USER=str(os.getenv("MILVUS_USER"))
MILVUS_PASSWORD=str(os.getenv("MILVUS_PASSWORD"))
MILVUS_DB_FR=str(os.getenv("MILVUS_DB_FR"))
MILVUS_URI=str(os.getenv("MILVUS_URI"))
MILVUS_TOKEN=str(os.getenv("MILVUS_TOKEN"))
# OPENAI_API_KEY=str(os.getenv("OPENAI_TOKEN"))
# MONGODB_URL = "mongodb+srv://infovisor:tjJldvnMc56Jj6n3@infovisor.vufvd.mongodb.net/?retryWrites=true&w=majority"
# MONGODB_DB_NAME = "infovisor"
# SECRET_KEY = "eeyoontaek"
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 525600
# data_path = './data/'
# collection_list = ['documents_FR', 'chunks_FR']
# already_milvus = "already_exist_milvus.json"
# MILVUS_USER='db_admin'
# MILVUS_PASSWORD='Qy9,8]Aja0<,{|6n'
# MILVUS_DB_FR='Cluster-01'
# MILVUS_URI='https://in01-7505d1216351bdd.aws-us-east-2.vectordb.zillizcloud.com:19532'
# MILVUS_TOKEN='60819dc0f177bd9c11ddcc26eec9598b19e32604282a5fb0143def163bfe7be7b2dbd482ddcd623bc6a2fc865995365a2e7df1e4'
OPENAI_API_KEY=str(os.getenv("OPENAI_API_KEY"))