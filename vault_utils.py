import hvac
import os

def fetch_vault_secrets():
    client = hvac.Client(
        url=os.getenv('VAULT_ADDR'),
        token=os.getenv('VAULT_TOKEN')
    )

    milvus_secrets = client.secrets.kv.v2.read_secret_version(
        path='secret/milvus/secret'
    )
    milvus_data = milvus_secrets['data']['data']

    db_secrets = client.secrets.kv.v2.read_secret_version(
        path='secret/db/secret'
    )
    db_data = db_secrets['data']['data']

    model_secrets = client.secrets.kv.v2.read_secret_version(
        path='secret/model/config'
    )
    model_data = model_secrets['data']['data']

    all_secrets = {**milvus_data, **db_data, **model_data}

    print("All secrets: ", all_secrets)

    return all_secrets