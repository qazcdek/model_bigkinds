from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import tiktoken
import os
from config import OPENAI_API_KEY

#from config import embed_path

from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore
#https://huggingface.co/blog/optimum-inference#36-evaluate-the-performance-and-speed

import torch
from transformers import AutoModel, AutoTokenizer

import argparse

class embed_helper():
    def __init__(self, dir_name, model_name, embed_path=None):
        self.dir_name = dir_name
        self.model_name = model_name
        self.embed_path = embed_path
        self.model_ = None
        self.tokenizer_ = None
        self.device = 'cpu'

    def download_embed(self):
        if self.model_name == "openai":
            return
        onnx_dir = os.path.join(self.embed_path, self.dir_name)
        if not os.path.isdir(onnx_dir):
            OptimumEmbedding.create_and_save_optimum_model(
                self.model_name, onnx_dir
        )
        return 
    def set_model(self):
        self.model_, self.tokenizer_=self.load_embedmodel()
    def load_embedmodel(self):
        if self.model_name == "openai":
            # 나중에 dimension 받아오기
            model = OpenAIEmbedding(model="text-embedding-3-large", dimensions=768, api_key=OPENAI_API_KEY)
            tokenizer = tiktoken.get_encoding("cl100k_base")
            return model, tokenizer
        else:
            model_path = os.path.join(self.embed_path, self.dir_name)
            if not os.path.exists(model_path):
                self.download_embed()            
            if self.device == 'cuda':
                model_ort = ORTModelForFeatureExtraction.from_pretrained(model_path, file_name="model.onnx",) #provider="CUDAExecutionProvider",)
            else:
                model_ort = ORTModelForFeatureExtraction.from_pretrained(model_path, file_name="model.onnx")
            tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side='right')
            return model_ort, tokenizer
    def get_truncation(self, texts, max_token=512):
        if self.model_name == 'openai':
            return self.tokenizer_.decode(self.tokenizer_.encode(texts)[:512])
        encoded_input = self.tokenizer_(texts, padding=False, truncation=True, return_tensors='pt', max_length=512)
        return self.tokenizer_.decode(encoded_input["input_ids"].squeeze()[1:-1])

    def get_embedding(self, texts):
        if self.model_name == 'openai':
            print('openai yes')
            result = self.model_.get_text_embedding(texts)
            print(len(result))
            print(type(result))
            print(type(result[0]))
            return result
        else:
            # Sentences we want sentence embeddings for
            sentences = texts
            
            # Tokenize sentences
            encoded_input = self.tokenizer_(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
            # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
            # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

            model_output_ort = self.model_(**encoded_input)
            # Compute token embeddings
            #print(len(model_output_ort[0][:, 0].squeeze().numpy().tolist()))
            return model_output_ort[0][:, 0].squeeze().to('cpu').numpy().tolist()
    def get_embedding_test(self, texts):
        if self.model_name == 'openai':
            print('openai yes')
            result = self.model_._get_text_embeddings(texts)
            print(len(result))
            print(type(result))
            print(type(result[0]))
            return result

def main(dir_name, model_name, download=False, test_str=None):
    embed_class = embed_helper(dir_name=dir_name, model_name=model_name)
    if download:
        embed_class.download_embed()
    if test_str != None:
        embed_class.set_model()
        print(embed_class.get_embedding(test_str))
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", dest='dir_name', default='bge_base_onnx')
    parser.add_argument("--model_name", dest='model_name', default="BAAI/bge-base-en-v1.5")
    return parser.parse_args()

# how to use
if __name__ == "__main__":
    args = get_args()
    main(dir_name=args.dir_name, model_name=args.model_name, download=True, test_str="Embedding success!")
    """embed_class = embed_helper(dir_name='bge_onnx', model_name="BAAI/bge-small-en-v1.5")
    embed_class.download_embed()
    embed_class.set_model()
    print(embed_class.get_embedding("Embedding success!"))"""