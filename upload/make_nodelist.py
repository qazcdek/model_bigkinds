import os
from tqdm import tqdm
from config import data_path
from language_model.embed_model import embed_helper
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter

class CustomTokenTextSplitter(TokenTextSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def split_text(self, text):
        encoded_text = self._tokenizer(text, return_tensors='pt',)
        token_ids = encoded_text["input_ids"].squeeze()[1:-1]
        splitted_text = []
        start_idx = 0
        len_text = len(token_ids)
        while start_idx < len_text:
            start_idx = max(0, start_idx-self.chunk_overlap)
            end_idx = min(start_idx+self.chunk_size, len_text)
            chunk = self._tokenizer.decode(token_ids[start_idx:end_idx])
            splitted_text.append(chunk)
            start_idx = end_idx
        return splitted_text

def make_nodelist(doc_textlist, is_pi=False):
    embed_class = embed_helper(dir_name="bge_base_onnx", embed_path='model', model_name="BAAI/bge-base-en-v1.5")
    embed_class.set_model()

    overall_str = 'document_overall_information'

    parser = CustomTokenTextSplitter(chunk_size=512, chunk_overlap=32)
    parser._tokenizer = embed_class.tokenizer_

    node_list_chunk = [[],[],[],[]]
    node_list_document = [[],[],[],[],[]]
    json_list_chunk = {}
    json_list_document = {}

    # k: document_number v: dict(metadata, textlist)
    for k in tqdm(doc_textlist):
        print(f"Start to make nodelist {k}")
        try:
            json_list_chunk[k] = []
            
            metadata= doc_textlist[k]['metadata']
                        
            str_metadata = {}
            for temp_key in metadata:
                str_metadata[temp_key] = str(metadata[temp_key])

            meta_str = ""
            for make_meta_str in metadata:
                if make_meta_str in ['html_url', 'frdoc_number', 'pdf_url', 'last_public_inspection_issue']:
                    continue
                elif make_meta_str == 'president':
                    meta_str += str(make_meta_str).strip() + ": " + str(metadata[make_meta_str]['identifier']).strip() + "\n"
                elif not metadata[make_meta_str] in [None, "", []]:
                    meta_str += str(make_meta_str).strip() + ": " + str(metadata[make_meta_str]).strip() + "\n"
            
            # print(type(metadata)) // <class 'dict'>
            textlist = []
            for text in doc_textlist[k]['textlist']:
                textlist.extend(parser.split_text(text))
            #textlist = custom_splitter(textlist)
            # // list [ str ]
            documents_chunk = [Document(text=t) for t in textlist]
            if len(documents_chunk) > 4096:
                print(f"{k} document body text length is over 4096!")
                documents_chunk = documents_chunk[:4096]
                with open(os.path.join(data_path, "over_length_document.csv"), 'a') as f:
                    f.write(f",{k}")
            full_text = "-- Metadata of Document:\n" + meta_str + "-- Document:\n" + "\n".join(textlist)
            # 
            nodes_chunk = parser.get_nodes_from_documents(documents_chunk)
            chunk_number = 0
            for node_chunk in nodes_chunk:
                node_chunk.id_ = k + "-" +str(chunk_number).zfill(4)
                chunk_number += 1
                node_chunk.metadata = str_metadata
                #node_text = node.get_content(metadata_mode="all")
                node_chunk_text = ""
                
                if not is_pi:
                    metakey_list = [overall_str, "abstract", "publication_date", "agency_names", "document_type"]
                else:
                    metakey_list = [overall_str, 'agency_names','publication_date','fr_type','last_updated']
                
                for metakey in metakey_list:
                    try:
                        node_chunk_text += metakey + ": " + str(node_chunk.metadata[metakey]).strip() + "\n"
                    except:
                        continue
                node_chunk_text += "body text:\n" + node_chunk.get_content(metadata_mode="none")
                node_chunk.embedding = embed_class.get_embedding(node_chunk_text)
                node_list_chunk[0].append(node_chunk.node_id)
                node_list_chunk[1].append(node_chunk.embedding)
                node_list_chunk[2].append(node_chunk.metadata)
                node_list_chunk[3].append(node_chunk.get_content(metadata_mode='none'))
                
                json_list_chunk[k].append({
                    "id": node_chunk.node_id, # str
                    "metadata": node_chunk.metadata, # dict
                    "text": node_chunk.get_content(metadata_mode="none")
                })
            node_ids=[node.node_id for node in nodes_chunk]
                
            ##########################################################################################        

            document_summary = embed_class.get_truncation(full_text)
            
            node_list_document[0].append(k)
            node_list_document[1].append(embed_class.get_embedding(document_summary))
            node_list_document[2].append(document_summary)
            node_list_document[3].append(metadata)
            node_list_document[4].append(node_ids)

            json_list_document[k] = {
                "id": k,
                "title": metadata['title'],
                "document_type": metadata['fr_type'],
                "agency_names": metadata['agency_names'],
                "publication_date": metadata['publication_date'],
                "summary": document_summary,
                "metadata": metadata,
                "nodeids": node_ids,
            }
        except:
            print(f"Failed to make nodelist {k}")
            pass

    return node_list_chunk, node_list_document, json_list_chunk, json_list_document

def make_nodelist_new(doc_textlist, is_pi=False):
    embed_class = embed_helper(dir_name="bge_base_onnx", embed_path='model', model_name="BAAI/bge-base-en-v1.5")
    embed_class.set_model()

    overall_str = 'document_overall_information'

    parser = CustomTokenTextSplitter(chunk_size=512, chunk_overlap=32)
    parser._tokenizer = embed_class.tokenizer_

    node_list_chunk = [[],[],[],[]]
    node_list_document = [[],[],[],[],[]]
    json_list_chunk = {}
    json_list_document = {}

    # k: document_number v: dict(metadata, textlist)
    for k in tqdm(doc_textlist):
        print(f"Start to make nodelist {k}")
        try:
            json_list_chunk[k] = []
            
            metadata= doc_textlist[k]['metadata']
            
            ##
            try:
                if metadata['document_type'] == "PR":
                    citation_number = ""
                elif type(metadata["executive_order_number"]) == str:
                    citation_number = "eo"+metadata["executive_order_number"]
                elif type(metadata["presidential_document_number"]) == str:
                    citation_number = metadata["presidential_document_number"]
                else:
                    citation_number = metadata["citation"]
            except:
                citation_number = ""
            metadata["citation_number"] = citation_number
            ##
            
            str_metadata = {}
            for temp_key in metadata:
                str_metadata[temp_key] = str(metadata[temp_key])

            meta_str = ""
            for make_meta_str in metadata:
                if make_meta_str in ['html_url', 'frdoc_number', 'pdf_url', 'last_public_inspection_issue']:
                    continue
                elif make_meta_str == 'president':
                    meta_str += str(make_meta_str).strip() + ": " + str(metadata[make_meta_str]['identifier']).strip() + "\n"
                elif not metadata[make_meta_str] in [None, "", []]:
                    meta_str += str(make_meta_str).strip() + ": " + str(metadata[make_meta_str]).strip() + "\n"
            
            # print(type(metadata)) // <class 'dict'>
            textlist = []
            for text in doc_textlist[k]['textlist']:
                textlist.extend(parser.split_text(text))
            #textlist = custom_splitter(textlist)
            # // list [ str ]
            documents_chunk = [Document(text=t) for t in textlist]
            if len(documents_chunk) > 4096:
                print(f"{k} document body text length is over 4096!")
                documents_chunk = documents_chunk[:4096]
                with open(os.path.join(data_path, "over_length_document.csv"), 'a') as f:
                    f.write(f",{k}")
            full_text = "-- Metadata of Document:\n" + meta_str + "-- Document:\n" + "\n".join(textlist)
            # 
            nodes_chunk = parser.get_nodes_from_documents(documents_chunk)
            chunk_number = 0
            for node_chunk in nodes_chunk:
                node_chunk.id_ = k + "-" +str(chunk_number).zfill(4)
                chunk_number += 1
                node_chunk.metadata = str_metadata
                #node_text = node.get_content(metadata_mode="all")
                node_chunk_text = ""
                
                if not is_pi:
                    metakey_list = [overall_str, "abstract", "publication_date", "agency_names", "document_type"]
                else:
                    metakey_list = [overall_str, 'agency_names','publication_date','fr_type','last_updated']
                
                for metakey in metakey_list:
                    try:
                        node_chunk_text += metakey + ": " + str(node_chunk.metadata[metakey]).strip() + "\n"
                    except:
                        continue
                node_chunk_text += "body text:\n" + node_chunk.get_content(metadata_mode="none")
                
                json_list_chunk[k].append({
                    "id": node_chunk.node_id, # str
                    "metadata": node_chunk.metadata, # dict
                    "text": node_chunk.get_content(metadata_mode="none")
                })
            node_ids=[node.node_id for node in nodes_chunk]
                
            ##########################################################################################        

            document_summary = embed_class.get_truncation(full_text)
            
            node_list_document[0].append(k)
            node_list_document[1].append(embed_class.get_embedding(document_summary))
            node_list_document[2].append(document_summary)
            node_list_document[3].append(metadata)
            node_list_document[4].append(node_ids)

            json_list_document[k] = {
                "id": k,
                "title": metadata['title'],
                "document_type": metadata['fr_type'],
                "agency_names": metadata['agency_names'],
                "publication_date": metadata['publication_date'],
                "summary": document_summary,
                "metadata": metadata,
                "nodeids": node_ids,
            }
        except:
            print(f"Failed to make nodelist {k}")
            pass

    return node_list_chunk, node_list_document, json_list_chunk, json_list_document