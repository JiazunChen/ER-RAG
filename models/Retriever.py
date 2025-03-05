from langchain.retrievers import EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import json
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from time import time
from tqdm import tqdm
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import TFIDFRetriever
from langchain.retrievers import MergerRetriever
from langchain.document_transformers import LongContextReorder
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
import langchain_core
from langchain.retrievers import MergerRetriever
import numpy as np
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
import torch
import re
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

from bs4 import BeautifulSoup
from tqdm import tqdm
import multiprocessing
from langchain_text_splitters import CharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel,FlagReranker

class Retriever2:
    def __init__(self, device1,device2,batch_size=64,  
                 hf_path="models/all-Mini-L6-v2", bge_large_path="models/bge-base-en-v1.5", m3_path="models/bge-m3",
                 parent_chunk_size=700,parent_chunk_overlap=150, 
                 child_chunk_size=200,child_chunk_overlap=50, 
                 separators=['.', "\n", " ", ""], token_path="models/Llama-3-8B-instruct",
                stopwords_list = 'models/processed_data/stopwords_list.npy',rerank_path ='models/bge-reranker-v2-m3'):

        self.hf_embeddings = HuggingFaceEmbeddings(model_name=hf_path,
                                                   model_kwargs={"device": device1},
                                                   encode_kwargs={'batch_size': batch_size,
                                                                  'normalize_embeddings': True})

        self.reranker = FlagReranker(rerank_path, use_fp16=True,device =device2)
        
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)

        self.parent_text_splitter = CharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separator=' '
        ) 
        self.child_text_splitter = CharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separator=' ',
        ) 
        self.stopwords_list = np.load(stopwords_list).tolist()
        self.oscar_map = np.load('models/processed_data/oscar_map.npy',allow_pickle=True).tolist()
        self.oscar_map_dlc = np.load('models/processed_data/oscar_map_dlc.npy',allow_pickle=True).tolist()
        self.imdb_movie_dataset = np.load('models/processed_data/all_imdb_movie.npy',allow_pickle=True).tolist()
        self.movie_index={}
        for idx,data in enumerate(self.imdb_movie_dataset):
            self.movie_index[data['title'].lower()] = idx
            self.movie_index[data['original_title'].lower()] = idx
        
        self.grammy_map = np.load('models/processed_data/grammy.npy',allow_pickle=True).tolist()
        self.ticker_name_map,self.ticker_info_map,self.ticker_name_set_map = np.load('models/processed_data/finance_data.npy',allow_pickle=True).tolist()


    def find_finance_name(self,name):
        name = name.lower().strip()
        if name in self.ticker_info_map.keys():
            return name
        name = name.replace('common stock','') 
        if name in self.ticker_name_map.keys():
            return self.ticker_name_map[name]
        name_set = set(name.split(' '))
        max_s = 0
        match = None
        for gd_name,gd_name_set in self.ticker_name_set_map.items():
            s = len(gd_name_set&name_set) / len(gd_name_set|name_set)
            if s >max_s:
                max_s = s
                match = self.ticker_name_map[gd_name]
        return match

    def get_finance_context(self,name):
        name = self.find_finance_name(name)
        if name is None:
            return ""
        return '<Doc>'+self.ticker_info_map[name]+'</Doc>\n'
        
    def get_movie_context(self,name):
        all_movie_keys = self.movie_index.keys()
        if name not in all_movie_keys:
            return ""
        else:
            res_key = name
        context = f'<Doc> \nInformation about {res_key}: '
        movie_info = self.imdb_movie_dataset[self.movie_index[res_key]]
        for key in movie_info.keys():
            if key not in ['cast','crew']:
                context+=f'the {key} is {movie_info[key]}. '.replace("\'",'')
        if 'cast' in movie_info.keys():
            context+='the cast are: '
            for actor in movie_info['cast']:
                name = actor['name']
                ch = actor['character']
                context+=f'{name} plays {ch};' 
                    
        if 'crew' in movie_info.keys():
            context+='the crew are: '
            for actor in movie_info['crew']:
                name = actor['name']
                ch = actor['job']
                context+=f'{ch} is {name};' 
        return context +'\n</DOC>\n'
    
    def clear(self):
        if hasattr(self, 'retriever'):
            del self.retriever
        Chroma().delete_collection()
        torch.torch.cuda.empty_cache()

    def get_result(self, query,k=5,rerank = True):
        torch.torch.cuda.empty_cache()
        docs = self.retriever.get_relevant_documents(query)
        print('len docs',len(docs))
        if docs ==[]:
            return [""]
        if len(docs)<=k:
            return [doc.page_content for doc in docs]
        with torch.no_grad():
            sentence_pairs = [[query,doc.page_content]  for doc in docs]
            sim =  self.reranker.compute_score(sentence_pairs, normalize=True,batch_size=16)
            indexs = torch.topk(torch.tensor(sim),min(k,len(docs))).indices
            del sim
            torch.torch.cuda.empty_cache()
            docs = [docs[idx].page_content for idx in indexs]
        return docs 
    
    def contains_year(self,sentence):
        pattern = r'\b(19|20)\d{2}\b'  
        match = re.search(pattern, sentence)
        if match:
            return True, match.group() 
        else:
            return False, None  
    
    def judge_grammy(self,query):
        if 'grammy' in query or 'best' in query: 
            has_year, year = self.contains_year(query)
            if has_year and  int(year) in self.grammy_map.keys():
                return year
        return None
    
    def get_music_grammy(self,query):
        year = self.judge_grammy(query)
        if year is not None:
            print('is grammy')
            return  '<Doc>'+self.grammy_map[int(year)]+ '</Doc>'
        return None
    
    def judge_oscar(self,query):
        if 'oscar' in query or 'academy' in query or 'best' in query: 
            has_year, year = self.contains_year(query)
            if has_year and  int(year) in self.oscar_map_dlc.keys():
                return year
        return None
    
    def get_movie_oscar(self,query):
        year = self.judge_oscar(query)
        if year is not None:
            print('is oscar')
            if int(year) in self.oscar_map.keys():
                description = self.oscar_map[int(year)]
            else:
                description = self.oscar_map_dlc[int(year)]
            sentence_pairs = [[query,doc]  for doc in description]
            torch.torch.cuda.empty_cache()
            sim = self.reranker.compute_score(sentence_pairs, normalize=True)
            indexs = torch.topk(torch.tensor(sim),min(10,len(description))).indices
            result = str('\n'.join([description[idx] for idx in indexs]))
             
            del sim 
            torch.torch.cuda.empty_cache()
            return result
        return None

    def init_retriever(self, search_results, recall_k=50, task3_topk = 5,max_length= 12000, task3 = False, separator=' ', method='ensemble', query=None, riddle=100,time_half_limit=1):
        st = time()
        self.method = method
        docs = []
        hashes = set()
        if task3 ==True:
            for idx, html in tqdm(enumerate(search_results)):
                html = html['page_snippet']   
                text = html.strip().lower()
                metadata ={}
                metadata["start_index"] =idx+task3_topk
                inputs = self.tokenizer.encode(text,max_length=max_length,add_special_tokens=False)
                if len(inputs)==max_length:
                    text = self.tokenizer.decode(inputs)
                docs.append(Document(page_content=text, metadata=metadata))
            torch.torch.cuda.empty_cache()
            with torch.no_grad():
                sentence_pairs = [[query,doc.page_content]  for doc in docs]
                sim =  self.reranker.compute_score(sentence_pairs, normalize=True,batch_size=25)
                indexs = torch.topk(torch.tensor(sim),min(task3_topk,len(docs))).indices
                del sim
                torch.torch.cuda.empty_cache()

            search_results = [search_results[idx] for idx in indexs]
            docs = [docs[i] for  i in range(len(docs)) if i not in indexs]
            print('len',len(docs),len(search_results))
         
        for idx, html in tqdm(enumerate(search_results[:task3_topk])):
            html_content = html['page_result']
            hash_value = hash(html_content)
            if hash_value in hashes or len(html_content) == 0:
                continue
            hashes.add(hash_value)
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=separator, strip=True).lower()
            text=html['page_snippet'].lower()+'\n\n'+text
            inputs = self.tokenizer.encode(text,max_length=max_length,add_special_tokens=False)
            print('len',len(inputs))
            if len(inputs)==max_length:
                text = self.tokenizer.decode(inputs)
                print('exceed html max size')
            metadata ={}
            metadata["start_index"] =idx
            docs.append(Document(page_content=text, metadata=metadata))
         
 
                    
        
        
        print('get_text',time()-st)

        if len(docs) == 0:
            return False
        hf_vectorstore = Chroma(
            collection_name="hf_split_parents", embedding_function=self.hf_embeddings
        )
        hf_retriever = ParentDocumentRetriever(
            vectorstore=hf_vectorstore,
            docstore=InMemoryStore(),
            child_splitter=self.child_text_splitter,
            parent_splitter=self.parent_text_splitter,
            search_kwargs = {'k':recall_k})
        hf_retriever.add_documents(docs, ids=None) 
        print('hf_retriever',time()-st)
        self.retriever = hf_retriever
        print('EnsembleRetriever',time()-st)
        return True
#r = Retriever2('cuda:1','cuda:0',batch_size=256)