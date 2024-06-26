#!/usr/bin/env python

## Original source from this youtube
# [**Link to my YouTube Channel**](https://www.youtube.com/BhaveshBhatt8791?sub_confirmation=1)
# 


##!pip install -q cassandra-driver
##!pip install -q cassio>=0.1.1
##!pip install -q gradientai --upgrade
##!pip install -q llama-index>0.5.0
##!pip install -q tiktoken==0.4.0
##!pip install llama-index-vector-stores-cassandra
##!pip install llama-index-llms-gradient
##!pip install llama-index-embeddings-gradient
##!pip install gradio==3.48.0
##!pip install -U scikit-learn 
##!pip install astrapy json transformers torch uuid


import os
import json
import pathlib
import shutil
from dotenv import load_dotenv

load_dotenv()
os.environ['GRADIENT_ACCESS_TOKEN'] = os.getenv("GRADIENT_ACCESS_TOKEN")
os.environ['GRADIENT_WORKSPACE_ID'] = os.getenv("GRADIENT_WORKSPACE_ID")


# # Import Cassandra & llama Index

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import ServiceContext
from llama_index.core import set_global_service_context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext

import gradio as gr

import numpy as np
from astrapy.db import AstraDB
import uuid
import textwrap
from difflib import Differ
import fitz


from llama_index.embeddings.gradient.base import  GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM


# In[167]:


#gradient = Gradient()
#embed_model = gradient.get_embeddings_model(slug="bge-large")


# In[168]:


import cassandra
print (cassandra.__version__)


# # Connect to the VectorDB

# In[169]:


# This secure connect bundle is autogenerated when you donwload your SCB,
# if yours is different update the file name below
cloud_config= {
  'secure_connect_bundle': 'secure-connect-q-and-a.zip'
}

# This token json file is autogenerated when you donwload your token,
# if yours is different update the file name below
with open("q-and-a-token.json") as f:
    secrets = json.load(f)

CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]

auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

row = session.execute("select release_version from system.local").one()
if row:
  print(row[0])
else:
  print("An error occurred.")


# # Define the Gradient's Model Adapter for LLAMA-2

# In[170]:


llm = GradientBaseModelLLM(
    base_model_slug="llama2-7b-chat",
    max_tokens=400,
)


# # Configure Gradient embeddings

# In[171]:


embed_model = GradientEmbedding(
    gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"],
    gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"],
    gradient_model_slug="bge-large",
)


# In[172]:


service_context = ServiceContext.from_defaults(
    llm = llm,
    embed_model = embed_model,
    chunk_size=512,
)

set_global_service_context(service_context)


# # Load the PDFs

# In[173]:


#documents = SimpleDirectoryReader("./data/").load_data()
#print(f"Loaded {len(documents)} document(s).")


# # Setup and Query Index

# In[174]:


# check if storage already exists
##PERSIST_DIR = "./storage"
##if not os.path.exists(PERSIST_DIR):
##    # load the documents and create the index
##    documents = SimpleDirectoryReader("./data/").load_data()
##    #index = VectorStoreIndex.from_documents(documents)
##    index = VectorStoreIndex.from_documents(documents,
##                                        service_context=service_context)
##    # store it for later
##    index.storage_context.persist(persist_dir=PERSIST_DIR,vector_store_fname='vector_store.json')
##else:
##    # load the existing index
##    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, vector_store='vector_store.json')
##    index = load_index_from_storage(storage_context)


##index = VectorStoreIndex.from_documents(documents,
##                                        service_context=service_context)
##query_engine = index.as_query_engine()



## 
class indexed:  
    def __init__(self) -> None:
           file_path1=""    
    def load_index(self,documents):
            index = VectorStoreIndex.from_documents(documents,
                                        service_context=service_context)
            self.query_engine = index.as_query_engine()
            self.retriever_engine = index.as_retriever()
            
    def query_index(self,prompt):
            response= self.query_engine.query(prompt)
            return response
    
    def retrieve_text(self,prompt):
            result= self.retriever_engine.retrieve(prompt)
            return result
    
    def load_file(self,file1, history: list[str] ):
        directory=os.path.join(pathlib.Path().resolve(),'./userFile1')
         # Check if the directory exists, if not create it
        if not os.path.exists(directory):
                os.makedirs(directory)

        # Save the file to the specified directory
        self.file_path1 = os.path.join(directory, os.path.basename(file1.name))
        #with open(file_path, "wb") as f:
        #    f.write(file.read())
        shutil.copyfile(file1.name,self.file_path1) 
       
        documents = SimpleDirectoryReader(directory).load_data() 
        self.load_index(documents)
        return f'File uploaded {os.path.basename(file1.name)}',history

    def get_answer_from_llm(self,prompt,history: list[str]):  
        response = self.query_index(prompt)
        return response, history

    def search_file(self,file2):
        print("FILE PATH -->",self.file_path1)
        directory2=os.path.join(pathlib.Path().resolve(),'./userFile2')
        #directory1=os.path.join(pathlib.Path().resolve(),'./userFile1')
        # Check if the directory exists, if not create it
        if not os.path.exists(directory2):
                os.makedirs(directory2)

         # Save the file to the specified directory
        file_path2 = os.path.join(directory2, os.path.basename(file2.name))
        shutil.copyfile(file2.name,file_path2)    
        all_text_file2=[]
        with fitz.open(file_path2) as doc:  # open document
                all_text_file2 = chr(12).join([page.get_text() for page in doc])
    
        text_file = open("file2.txt", "w")
        text_file.write(all_text_file2)
        text_file.close()
        
        all_text_file1=[]
        with fitz.open(self.file_path1) as doc:  # open document
                all_text_file1 = chr(12).join([page.get_text() for page in doc])
        
        text_file = open("file1.txt", "w")
        text_file.write(all_text_file1)
        text_file.close()

        with open('file1.txt') as file_1, open('file2.txt') as file_2: 
                differ = Differ() 
                lines_file2=[]    
                lines_file1=[] 
                for line in differ.compare(file_1.readlines(), file_2.readlines()): 
                        if(line[0]== '+' ):  #print(line)
                                lines_file2.append(line[1:])
                        if(line[0]== '-' ):  #print(line)
                                lines_file1.append(line[1:])
        return lines_file2,lines_file1


# In[189]:


ind1=indexed()


# In[185]:





# In[190]:


iface1 = gr.Interface(
    fn=ind1.load_file,
    inputs=[gr.inputs.File(label="Upload a file")  , gr.State(value=[]) ],
    outputs=["text",  gr.State()]
)
#########################
iface2 = gr.Interface(
    fn=ind1.get_answer_from_llm,
    inputs=["text", gr.State()],
    outputs=["text", gr.State()],
    title="Question Answering"
)
iface3 = gr.Interface(
    #fn=similarity_search 
    fn=ind1.search_file,
    inputs=[gr.inputs.File(label="Upload a file")  ],
    #outputs=["text","text","text"]
    outputs=["text","text"]
)

demo = gr.TabbedInterface([iface1, iface2, iface3], ["Load File", "Question Answering","Comparison"])
# Run the interface
demo.launch(share=True)


# In[ ]:


res,text,similarity=similarity_search("/Users/anujk/Downloads/Cover_letter_BCG.pdf")
print(res, text,similarity)
#result=compare_docs(" AWS") 
#print( result[1] )


