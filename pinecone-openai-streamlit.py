#!/usr/bin/env python
# coding: utf-8

# In[41]:


import os
import openai
import pinecone
import time
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()


# In[42]:

openai_key = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"]=openai_key

pinecone_key = os.getenv("PINECONE_KEY")
pinecone.init(api_key=os.getenv(pinecone_key), environment='gcp-starter')

pinecone_index_name = 'document-search-index'

open_embedding='DENY' # set to ALLOW if you want to call openai embedding


# In[43]:


# initialize pinecone
pinecone.init(
    api_key=pinecone_key,  # find at app.pinecone.io
    environment="gcp-starter",  # next to api key in console
    
)


# In[44]:


index_name='document-search-index'
index = pinecone.Index(index_name)
index.describe_index_stats()


# In[45]:


dir_file='./data/economic.txt'
file=open(dir_file,'r')
content=file.read()
file.close()
#print(content)


# In[46]:


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2048,
    chunk_overlap  = 20
)
texts= text_splitter.split_text(content)


# In[47]:


def get_embedding_vector_from_openai(txt):
    return openai.embeddings.create( input=txt, model='text-embedding-ada-002').data[0].embedding


# In[48]:


# Embedding the whole text document

if open_embedding=='ALLOW':
    for t in texts:
        response=get_embedding_vector_from_openai(t)
    
        data = pinecone.Vector(
            id=str(uuid.uuid4()),
            values=response,
            metadata={'title': "economic"}
                )
    #Only enable when inserting into Pinecone vector DB
        index.upsert([data])


# In[49]:


def query_pinecone_index(query):
    index = pinecone.Index(pinecone_index_name)
    query_embedding_vector = get_embedding_vector_from_openai(query)
    response = index.query(
        vector=query_embedding_vector,
        top_k=1,
        include_metadata=True
    )
    return response['matches'][0]['metadata']['title']


# In[50]:


def load_document_content(title):
    documents_path = 'data'
    file_path = os.path.join(documents_path, title + '.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# In[51]:


def create_prompt(question,document_content):
    return 'You are given a document and a question. Your task is to answer the question based on the document.\n\n'            'Document:\n\n'            f'{document_content}\n\n'            f'Question: {question}'
prompt='You are given a document and a question. Your task is to answer the question based on the document.\n'


# In[52]:


def get_answer_from_openai(question):
    relevant_document_title = query_pinecone_index(question)
    print(f'Relevant document title: {relevant_document_title}')
    document_content = load_document_content(relevant_document_title)
    prompt = create_prompt(question, document_content)
    #print(f'Prompt:\n\n{prompt}\n\n')
    completion=openai.chat.completions.create(
        model='gpt-4-turbo-preview',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    return completion.choices[0].message.content


# In[53]:


def chatbot_response(prompt, user_input):
    prompt = prompt  + user_input 
    chat_response = get_answer_from_openai(prompt)
    prompt = prompt + chat_response
    return chat_response, prompt


# In[54]:


def get_text():
    input_text = st.text_input("Enter your question to the Bot about your document: ","", key="text")
    return input_text


# In[55]:


def clear_text():
    st.session_state["text"] = ""
    
user_input = get_text()
st.button("Clear Text", on_click=clear_text)


# In[56]:


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'bot_prompt' not in st.session_state:
    st.session_state['bot_prompt'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []

if len(st.session_state.bot_prompt) == 0:
    pr: list = prompt.split('\n')
    pr = [p for p in pr if len(p)]  # remove empty string
    st.session_state.bot_prompt = pr


# In[40]:


if user_input:
    st.session_state.bot_prompt.append(f'{user_input}')
    input_prompt: str = '\n'.join(st.session_state.bot_prompt)
    
    output, prompt = chatbot_response(prompt, user_input)
    #print(prompt)
    st.write(prompt)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    
    st.session_state.bot_prompt.append(output)
    
    #if st.session_state['generated']:
    #    for i in range(len(st.session_state['generated'])-1, -1, -1):
    #        message(st.session_state["generated"][i], key=str(i))
    #        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    


# In[ ]:


#question = input('Enter a question: ')
#question="How does the moon affect Coca Cola"
#answer = get_answer_from_openai(question)
#print(answer)

