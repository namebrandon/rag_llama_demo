#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('python -m pip install --upgrade pip')
get_ipython().system('pip install transformers')
get_ipython().system('pip install llama_index')
get_ipython().system('pip install accelerate')
get_ipython().system('pip install sentence_transformers')
get_ipython().system('pip install langchain')


# In[1]:


import logging
import sys, os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM

os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache/'
os.environ['HF_DATASETS_CACHE']='/workspace/cache/'


# In[2]:


from llama_index.prompts.prompts import SimpleInputPrompt

system_prompt = """<|SYSTEM|># A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


# In[3]:


import torch

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.1, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="ehartford/Wizard-Vicuna-13B-Uncensored",
    model_name="ehartford/Wizard-Vicuna-13B-Uncensored",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.float16}
)
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm)


# In[4]:


resp = llm.complete("Summarize the short story 'Gray Denim' by Harl Vincent.")
print(resp)


# In[5]:


from pathlib import Path
from llama_index import download_loader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext


embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",
        model_kwargs={"device": "cuda"},
    )
)

service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model=embed_model)


# In[7]:


EpubReader = download_loader("EpubReader")

loader = EpubReader()
documents = loader.load_data(file=Path("Super-Science-December-1930.epub"))


# In[8]:


index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist(persist_dir="./sfbook")


# In[9]:


query_engine = index.as_query_engine()
response = query_engine.query("Summarize the short story 'Gray Denim' by Harl Vincent.")


# In[10]:


print(response)


# In[11]:


response = query_engine.query("Who are the Red Police in the short story 'Gray Denim' by Harl Vincent.")
print(response)


# In[ ]:




