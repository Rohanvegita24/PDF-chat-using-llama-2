from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import FakeEmbeddings
import os
from langchain.chains import RetrievalQA
import gradio as gr
import torch



embeddings = FakeEmbeddings(size=1352)
llm = HuggingFacePipeline.from_model_id(model_id="TheBloke/Llama-2-7B-AWQ",task="text-generation",device=0,pipeline_kwargs={"max_new_tokens": 150})
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def RAG(file,query):
    loader=UnstructuredFileLoader(file)
    
    documents = loader.load()

    docs= text_splitter.split_documents(documents)
    db=Chroma.from_documents(docs,embeddings)
    
    
    chain = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=db.as_retriever(), 
                                    input_key="question")
    final=chain.invoke(query)
   
   
     
    return final['result']

iface = gr.Interface(
    fn=RAG, 
    inputs=["file","text"],
    outputs="text",
    title="llama-2",
).launch()
