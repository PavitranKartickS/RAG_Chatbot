import streamlit as st
import warnings
import logging
import tempfile
import shutil

import httpx

import os
from dotenv import load_dotenv
import httpx
from langchain_groq import ChatGroq
from langchain_core.output_parsers  import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

#load api from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


#disable warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("Chatterji Support")
# Setup session to hold all previous conversations
if 'messages' not in st.session_state:
    st.session_state.messages = []

#display all old messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

#creating a vectorstore

@st.cache_resource
def get_vectorstore():
    pdf_file="./OPLore.pdf"
    loaders=[PyPDFLoader(pdf_file)]
    #create chunks of data, embeddings
    index=VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='C:/temp/Project Works/RAG PDF chatbot/all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),

    ).from_loaders(loaders)
    return index.vectorstore

prompt = st.chat_input("Enter a messge")

if prompt:
    st.chat_message('user').markdown(prompt)
    # Store user prompts 
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template("""You are helpful and efficient assistant  as named "Chatterji", you provide the most precise and accurate solutions based on
                                                       User prompts. Be polite.
    """)

    #call the model
    model = "llama3-8b-8192"

    groq_chat=ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        http_client=httpx.Client(verify=False),
        model=model,
        
    )

    try:
        vectorstore=get_vectorstore()
        if vectorstore is None:
            st.error("Document loading failed")

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
            return_source_documents=True,
        )

        result=chain({'query':prompt})
        response = result["result"]

        #response=chain.invoke({"user_prompt": prompt})


        st.chat_message('assistant').markdown(response)
        # store the response
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error occurred: {e}")
    
