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

#creating a vectorstore
@st.cache_resource
def get_vectorstore(_file_paths):
    if not _file_paths:
        return None
    try:
        file_paths = [st.session_state.uploaded_files[key] for key in file_keys]
        loaders=[PyPDFLoader(file_path) for file_path in _file_paths]
        #create chunks of data, embeddings
        index=VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='C:/temp/Project Works/RAG PDF chatbot/all-MiniLM-L12-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        ).from_loaders(loaders)
        return index.vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    # Setup session to hold all previous conversations
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    st.markdown("""
        <style>
        .title{
            color:#85e5a7 !important;
            font-size: 40px !important;
            font-weight: bold !important;
            margin-bottom: 20px !important;
        }
        .subheader{
            color:#007069 !important;
        }
        .chat-input-container {
            position: fixed !important;
            bottom: 0 !important;
            padding: 10px !important;
            z-index: 1000 !important;
            width: 100% !important;
        }
        .chat-messages {
            margin-bottom: 80px !important;
        } 
        </style>
    """,unsafe_allow_html=True)

    #Verifying Session state initialization
    #st.write("session state initialized", st.session_state)

    #Columns for Design:
    left_col, chat_col, right_col = st.columns([1.5,4,1])

    #File Management Column:
    with left_col:
        #Uploading Files for PDF
        st.markdown("<h3 class='subheader'>Upload your PDF Files</h3>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(" Drop your PDF Files", type="pdf", accept_multiple_files=True)

        #Processing the uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                #Save to temp directory
                with tempfile.NamedTemporaryFile(delete=False, suffix="pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                    #storing file name and path in session state:
                    st.session_state.uploaded_files[uploaded_file.name] = tmp_file_path
            st.success("File(s) uploaded successfully")
            #clear vectorstore to force rebuild with new files
            st.session_state.vectorstore = None

        #Displaying Uploaded files and allowing deletion:
        st.markdown("<h3 class='subheader'>Manage Uploaded Files</h3>", unsafe_allow_html=True)
        if st.session_state.uploaded_files:
            st.write("Files Uploaded:")
            for file_name in list(st.session_state.uploaded_files.keys()):
                col1, col2 = st.columns([3,1])
                col1.write(file_name)
                if col2.button("Delete", key=f"delete_{file_name}"):
                    #removing file from session state and drive
                    try:
                        file_path = st.session_state.uploaded_files.pop(file_name)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        st.success(f"Deleted {file_name}")
                        # Clear vectorstore to force rebuild
                        st.session_state.vectorstore = None
                    except KeyError:
                        st.error(f"File {file_name} not found in session state")



    #Right Column for Chat Interface:
    with chat_col:
        st.markdown("<h1 class = 'title'>RAG AI CHATBOT</h1>", unsafe_allow_html=True)  

        
        with st.container():
            st.markdown("<div class='chat-messages'>", unsafe_allow_html=True)
            #display all old messages
            for message in st.session_state.messages:
                st.chat_message(message['role']).markdown(message['content'])
            st.markdown("</div>", unsafe_allow_html=True)
            
        with st.container():
            st.markdown("<div class = chat-input-container>", unsafe_allow_html=True)
            #Chat Input and Processing
            prompt = st.chat_input("Enter a messge")
            st.markdown("</div>", unsafe_allow_html=True)


        if prompt:
            st.chat_message('user').markdown(prompt)
            # Store user prompts 
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            groq_sys_prompt = ChatPromptTemplate.from_template("""You are helpful and efficient assistant  as named "Chatterji", you provide the most precise and accurate solutions based on
                                                            User prompts. Be polite.""")

            #call the model
            model = "llama3-8b-8192"
            groq_chat=ChatGroq(
                groq_api_key=os.environ.get("GROQ_API_KEY"),
                http_client=httpx.Client(verify=False),
                model=model,
                
            )

            try:
                #Build/retrieve vectorstore from uploaded files
                if st.session_state.uploaded_files and st.session_state.vectorstore is None:
                    file_keys = tuple(sorted(st.session_state.uploaded_files.keys()))
                    st.session_state.vectorstore = get_vectorstore(file_keys)
                
                if st.session_state.vectorstore is None:
                    st.error("No Documents loaded, please upload PDF files")
                else:
                    chain = RetrievalQA.from_chain_type(
                        llm=groq_chat,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k':3}),
                        return_source_documents=True,
                    )

                    result=chain({'query':prompt})
                    response = result["result"]
                    st.chat_message('assistant').markdown(response)
                    # store the response
                    st.session_state.messages.append({'role': 'assistant', 'content': response})
                

            except Exception as e:
                st.error(f"Error occurred: {e}")
    
