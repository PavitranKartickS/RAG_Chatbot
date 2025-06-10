import streamlit as st
import warnings
import logging

import os
from dotenv import load_dotenv
import httpx
from langchain_groq import ChatGroq
from langchain_core.output_parsers  import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
    chain = groq_sys_prompt | groq_chat | StrOutputParser()

    response=chain.invoke({"user_prompt": prompt})


    st.chat_message('assistant').markdown(response)
    # store the response
    st.session_state.messages.append({'role': 'assistant', 'content': response})
