import streamlit as st
import warnings
import logging

#disable warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("RAG Chat UI")
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

    response = "Hi!, I am Chatterji"

    st.chat_message('assistant').markdown(response)
    # store the response
    st.session_state.messages.append({'role': 'assistant', 'content': response})
