# Building a chatbot using langchain and streamlit.io
# streamlit turns data scripts into web apps without needing to know front-end. More advanced version may use Flask python backend + React frontend.
# documentation: https://docs.streamlit.io/

import streamlit as st

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv('.env')

base_url = 'http://localhost:11434'
model = 'llama3.2'
# streamlit title, description, and user_id
st.title("My First Chatbot")
st.write("Description for your chatbot")
user_id = st.text_input('Enter your user id', 'kjin')


def get_session_history(session_id):  # Get history from sqlite db
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")


# LLM Setup with history
llm = ChatOllama(base_url=base_url, model=model)

system = SystemMessagePromptTemplate.from_template(
    'You are a helpful assistant.')
human = HumanMessagePromptTemplate.from_template(
    '{input}')  # new input after looking at history

messages = [system, MessagesPlaceholder(variable_name='history'), human]

prompt = ChatPromptTemplate(messages=messages)

chain = prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(chain, get_session_history,
                                                   input_messages_key='input',
                                                   history_messages_key='history')


def chat_with_llm(session_id, input):  # chatbot function (from previous lesson)
    # use stream to not output as a single chunk
    for output in runnable_with_history.stream({'input': input}, config={'configurable': {'session_id': session_id}}):
        yield output


# ================================Setup streamlit formatting for frontend app (very simple version)================================
# To quickly view this, use the command: streamlit run streamlit_chatbot.py
# This leads to local URL where you can view your chatbot application in the browser.

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# button to clear chat history and start new convo (NOTE: chat history is saved in db, so it can be referenced even after convo is cleared)
if st.button("Start New Conversation"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# setup chat session in streamlit
# default value shown in chat box of application
prompt = st.chat_input("Hello!")

if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})

    with st.chat_message("user"):
        st.markdown(prompt)  # format user input to show as user message

    # output response from llm (will also be formatted as llm output and will be streamed)
    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm(user_id, prompt))

    # update history with user message
    st.session_state.chat_history.append(
        {'role': 'assistant', 'content': response})
