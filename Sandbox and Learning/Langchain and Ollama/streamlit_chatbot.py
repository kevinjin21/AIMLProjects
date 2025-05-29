# Building a chatbot using langchain and streamlit.io
# streamlit turns data scripts into web apps without needing to know front-end. More advanced version may use Flask python backend + React frontend.
# documentation: https://docs.streamlit.io/

import streamlit as st
from typing import Annotated

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from langgraph.types import Command
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any
import uuid

import os
from dotenv import load_dotenv
load_dotenv('.env')

base_url = 'http://localhost:11434'
model = 'llama3.2'
# streamlit title, description, and user_id
st.title("My First Chatbot")
st.write("Description for your chatbot")

### LONG TERM MEMORY IMPLEMENTATION
# user_id = st.text_input('Enter your user id', 'kjin')
# def get_session_history(session_id):  # Get history from sqlite db
#    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")
# with st.chat_message("assistant"):
#        response = st.write_stream(chat_with_llm(user_id, prompt))
# system = SystemMessagePromptTemplate.from_template(
#    'You are a helpful assistant.')
# human = HumanMessagePromptTemplate.from_template(
#    '{input}')  # new input after looking at history
# messages = [system, MessagesPlaceholder(variable_name='history'), human]
# prompt = ChatPromptTemplate(messages=messages)
# chain = prompt | llm | StrOutputParser(
# runnable_with_history = RunnableWithMessageHistory(chain, get_session_history,
#                                                   input_messages_key='input',
#                                                   history_messages_key='history')
# def chat_with_llm(session_id, input):  # chatbot function (from previous lesson)
#    # use stream to not output as a single chunk
#    for output in runnable_with_history.stream({'input': input}, config={'configurable': {'session_id': session_id}}):
#        yield output

### SHORT TERM MEMORY IMPLEMENTATION
def chat_with_llm(user_id, input_text):
    '''Format output to display in UI'''
    config = {'configurable': {'thread_id': user_id}}
    
    try:
        response = st.session_state.agent.stream(
            input_text,
            config=config,
            stream_mode="updates"
        )
        
        final_response = None
        for chunk in response:
            for node, update in chunk.items():
                print(node)
                print(update)
                print('')
                if node == "pre_model_hook":
                    messages = update.get("llm_input_messages", [])
                    if messages:
                        st.session_state.summary = messages[-1].content
                    continue
                elif node == "tools":
                    messages = update.get("messages", [])
                    if messages:
                        return messages[-1].content
                return update["messages"][-1].content
            

        return "I couldn't process that request."
                #return update["messages"][-1].content
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        return "I apologize, but I encountered an error processing your request."
    
    return None

# ================================Setup streamlit formatting for frontend app (very simple version)================================
# To quickly view this, use the command: streamlit run streamlit_chatbot.py
# This leads to local URL where you can view your chatbot application in the browser.

# Initialize session states for chat history and messages
if "agent" not in st.session_state:
    # Initialize Agent and tools
    llm = ChatOllama(base_url=base_url, model=model)

    summarization_node = SummarizationNode( 
        token_counter=count_tokens_approximately,
        model=llm,
        max_tokens=2048,
        max_summary_tokens=512,
        output_messages_key="llm_input_messages",
    )

    # https://langchain-ai.github.io/langgraph/agents/context/#tools
    # todo: add real tooling in future iterations
    # todo: tool should update context
    @tool
    def get_weather(city: str,
                    tool_call_id: Annotated[str, InjectedToolCallId],
                    config: RunnableConfig) -> Command:
        """Get weather for a given city."""
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"It's always sunny in {city}!",
                    tool_call_id=tool_call_id
                )
            ]
        })

    tools = [get_weather]
    checkpointer = InMemorySaver()

    class State(AgentState):
        """Agent state that includes context tracking."""
        context: dict[str, Any] = {}

    prompt = """You are a helpful AI assistant with access to tools. 
    Answer the user's questions to the best of your ability.

    When someone asks about the weather, use the get_weather tool with the city name.

    For all other questions, respond directly without using tools.
    Always be polite and professional.
    """

    st.session_state.agent = create_react_agent(
        model=llm,
        tools=tools,
        pre_model_hook=summarization_node,
        state_schema=State,
        checkpointer=InMemorySaver(),
        prompt=prompt
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Session state ID auto-generated with new conversation
if "session_id" not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

# for debugging
st.write(f"Session ID: {st.session_state['session_id']}")

# Reset history, messages, id for new conversation
if st.button("Start New Conversation"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4()) 

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Default value shown in chat box of application
prompt = st.chat_input("Hello!")

if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    inputs = {"messages": [("user", prompt)]}

    with st.chat_message("user"):
        st.markdown(prompt)  # format user input to show as user message

    # output response from llm (will also be formatted as llm output and will be streamed)
    with st.chat_message("assistant"):
        response = chat_with_llm(st.session_state.session_id, inputs)
        if response:
            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        # note: use st.write_stream for streaming output
        st.session_state.chat_history.append(
        {'role': 'assistant', 'content': response})