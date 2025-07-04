from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

base_url = 'http://localhost:11434'
model = 'llama3.2'
llm = ChatOllama(base_url=base_url, model=model)

system = SystemMessagePromptTemplate.from_template("You are a helpful AI assistant who will answer user questions based on the provided context.")

prompt = """Answer user question based on the provided context ONLY. If you do not know the answer, just say "I don't know".
            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:
"""

prompt = HumanMessagePromptTemplate.from_template(prompt)

messages = [system, prompt]
template = ChatPromptTemplate(messages=messages)

qna_chain = template | llm | StrOutputParser()

def ask_llm(context, question):
    return qna_chain.invoke({"context": context, "question": question})