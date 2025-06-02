from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
base_url = 'http://localhost:11434'
model = 'llama3.2'
llm = ChatOllama(base_url=base_url, model=model)

system = SystemMessagePromptTemplate.from_template("""
    You are a financial transaction categorizer. Your task is to analyze transaction descriptions 
    and assign them to the most appropriate category from the provided list.
    You should be consistent in your categorization and focus on the primary purpose of the transaction.
""")

categorize_prompt = """
    Categorize this transaction description into EXACTLY ONE of these categories:
    {categories}

    Rules:
    - Choose exactly one category
    - Respond ONLY with the category name, nothing else
    - If unsure, use 'Other'

    Transaction Description: {description}
"""

prompt = HumanMessagePromptTemplate.from_template(categorize_prompt)
messages = [system, prompt]
template = ChatPromptTemplate(messages=messages)

categorize_chain = template | llm | StrOutputParser()

def categorize_transaction(description: str, categories: list) -> str:
    """
    Categorize a single transaction description
    """
    categories_str = ", ".join(categories)
    return categorize_chain.invoke({
        "description": description,
        "categories": categories_str
    })