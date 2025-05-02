from dotenv import load_dotenv
import re
import os
import glob
from typing import List, Union, Optional, Tuple
from abc import ABC, abstractmethod

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    YoutubeLoader
)
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
base_url = 'http://localhost:11434'

class BaseDocumentLoader(ABC):
    """Abstract base class for all document loaders"""
    
    @abstractmethod
    def load(self, source: Union[str, List[str]]) -> List:
        """Load documents from source"""
        pass

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean the text content"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r'\t+', '\t', text)
        return text

    @staticmethod
    def format_docs(docs: List) -> str:
        """Format documents into a single string"""
        return '\n\n'.join([doc.page_content for doc in docs])
    
    @staticmethod
    def save_text(content: str, filepath: str) -> None:
        """Save content to a text file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def save_markdown(content: str, filepath: str) -> None:
        """Save content to a markdown file with proper extension"""
        if not filepath.endswith('.md'):
            filepath = f"{filepath}.md"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def save_output(content: str, filepath: str, format: str = 'txt') -> None:
        """Save content in specified format
        
        Args:
            content: The text content to save
            filepath: Path where to save the file
            format: File format ('txt' or 'md')
        """
        if format.lower() == 'md':
            BaseDocumentLoader.save_markdown(content, filepath)
        else:
            BaseDocumentLoader.save_text(content, filepath)
       
class WebPageLoader(BaseDocumentLoader):
    """Loader for web pages"""
    
    def __init__(self):
        os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    
    async def load(self, urls: List[str]) -> List:
        loader = WebBaseLoader(web_paths=urls)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        return docs

class PDFLoader(BaseDocumentLoader):
    """Loader for PDF documents"""
    
    def load(self, file_paths: Union[str, List[str]]) -> List:
        docs = []
        paths = [file_paths] if isinstance(file_paths, str) else file_paths
        
        for path in paths:
            if path.endswith('.pdf'):
                loader = PyMuPDFLoader(path)
                docs.extend(loader.load())
        return docs

class WordLoader(BaseDocumentLoader):
    """Loader for Word documents"""
    
    def load(self, file_paths: Union[str, List[str]]) -> List:
        docs = []
        paths = [file_paths] if isinstance(file_paths, str) else file_paths
        
        for path in paths:
            if path.endswith(('.doc', '.docx')):
                loader = UnstructuredWordDocumentLoader(path)
                docs.extend(loader.load())
        return docs

class ExcelLoader(BaseDocumentLoader):
    """Loader for Excel documents"""
    
    def load(self, file_paths: Union[str, List[str]]) -> List:
        docs = []
        paths = [file_paths] if isinstance(file_paths, str) else file_paths
        
        for path in paths:
            if path.endswith(('.xls', '.xlsx')):
                loader = UnstructuredExcelLoader(path, mode="elements")
                docs.extend(loader.load())
        return docs
    
    # Overload format_docs to preserve Excel structure using HTML metadata.
    @staticmethod
    def format_docs(docs: List) -> str:
        """Format documents to preserve Excel structure using HTML metadata"""
        formatted_text = ""
        for doc in docs:
            # Get sheet name or default to "Sheet"
            sheet_name = doc.metadata.get('sheet_name', 'Sheet')
            # Get HTML representation or fall back to plain text
            content = doc.metadata.get('text_as_html', doc.page_content)
            
            formatted_text += f"### {sheet_name}:\n\n{content}\n\n---\n\n"
        
        return formatted_text.strip()

class PowerPointLoader(BaseDocumentLoader):
    """Loader for PowerPoint documents"""
    
    def load(self, file_paths: Union[str, List[str]]) -> List:
        docs = []
        paths = [file_paths] if isinstance(file_paths, str) else file_paths
        
        for path in paths:
            if path.endswith(('.ppt', '.pptx')):
                loader = UnstructuredPowerPointLoader(
                    path,
                    mode="elements",  # Returns more structured elements
                    include_metadata=True  # Includes additional metadata
                )
                elements = loader.load()

                # Group elements by slide number
                slides = {}
                for element in elements:
                    slide_number = element.metadata.get('page_number', 1)
                    if slide_number not in slides:
                        slides[slide_number] = []
                    slides[slide_number].append(element.page_content)
                
                # Combine elements for each slide
                for slide_num, contents in slides.items():
                    combined_content = '\n'.join(contents)
                    doc = elements[0].__class__(
                        page_content=combined_content,
                        metadata={
                            "slide_number": slide_num,
                            "total_slides": len(slides),
                            "source_type": "powerpoint"
                        }
                    )
                    docs.append(doc)   
        return docs

    # Overload format_docs to aggregate content by slide number, not element.
    @staticmethod
    def format_docs(docs: List) -> str:
        """Format documents into a single string with slide numbers"""
        formatted_text = ""
        for doc in docs:
            slide_num = doc.metadata.get('slide_number', '?')
            formatted_text += f"### Slide {slide_num}:\n\n{doc.page_content.strip()}\n\n\n"
        return formatted_text.strip()

class YouTubeLoader(BaseDocumentLoader):
    """Loader for YouTube video transcripts"""
    
    def load(self, urls: Union[str, List[str]], chunk_size_seconds: int = 600) -> List:
        """Load transcripts from YouTube videos"""
        docs = []
        urls = [urls] if isinstance(urls, str) else urls
        
        for url in urls:
            try:
                # Use Langchain's YouTubeLoader with correct parameters
                loader = YoutubeLoader.from_youtube_url(
                    url,
                    #add_video_info=True,  # error with pytube
                    language=["en"],  # Specify language for transcript
                    transcript_format=TranscriptFormat.CHUNKS,
                    chunk_size_seconds=chunk_size_seconds
                )
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading transcript for {url}: {str(e)}")
                continue
        
        return docs

    @staticmethod
    def format_docs(docs: List) -> str:
        """Format documents with video information"""
        formatted_text = ""
        for doc in docs:
            title = doc.metadata.get('title', 'Untitled')
            timestamp = doc.metadata.get('start_timestamp', '')
            formatted_text += f"### {title}--Timestamp: {timestamp}\n\n{doc.page_content}\n\n---\n\n"
        return formatted_text.strip()

class DocumentManager:
    """Facade pattern that simplifies document loading by providing a single interface to handle multiple document types (PDF, Word, Excel, etc.) and their processing. 
    Acts as a high-level interface that coordinates between different document loaders and handles common operations like directory scanning and text chunking."""
    
    def __init__(self):
        self.web_loader = WebPageLoader()
        self.pdf_loader = PDFLoader()
        self.word_loader = WordLoader()
        self.excel_loader = ExcelLoader()
        self.powerpoint_loader = PowerPointLoader()
        self.llm = ChatOllama(base_url=base_url, model='llama3.2') # default llm
        self.docs = [] # initialize as empty list
        self.context = ''
        self.vector_store_path = None
        self.embedding = OllamaEmbeddings(base_url=base_url, model='nomic-embed-text')
        self.rag_prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        Do not answer based on assumptions, use the context only.
        Keep your answer concise and neatly structured. Use bullet points if necessary.
        Make sure your answer is relevant to the question and it is answered from the context only.
        ### Question: 
        {question} 

        ### Context: 
        {context} 

        ### Answer:
        """

    def load_directory(self, directory_path: Union[str, List[str]]) -> Tuple[List, str]:
        """Load all supported documents from directory path or list of file paths
    
        Args:
            directory_path: Either a directory path string or list of file paths
            
        Returns:
            Tuple containing list of loaded documents and formatted context string
        """
        docs = []
        context = ''

        # Handle directory path string vs list of files
        if isinstance(directory_path, str):
            all_files = glob.glob(directory_path)
        else:
            all_files = directory_path
        
        # Group files by type
        pdfs = [f for f in all_files if f.endswith('.pdf')]
        word_docs = [f for f in all_files if f.endswith(('.doc', '.docx'))]
        excel_files = [f for f in all_files if f.endswith(('.xls', '.xlsx'))]
        powerpoint_files = [f for f in all_files if f.endswith(('.ppt', '.pptx'))]
        
        # Load each type
        if pdfs:
            files = self.pdf_loader.load(pdfs)
            docs.extend(files)
            context += self.pdf_loader.format_docs(files)
        if word_docs:
            files = self.word_loader.load(word_docs)
            docs.extend(files)
            context += self.word_loader.format_docs(files)
        if excel_files:
            files = self.excel_loader.load(excel_files)
            docs.extend(files)
            context += self.excel_loader.format_docs(files)
        if powerpoint_files:
            files = self.powerpoint_loader.load(powerpoint_files)
            docs.extend(files)
            context += self.powerpoint_loader.format_docs(files)

        # Add to current list of documents
        if docs:
            self.docs.extend(docs)

        self.context += context
            
        return docs, context
    
    def load_llm(self, llm):
        """Change default llm with user-loaded llm"""
        self.llm = llm
        return self.llm
    
    def load_embedding(self, embedding):
        """Change default embeddings to user-defined embedding"""
        self.embedding = embedding
        return self.embedding

    def chunk_documents(self, docs: Optional[List] = None, chunk_size: int = 1000, 
                         chunk_overlap: int = 100) -> List[str]:
        """Process documents into chunks. For use on list of documents.
    
        Args:
            docs: Optional list of documents. If None, uses self.docs
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        if docs:
            return text_splitter.split_documents(docs)
        else:
            return text_splitter.split_documents(self.docs)
        
    def chunk_text(self, context: Optional[str] = None, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
        """Process text into chunks. For use on context in raw string form.

        Args:
            context: Optional string to chunk. If None, uses self.context
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if context is None:
            context = self.context

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = text_splitter.split_text(context)
        return chunks
    
    def create_vector_store(self, chunks, path) -> None:
        """
        Create vector store from chunked documents. 
        This does NOT need to be called every time: vectorized documents can be referenced later if there are no file changes.

        Args:
        chunks: Either List[Document] objects or List[str] text chunks
        path: Path to save the FAISS index
        
        Returns:
            None
        """
        # Check if chunks are strings or documents
        if chunks and isinstance(chunks[0], str):
            vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=self.embedding,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
        else:
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embedding,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )

        self.vector_store_path = path
        vector_store.save_local(path)

    @staticmethod
    def format_docs(docs) -> str:
        # to be used as runnable for llm; formats retrieved context documents
        return '\n\n'.join([x.page_content for x in docs])
    
    def retrieve(self, question, prompt: Optional[str] = None, path: Optional[str] = None):
        """
        Load vector store using filepath and embeddings.
        Make sure create_vector_store has been called or a vector store path has been provided.
        Set retriever settings, question, and prompt to retrieve from vector store.
        """
        if path is None:
            if self.vector_store_path is None:
                raise ValueError("No vector store path provided") # need to initialize vector store before using retrieve
            path = self.vector_store_path

        vector_store = FAISS.load_local(path, self.embedding, allow_dangerous_deserialization=True)

        retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
        )

        # Allow user to define prompt in retrieve() call; otherwise will use default prompt.
        if prompt is None:
            prompt = self.rag_prompt

        prompt = ChatPromptTemplate.from_template(prompt)

        # Create chain with retriever, llm, and prompt. Format the docs to be used as context for llm
        rag_chain = (
        {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
        | prompt | self.llm | StrOutputParser()
        )

        response = rag_chain.invoke(question)
        return response

# Usage example:
if __name__ == "__main__":
    ppt_loader = PowerPointLoader()
    
    # Load from directory
    docs = ppt_loader.load("path/to/documents")
    
    # Process documents and save as string
    docs = ppt_loader.format_docs(docs)
    
    print(f"Loaded {len(docs)} documents")
    # Can then proceed to chunk strings or feed to LLM.