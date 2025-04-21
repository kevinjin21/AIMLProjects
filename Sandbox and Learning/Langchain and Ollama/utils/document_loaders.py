from dotenv import load_dotenv
import re
import os
import glob
from typing import List, Union, Optional
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
from langchain_ollama import ChatOllama

load_dotenv()

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
    
    def __init__(self, base_url: str = 'http://localhost:11434', model: str = 'llama3.2'):
        self.web_loader = WebPageLoader()
        self.pdf_loader = PDFLoader()
        self.word_loader = WordLoader()
        self.excel_loader = ExcelLoader()
        self.powerpoint_loader = PowerPointLoader()
        self.youtube_loader = YouTubeLoader()
        self.llm = ChatOllama(base_url=base_url, model=model)

    def load_directory(self, directory_path: str) -> List:
        """Load all supported documents from a directory"""
        all_files = glob.glob(os.path.join(directory_path, '*.*'))
        docs = []
        
        # Group files by type
        pdfs = [f for f in all_files if f.endswith('.pdf')]
        word_docs = [f for f in all_files if f.endswith(('.doc', '.docx'))]
        excel_files = [f for f in all_files if f.endswith(('.xls', '.xlsx'))]
        powerpoint_files = [f for f in all_files if f.endswith(('.ppt', '.pptx'))]
        
        # Load each type
        if pdfs:
            docs.extend(self.pdf_loader.load(pdfs))
        if word_docs:
            docs.extend(self.word_loader.load(word_docs))
        if excel_files:
            docs.extend(self.excel_loader.load(excel_files))
        if powerpoint_files:
            docs.extend(self.powerpoint_loader.load(powerpoint_files))
            
        return docs

    def process_documents(self, docs: List, chunk_size: int = 1000, 
                         chunk_overlap: int = 100) -> List[str]:
        """Process documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)

# Usage example:
if __name__ == "__main__":
    manager = DocumentManager()
    
    # Load from directory
    docs = manager.load_directory("path/to/documents")
    
    # Process documents
    chunks = manager.process_documents(docs)
    
    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} chunks")