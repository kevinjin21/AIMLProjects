from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime
import os
import re
import yaml
import getpass
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

def youtube_transcript(url):
    """Load a YouTube transcript from url. Allow user to re-input url if it is invalid format or not found.
    
    Args:
        url: The YouTube video URL to get transcript from
        
    Returns:
        Chunked documents to be processed by LLM
    """
    video_id = extract_video_id(url)
    if not video_id:
        print("Invalid YouTube URL format")
        return None
    
    try:
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=False, 
        )
        transcript = loader.load()
        print(f"Loaded transcript of len: {len(transcript[0].page_content)} from {url}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        docs = text_splitter.split_documents(transcript)
        return docs
    except Exception as e:
        print(f"Error loading transcript: {e}")
        return None
    
def extract_video_id(url):
    """Extract the video ID from various YouTube URL formats."""
    video_id = None
    # Regular YouTube URL
    if "youtube.com/watch" in url:
        video_id = url.split("v=")[1].split("&")[0]
    # Shortened youtu.be URL
    elif "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
    return video_id

def get_user_text_input():
    """Allow user to paste text directly when transcript isn't available."""
    print("No transcript available. Please paste text to analyze:")
    lines = []
    print("Enter text (type 'END' on a new line when finished):")
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)

def create_format_extraction_prompt(chunk, chunk_number, total_chunks):
    prompt = f"""
    You are a literary analyst with expertise in identifying story structures, writing styles, and narrative patterns.

    I'll provide you with chunk {chunk_number} of {total_chunks} from a story or series of stories by the same author. Your task is to analyze the format and style elements in detail.

    For each chunk, identify:

    1. NARRATIVE STRUCTURE:
    - How does the author structure scenes?
    - Are there patterns in how the story progresses?
    - How are transitions handled?

    2. STYLISTIC ELEMENTS:
    - Sentence structure patterns (length, complexity, variety)
    - Paragraph structure and length patterns
    - Distinctive vocabulary or word choices
    - Use of literary devices (metaphors, similes, etc.)
    - Tone and mood creation techniques

    3. CHARACTER PRESENTATION:
    - How are characters introduced and developed?
    - Patterns in character descriptions
    - How is character dialogue formatted and styled?

    4. SETTING & DESCRIPTION:
    - How are settings established?
    - Balance between description and action
    - Sensory detail patterns

    5. DIALOGUE PATTERNS:
    - Dialogue tag usage
    - Formatting of conversations
    - Balance between dialogue and narrative

    6. UNIQUE FORMATTING ELEMENTS:
    - Any special formatting for specific story elements
    - Section breaks or chapter structures
    - Any recurring motifs or themes

    Provide concrete examples from the text for each element you identify. Focus on the FORMAT rather than the specific content.

    TEXT CHUNK:
    {chunk}

    FORMAT ANALYSIS:
    """
    return prompt

def create_consolidation_prompt(chunk_analyses):
    prompt = f"""
    You are a literary format expert. You've been given analyses of multiple chunks from the same author's work(s).

    Your task is to consolidate these analyses into a comprehensive understanding of the author's format and style. Create a detailed format guide that captures the essence of how the author structures and styles their stories.

    The format guide should be detailed enough that it could be used as instructions for generating new stories in the same style, but with different characters and plots.

    Focus on patterns that appear consistently across the analyses. Resolve any contradictions by favoring the most frequently observed patterns.

    CHUNK ANALYSES:
    {chunk_analyses}

    CONSOLIDATED FORMAT GUIDE:
    """
    return prompt

def create_story_generation_prompt(format_guide, user_elements):
    prompt = f"""
    You are a creative writer tasked with generating a new story that follows a specific format and style guide, while incorporating user-specified elements.

    FORMAT GUIDE:
    {format_guide}

    USER-SPECIFIED ELEMENTS:
    {user_elements}

    Instructions:
    1. Create a new, original story that strictly adheres to the format and style described in the format guide.
    2. Incorporate all the user-specified elements (characters, settings, themes, etc.) into your story.
    3. The story should feel as if it was written by the same author whose style was analyzed to create the format guide.
    4. Focus on matching the structural and stylistic elements rather than copying specific plot points.

    STORY:
    """
    return prompt

def get_llm_response(llm, prompt):
    """Get response from LLM"""
    response = llm.invoke(prompt)
    return response.content

def extract_story_format(llm, url, transcript_files=[]):
    """Extract format from one or more youtube urls
    
    Args:
        llm: Language model to use for analysis
        url: Either a single YouTube URL (string) or a list of URLs
        
    Returns:
        A format guide generated from analyzing the transcript(s)
    """
    # Check if url is a single string or a list
    if isinstance(url, str):
        urls = [url]  # Convert single URL to a list with one element
    else:
        urls = url  # Use the list of URLs directly
    
    print(f'📖Extracting story format from {len(urls)} URL(s)')
    
    # Step 1: Chunk the text from all URLs
    all_chunks = []
    for single_url in urls:
        print(f'Processing URL: {single_url}')
        transcripts = youtube_transcript(single_url)  # Currently in Document object format
        if transcripts:  # Check if we got valid transcripts
            # Extract page content from each transcript document
            url_chunks = [transcript.page_content for transcript in transcripts]
            all_chunks.extend(url_chunks)
    
    # Now all_chunks contains transcript chunks from all URLs
    chunks = all_chunks
    
    # Process transcript files if provided
    if transcript_files:
        print(f'Processing {len(transcript_files)} transcript files')
        for file_path in transcript_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    print(f"Loaded transcript from file: {file_path}")
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100,
                    )
                    file_chunks = text_splitter.split_text(text)
                    chunks.extend(file_chunks)
                    
            except Exception as e:
                print(f"Error loading transcript from file {file_path}: {e}")
    
    if not chunks:
        raise ValueError("❌No valid transcripts were found from URLs or files. Generation stopping...")

    print(f'📖Analyzing {len(chunks)} chunks to generate format...')
    # Step 2: Analyze each chunk
    chunk_analyses = []
    for i, chunk in enumerate(chunks):
        extraction_prompt = create_format_extraction_prompt(chunk, i+1, len(chunks))
        analysis = get_llm_response(llm, extraction_prompt)
        chunk_analyses.append(analysis)

    # Step 3: Consolidate the analyses and create format guide
    consolidated_analyses = "\n\n".join(chunk_analyses)
    print(f'Consolidated analysis of transcript. Creating format guide generation prompt...')
    consolidation_prompt = create_consolidation_prompt(consolidated_analyses)
    format_guide = get_llm_response(llm, consolidation_prompt)
    print(f'Format Guide generated...\n {format_guide}')
    return format_guide

def generate_new_story(llm, format_guide, user_elements):
    """Generate a new story using the extracted format"""
    generation_prompt = create_story_generation_prompt(format_guide, user_elements)
    print(f'✏️Generating new story with user elements: {user_elements}')
    new_story = get_llm_response(llm, generation_prompt)
    print(f"💭New story generated!\n{new_story}")
    return new_story

def save_to_markdown(content, filename, url, directory='generations'):
    """
    Save content to a markdown file in the specified directory
    
    Args:
        content: The markdown content to save
        filename: Name for the markdown file (without extension)
        directory: Directory to save the file in (will be created if it doesn't exist)
    
    Returns:
        Path to the saved file
    """
    import os
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Add .md extension if not present
    if not filename.endswith('.md'):
        filename = f"{filename}.md"
    
    file_path = os.path.join(directory, filename)

    # Add url metadata to credit youtuber for writing style
    content += f"\n\nStory style inspired by Youtube video URL: {url}"
    
    # Write content to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Markdown file saved successfully: {file_path}")
    return file_path

def main():
    # TODO CLEANUP MAIN, ADD BETTER WAY TO INPUT YOUTUBE URL, USER ELEMNTS
    # TODO CLEANUP CODE BEFORE ADDING TO GITHUB REPO
    # todo add images to readme.md
    url='https://www.youtube.com/watch?v=0XDiWYFGGqY'
    transcripts = ['transcripts/dry_humor_example.txt']
    pre_loaded_format = 'formats/.*'

    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")
    
    claude = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )

    # User specifies elements for a new story
    default_user_elements = """
    Characters:
    - Rick: A grandfather scientist with an alcohol problem
    - Morty: A simple grandson with a big heart

    Setting:
    - Midwest USA

    Theme:
    - Adventure and science-fiction

    Length:
    - Less than 1000 words
    """

    try:
        with open('story_templates.yaml', 'r') as f:
            templates = yaml.safe_load(f)
            user_elements = templates["dry_humor_office"]["elements"]
    except Exception as e:
        print(f"❌ Unexpected error with template file: {e}. Using default user elements.")
        user_elements = default_user_elements

    # Load existing format or provide URL to generate new format
    load_format = input("Would you like to load a pre-generated format? (y/n): ").lower().strip() == 'y'
    if load_format:
        with open(pre_loaded_format, 'r', encoding='utf-8') as file:
            format_guide = file.read()

    else:
        format_guide = extract_story_format(claude, url, transcripts)

        # Optionally save the format guide for future use
        #todo save_format = input("Would you like to save the format guide for future use? (y/n): ").lower().strip() == 'y'
        save_format = True
        if save_format:
            # Generate a meaningful filename based on the URL or video ID
            if isinstance(url, str):
                url_id = extract_video_id(url) or "unknown"
            else:
                url_id = "multiple_sources"
            
            format_filename = f"format_guide_{url_id}"
            format_path = save_to_markdown(format_guide, format_filename, url, directory='formats')
            print(f"Format guide saved for future use at: {format_path}")
    
    # Generate new story based on format
    new_story = generate_new_story(claude, format_guide, user_elements)

    # Save story to markdown file
    # Generate filename from elements or timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_story_{timestamp}"
    
    # Save both the story and the format guide
    save_to_markdown(new_story, filename, url)
    #save_to_markdown(format_guide, f"generations/format_guide_{timestamp}")
    #todo add a way to save title into markdown or have llm generate title as well
    
if __name__ == "__main__":
    main()