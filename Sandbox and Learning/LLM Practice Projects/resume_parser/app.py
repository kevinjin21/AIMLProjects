# Simple streamlit application to parse resume via LLM
# Streamlit playlist: https://www.youtube.com/watch?v=hff2tHUzxJM&list=PLc2rvfiptPSSpZ99EnJbH5LjTJ_nOoSWW

# to run in terminal: move to current folder (resume_parser); streamlit run .\app.py

import streamlit as st
import pymupdf

from scripts.llm import ask_llm, validate_json

st.title("Resume Parsing")
st.write("Upload a resume in PDF format to extract information")

# allow user to upload a file from local machine
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    bytearray = uploaded_file.read()
    pdf = pymupdf.open(stream=bytearray, filetype="pdf") # only works with pdf files currently

    context = ""
    for page in pdf:
        context = context + "\n\n" + page.get_text()

    pdf.close()


question = """You are tasked with parsing a job resume. Your goal is to extract relevant information in a valid structured 'JSON' format. 
                Do not write preambles or explanations."""

if st.button("Parse Resume"):
    with st.spinner("Parsing Resume..."):
        response = ask_llm(context=context, question=question)

    
    with st.spinner("Validating JSON..."):
        response = validate_json(response)

    
    st.write("**Extracted Information**")
    st.write(response)

    st.write("You can copy the JSON output and use it in your application.")

    #st.balloons() # add some fun balloons when the parsing is done
    st.snow()