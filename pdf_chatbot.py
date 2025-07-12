import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="PDF Insight ChatBot")
st.title("PDF Insight ChatBot")
st.markdown("Chat with your PDF using the AI ChatBot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
question = st.text_input("Ask a question about the PDF")

if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()

    # Iam using Ollama Mistral modeln as pre-trained bot
    llm = Ollama(model="mistral")
    chain = load_qa_chain(llm, chain_type="stuff")

  
    response = chain.run(input_documents=pages, question=question)

    st.markdown("Answer:")
    st.write(response)
