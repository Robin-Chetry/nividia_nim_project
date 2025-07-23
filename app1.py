import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Streamlit app title
st.title("🚀 NVIDIA NIM - RAG PDF Chat Demo")

# Define prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")

# Initialize LLM
llm = ChatNVIDIA(model="moonshotai/kimi-k2-instruct")


# Vector embedding logic
def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Loading documents and generating embeddings..."):
            # Load and preprocess PDF documents
            loader = PyPDFDirectoryLoader("./us_census")
            docs = loader.load()

            # Chunking text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            chunks = text_splitter.split_documents(docs[:30])

            # Generate vector store
            embeddings = NVIDIAEmbeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)

            # Store in session
            st.session_state.vectors = vector_store
            st.session_state.final_documents = chunks


# Button to build the vector store
if st.button("📚 Embed PDF Documents"):
    vector_embedding()
    st.success("Vector Store DB is ready!")

# Question input from user
user_query = st.text_input("🔎 Ask a question based on the documents:")

# If a query is asked
if user_query and "vectors" in st.session_state:
    # Create the RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_query})
    end = time.process_time()

    # Display response
    st.markdown("### 🧠 Answer:")
    st.write(response['answer'])
    st.caption(f"⏱️ Response time: {end - start:.2f}s")

    # Show relevant context
    with st.expander("📄 Document Chunks Used"):
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.write("---")
