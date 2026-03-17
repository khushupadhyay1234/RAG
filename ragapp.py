import streamlit as st
import os
import time

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# ===================== API KEY =====================
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Set it in .env file")
    st.stop()

# ===================== LLM =====================
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ===================== PROMPT =====================
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based only on the context below.

<context>
{context}
</context>

Question: {input}
"""
)

# ===================== VECTOR DB =====================
def create_vector_embedding():
    if "vectors" not in st.session_state:
        embeddings = OllamaEmbeddings()

        loader = PyPDFDirectoryLoader("researchpaper")
        docs = loader.load()

        # 🔥 DEBUG
        st.write(f"📄 Loaded documents: {len(docs)}")

        if not docs:
            st.error("❌ No PDFs found in 'data' folder")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_documents = text_splitter.split_documents(docs)

        # 🔥 DEBUG
        st.write(f"✂️ Chunks created: {len(final_documents)}")

        if not final_documents:
            st.error("❌ No text extracted from PDFs")
            st.stop()

        vectors = FAISS.from_documents(final_documents, embeddings)

        st.session_state.vectors = vectors
        st.success("✅ Vector database is ready!")

# ===================== UI =====================
st.title("📄 RAG PDF Chat App")

if st.button("Create Vector DB"):
    create_vector_embedding()

user_prompt = st.text_input("Enter your query from research paper")

# ===================== QUERY =====================
if user_prompt and "vectors" in st.session_state:

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()

    response = retrieval_chain.invoke({'input': user_prompt})

    st.write(f"⏱ Response time: {time.process_time() - start:.2f} sec")

    st.subheader("🧠 Answer")
    st.write(response['answer'])

    with st.expander("📚 Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.markdown(f"**Document {i+1}:**")
            st.write(doc.page_content)
            st.write('-----------------------------')

elif user_prompt:
    st.warning("⚠️ Please create vector database first!")
