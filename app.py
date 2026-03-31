import streamlit as st
import os
from typing import TypedDict

from langgraph.graph import StateGraph
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  
st.set_page_config(page_title="📄 AI PDF Assistant", layout="wide")
st.title("📄 AI PDF Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    chat_history: list

if uploaded_file is not None and "graph" not in st.session_state:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(model="llama-3.1-8b-instant")

    def retrieve_node(state):
        docs = vectorstore.similarity_search(state["question"])
        context = " ".join([doc.page_content for doc in docs])
        return {"context": context}

    def generate_node(state):
        prompt = f"""
Context:
{state['context']}

Question:
{state['question']}
"""
        response = llm.invoke(prompt)
        new_history = state["chat_history"] + [
            f"User: {state['question']}",
            f"AI: {response.content}"
        ]
        return {
            "answer": response.content,
            "chat_history": new_history
        }

    builder = StateGraph(GraphState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.set_finish_point("generate")

    st.session_state.graph = builder.compile()
    st.session_state.chat_history = []

question = st.text_input("Ask something:")

if uploaded_file is None:
    st.warning("Please upload a PDF first 📄")

if uploaded_file is not None and "graph" in st.session_state and st.button("Ask") and question:
    try:
        result = st.session_state.graph.invoke({
            "question": question,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history = result["chat_history"]

        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(result["answer"])

    except Exception as e:
        st.error(f"Error: {e}")

if uploaded_file is not None and "graph" not in st.session_state:
    st.info("Upload PDF to initialize AI")
