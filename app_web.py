import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
Python

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Asistente RAG Comercial", layout="wide")
st.title("🤖 Asistente Virtual Inteligente")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Falta configurar la variable GROQ_API_KEY en los Secrets de Streamlit.")
    st.stop()

# Configuración de modelos
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Panel lateral para carga de documentos
st.sidebar.header("📁 Base de Conocimiento")
uploaded_files = st.sidebar.file_uploader(
    "Subí los documentos del negocio (PDF o TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

documents = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
            
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = uploaded_file.name
        documents.extend(loaded_docs)
        os.remove(tmp_path)

if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

    system_prompt = (
        "Sos un asistente virtual comercial profesional y educado.\n"
        "Instrucciones de respuesta:\n"
        "1. Si el usuario te saluda o hace conversación informal (ej. 'hola', 'buenas tardes'), responde amablemente y ofrece ayuda sobre la información del negocio.\n"
        "2. Si realiza una consulta específica, responde de forma clara utilizando ÚNICAMENTE el siguiente contexto:\n\n"
        "{context}\n\n"
        "3. Si la respuesta no está en el contexto y no es un saludo, indica de forma educada que no dispones de esa información en la base actual."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Escribí tu consulta aquí..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            
            # Cita de fuentes
            sources = set([doc.metadata.get("source", "Documento") for doc in response.get("context", [])])
            if sources and "no dispongo" not in answer.lower():
                answer += f"\n\n---\n**Fuente consultada:** {', '.join(sources)}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Cargá uno o más archivos PDF/TXT en el panel lateral para iniciar la base de conocimiento.")