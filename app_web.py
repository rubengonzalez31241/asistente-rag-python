import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Cargar variables de entorno local (.env) o Secrets de Streamlit
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ No se encontró la API Key de Groq. Configurala en tu archivo .env o en los Secrets de Streamlit.")
    st.stop()

# Configuración de la página en Streamlit
st.set_page_config(page_title="Asistente RAG Comercial", page_icon="🤖", layout="wide")
st.title("🤖 Asistente Virtual Inteligente")

# 2. Sidebar para cargar documentos de base de conocimiento
st.sidebar.header("📁 Base de Conocimiento")
uploaded_files = st.sidebar.file_uploader(
    "Subí los documentos del negocio (PDF o TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# Cargar y procesar documentos
documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path, encoding="utf-8")
            documents.extend(loader.load())

        os.remove(tmp_path)

if documents:
    # Generar índice vectorial con FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Inicializar LLM con Groq
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
# Prompt del sistema que recibe Contexto + Historial
    system_prompt = (
        "Sos un asistente virtual comercial enfocado en tomar pedidos y cerrar ventas.\n\n"
        "REGLAS OBLIGATORIAS DE INTERACCIÓN:\n"
        "1. Revisa el HISTORIAL DE LA CONVERSACIÓN antes de responder para mantener el hilo del pedido actual.\n"
        "2. Usa ÚNICAMENTE este contexto para los productos y condiciones:\n{context}\n\n"
        "HISTORIAL PREVIO DE LA CHARLA:\n{chat_history}\n\n"
        "INSTRUCCIONES DE CIERRE DE VENTA:\n"
        "- Si el cliente ya eligió sus productos y el medio de pago/entrega, NO le preguntes en qué más podés ayudarlo.\n"
        "- Generá de inmediato el RESUMEN FINAL DEL PEDIDO con el descuento correspondiente aplicado (ej: 10% off por efectivo).\n"
        "- Indícale explícitamente que para mandar la orden a la cocina debe enviar ese resumen por WhatsApp al número indicado en el contexto."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # Historial de conversación
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial previo en la interfaz
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Entrada de texto del usuario
    if user_input := st.chat_input("Escribí tu consulta aquí..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Construir el texto del historial para que el LLM tenga memoria
        formatted_history = ""
        for m in st.session_state.messages[:-1]:
            role = "Cliente" if m["role"] == "user" else "Asistente"
            formatted_history += f"{role}: {m['content']}\n"

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Procesando comanda..."):
                # Se busca contexto en FAISS basándose en la pregunta actual
                docs = retriever.invoke(user_input)
                context_text = "\n\n".join([doc.page_content for doc in docs])
                
                # Se formatea el prompt con contexto e historial real
                full_prompt = prompt.format(
                    context=context_text,
                    chat_history=formatted_history if formatted_history else "Sin historial previo.",
                    question=user_input
                )
                
                # Invocar LLM directamente
                response_obj = llm.invoke(full_prompt)
                response = response_obj.content
                st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    