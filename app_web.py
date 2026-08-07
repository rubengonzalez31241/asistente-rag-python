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

    # Prompt del sistema enfocado en atención comercial y cierre de ventas
    system_prompt = (
        "Sos un asistente virtual comercial profesional y enfocado en ventas.\n"
        "Instrucciones de respuesta:\n"
        "1. Si el usuario saluda o hace conversación informal (ej. 'hola', 'buenas tardes'), responde amablemente ofreciendo ayuda.\n"
        "2. Si realiza consultas específicas, responde utilizando ÚNICAMENTE el siguiente contexto:\n\n"
        "{context}\n\n"
        "3. Si el usuario muestra intención de comprar o agregar productos a su pedido (ej. 'quiero comprar', 'me lo llevo', 'agregalo'):\n"
        "   - Armá un resumen del pedido con el producto, talle/variante y precio.\n"
        "   - Pregúntale los datos faltantes para cerrar la orden (medio de pago y si es con envío a domicilio o retiro).\n"
        "   - Cuando confirme, mostrale el total a pagar y dale el número/enlace de WhatsApp del negocio indicado en el contexto para coordinar el pago y la entrega.\n"
        "4. Si la respuesta no está en el contexto y no es un saludo, indica de forma educada que no dispones de esa información en la base actual."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # Construir la cadena RAG
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Historial de conversación en Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial previo
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Entrada de texto del usuario
    if user_input := st.chat_input("Escribí tu consulta aquí..."):
        # Agregar y mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            with st.spinner("Procesando consulta...")
                response = rag_chain.invoke(user_input)
                st.write(response)
        
        # Guardar respuesta en el historial
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👈 Por favor, subí al menos un archivo (.txt o .pdf) en el panel lateral para comenzar.")