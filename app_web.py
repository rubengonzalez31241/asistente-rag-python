import streamlit as st
import os
import warnings
import os
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

st.set_page_config(page_title="Asistente RAG", page_icon="🤖")
st.title("🤖 Asistente de Consulta RAG")

# Reemplazá con tu clave de Groq (gsk_...)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def cargar_base():
    cargador = TextLoader("conocimiento.txt", encoding="utf-8")
    documento = cargador.load()
    separador = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    fragmentos = separador.split_documents(documento)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(fragmentos, embeddings)

base_vectorial = cargar_base()

# Interfaz de Chat
consulta = st.text_input("Escribí tu pregunta:")

if st.button("Consultar") and consulta:
    resultados = base_vectorial.similarity_search(consulta, k=2)
    contexto = "\n\n".join([doc.page_content for doc in resultados])
    
    prompt = f"""
    Responde utilizando ÚNICAMENTE la siguiente información provista.
    Si no está en el texto, di "No dispongo de esa información".
    
    Contexto:
    {contexto}
    
    Pregunta: {consulta}
    """
    
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    respuesta = llm.invoke(prompt)
    
    st.markdown("### Respuesta:")
    st.write(respuesta.content)