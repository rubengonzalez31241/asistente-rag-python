import os
import warnings

# Ocultar advertencias de obsolescencia en la consola
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# 1. Cargar el documento
print("Cargando la base de conocimiento...")
cargador = TextLoader("conocimiento.txt", encoding="utf-8")
documento = cargador.load()

# 2. Dividir en fragmentos
separador = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
fragmentos = separador.split_documents(documento)

# 3. Crear embeddings locales y guardar en FAISS
print("Generando vectores localmente y guardando en FAISS...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
base_vectorial = FAISS.from_documents(fragmentos, embeddings)

print("¡Base de datos vectorial creada con éxito!")

# 4. Probar la búsqueda por similitud
consulta = "¿Qué servicios o precios hay disponibles?"
resultados = base_vectorial.similarity_search(consulta, k=2)

print("\n--- Resultados encontrados ---")
for i, doc in enumerate(resultados, 1):
    print(f"\nFragmento {i}:")
    print(doc.page_content)

# 5. Generar la respuesta final con Groq (LLaMA 3.1)
GROQ_API_KEY = "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"

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

contexto = "\n\n".join([doc.page_content for doc in resultados])

prompt = f"""
Responde a la siguiente pregunta utilizando ÚNICAMENTE la información provista en el contexto.
Si la respuesta no se encuentra en el contexto, di "No dispongo de esa información".

Contexto:
{contexto}

Pregunta: {consulta}
Respuesta:
"""

respuesta = llm.invoke(prompt)

print("\n--- Respuesta de la IA ---")
print(respuesta.content)