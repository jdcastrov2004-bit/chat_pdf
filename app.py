import os
import streamlit as st
from PIL import Image
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import platform
import time
from streamlit_lottie import st_lottie
import json
import langchain
set_verbose(True)

# Configuración de la página
st.set_page_config(
    page_title="RAG PDF Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #152b63;
    }
    .results-area {
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
    }
    .footer {
        margin-top: 2rem;
        color: #6B7280;
        font-size: 0.8rem;
        text-align: center;
    }
    .api-input {
        margin-bottom: 1rem;
    }
    .file-uploader {
        margin-bottom: 1.5rem;
    }
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .personality-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Panel lateral
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 RAG PDF Analyzer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Este asistente te ayudará a analizar documentos PDF utilizando tecnología RAG 
    (Generación Aumentada por Recuperación) y modelos de lenguaje avanzados.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Configuración")
    st.markdown('<div class="api-input">', unsafe_allow_html=True)
    ke = st.text_input('API Key de OpenAI', type="password", 
                      help="Introduce tu clave de API de OpenAI para activar el análisis")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sección de personalidad del bot
    st.markdown("### Personalidad del Asistente")
    
    default_prompt = "Eres un asistente experto en análisis de documentos. Responde de manera clara, precisa y profesional."
    
    personality_options = {
        "Profesional": "Eres un asistente experto en análisis de documentos. Responde de manera clara, precisa y profesional.",
        "Amigable": "Eres un asistente amigable y cercano. Explica los conceptos de forma sencilla y usa un tono conversacional y empático.",
        "Académico": "Eres un asistente académico con amplios conocimientos. Proporciona respuestas detalladas con referencias a conceptos relevantes. Utiliza un lenguaje técnico cuando sea apropiado.",
        "Conciso": "Eres un asistente que valora la brevedad. Proporciona respuestas directas y concisas sin información superflua. Ve directo al punto.",
        "Creativo": "Eres un asistente creativo y entusiasta. Usa metáforas, analogías y ejemplos imaginativos para explicar conceptos. Mantén un tono inspirador.",
        "Personalizado": "custom"
    }
    
    personality_type = st.selectbox(
        "Selecciona la personalidad del asistente",
        options=list(personality_options.keys()),
        help="Define cómo responderá el asistente a tus preguntas"
    )
    
    if personality_type == "Personalizado":
        system_prompt = st.text_area(
            "Define el prompt de sistema personalizado",
            default_prompt,
            help="Instrucciones que definen cómo debe comportarse el asistente"
        )
    else:
        system_prompt = personality_options[personality_type]
    
    # Mostrar ejemplo de la personalidad seleccionada
    st.markdown(f"""
    <div class="personality-box">
    <strong>Vista previa:</strong><br>
    <em>{system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}</em>
    </div>
    """, unsafe_allow_html=True)
        
    if ke:
        st.success("✅ API Key configurada")
    
    st.markdown("---")
    
    # Información técnica
    with st.expander("ℹ️ Información técnica"):
        st.write("Versión de Python:", platform.python_version())
        st.write("Modelo: GPT-4o-mini")
        st.write("Chunk size: 500")
        st.write("Chunk overlap: 20")
    
    st.markdown("---")
    
    # Footer en el sidebar
    st.markdown('<div class="footer">RAG PDF Analyzer v1.0</div>', unsafe_allow_html=True)

# Contenido principal
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="main-header">📚 Generación Aumentada por Recuperación (RAG)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Sube un documento PDF y haz preguntas sobre su contenido. 
    El sistema utilizará tecnología RAG para encontrar las respuestas más relevantes.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="container">', unsafe_allow_html=True)
    #image = Image.open('Chat_pdf.png')
    #st.image(image, width=200)
    with open('robotS.json') as source:
         animation=json.load(source)
    st_lottie(animation, width=350)
    st.markdown('</div>', unsafe_allow_html=True)

# Configurar API key
if ke:
    os.environ['OPENAI_API_KEY'] = ke

    # Cargar archivo PDF predeterminado
    try:
        pdfFileObj = open('example.pdf', 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
    except:
        st.warning("Archivo example.pdf no encontrado en el directorio. Se utilizará el archivo que subas.")

    # Sección de carga de archivo
    st.markdown('<div class="sub-header">📤 Sube tu documento</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        pdf = st.file_uploader("Selecciona un archivo PDF", type="pdf", 
                              help="El archivo será procesado localmente")
        st.markdown('</div>', unsafe_allow_html=True)

    # Procesar el PDF
    if pdf is not None:
        with st.spinner("Procesando documento..."):
            # Mostrar información del documento
            col1, col2, col3 = st.columns(3)
            
            # Extraer texto
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            with col1:
                st.metric("Páginas", len(pdf_reader.pages))
            with col2:
                st.metric("Caracteres", len(text))
            with col3:
                st.metric("Tamaño", f"{round(len(pdf.getvalue())/1024, 2)} KB")
            
            # Dividir en chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=20,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Progress bar simulado para la creación de embeddings
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Crear embeddings
            with st.spinner("Generando base de conocimiento..."):
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                st.success("✅ Documento procesado correctamente")
            
            # Sección de preguntas
            st.markdown('<div class="sub-header">🔍 Realiza tu consulta</div>', unsafe_allow_html=True)
            
            # Mostrar la personalidad activa
            st.markdown(f"""
            <div class="personality-box">
            <strong>🤖 Personalidad activa:</strong> {personality_type}
            </div>
            """, unsafe_allow_html=True)
            
            user_question = st.text_area(
                "Escribe tu pregunta sobre el documento",
                placeholder="Ej: ¿Cuáles son los puntos principales del documento?",
                height=100
            )
            
            if user_question:
                with st.spinner("Analizando tu pregunta..."):
                    docs = knowledge_base.similarity_search(user_question)
                    llm = OpenAI(model_name="gpt-4o-mini")
                    
                    # Configurar el modelo con la personalidad seleccionada
                    formatted_prompt = f"""
{system_prompt}

Contexto del documento:
{docs}

Pregunta del usuario:
{user_question}

Respuesta:
"""
                    # Crear la cadena con el prompt personalizado
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                        
                        # Mostrar resultados
                        st.markdown('<div class="results-area">', unsafe_allow_html=True)
                        st.markdown("### Respuesta:")
                        st.markdown(response)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Mostrar estadísticas de uso
                        with st.expander("📊 Detalles de procesamiento"):
                            st.write(f"Tokens totales: {cb.total_tokens}")
                            st.write(f"Tokens de prompt: {cb.prompt_tokens}")
                            st.write(f"Tokens de completado: {cb.completion_tokens}")
                            st.write(f"Costo estimado: ${cb.total_cost:.5f}")
                            st.write(f"Personalidad: {personality_type}")
else:
    st.warning("⚠️ Por favor, introduce tu API Key de OpenAI en el panel lateral para comenzar.")
    st.info("Esta aplicación requiere una clave de API válida de OpenAI para funcionar correctamente.")
