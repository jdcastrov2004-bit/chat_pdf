import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

st.title("Generación Aumentada por Recuperación (RAG) 💬")
st.caption(f"Versión de Python: {platform.python_version()}")

image = Image.open("Chat_pdf.png")
st.image(image, width=350)

st.write(
    "Esta actividad te permite **cargar un PDF**, indexarlo en fragmentos con **embeddings** y hacerle preguntas en lenguaje natural. "
    "El sistema recupera los fragmentos más relevantes y genera una respuesta fundamentada en el contenido del documento."
)

with st.sidebar:
    st.subheader("Instrucciones")
    st.write(
        "1) Ingresa tu clave de OpenAI.\n"
        "2) Carga un PDF.\n"
        "3) Escribe tu pregunta.\n"
        "4) Revisa la respuesta y, si deseas, formula nuevas preguntas."
    )
    st.caption("Tus datos se procesan localmente; la generación se hace vía API de OpenAI.")

ke = st.text_input("Ingresa tu Clave de OpenAI", type="password")
if ke:
    os.environ["OPENAI_API_KEY"] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

if pdf is not None and ke:
    try:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if not text.strip():
            st.error("No se pudo extraer texto del PDF. Verifica que el documento no sea solo imágenes.")
        else:
            st.info(f"Texto extraído: {len(text)} caracteres")
            splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=100, length_function=len)
            chunks = splitter.split_text(text)
            st.success(f"Documento dividido en {len(chunks)} fragmentos")

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            st.subheader("¿Qué quieres saber del documento?")
            user_question = st.text_area("Escribe tu pregunta aquí...")

            if user_question:
                docs = knowledge_base.similarity_search(user_question, k=4)
                llm = OpenAI(temperature=0, model_name="gpt-4o")
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)

                st.markdown("### Respuesta:")
                st.markdown(response)

                with st.expander("Fragmentos recuperados"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Fragmento {i}:**")
                        st.write(d.page_content)
    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Carga un archivo PDF para comenzar")
