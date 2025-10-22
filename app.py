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

st.set_page_config(page_title="RAG - Análisis de PDFs", page_icon="💬", layout="centered")

st.title("💬 Generación Aumentada por Recuperación (RAG)")
st.caption(f"🧠 Versión de Python: {platform.python_version()}")

image = Image.open("Chat_pdf.png")
st.image(image, width=380)

st.write(
    "Bienvenido/a a la experiencia **RAG (Retrieval-Augmented Generation)**. 📄🤖  \n"
    "Aquí podrás **subir un archivo PDF**, hacerle preguntas en lenguaje natural y recibir respuestas generadas por un modelo de **IA** "
    "que combina comprensión semántica con información directa del documento. 🧩"
)

with st.sidebar:
    st.subheader("⚙️ Instrucciones de uso")
    st.markdown(
        """
        1️⃣ **Ingresa tu clave de OpenAI** para habilitar el motor de inteligencia artificial.  
        2️⃣ **Carga tu archivo PDF** (máx. unas cuantas páginas de texto).  
        3️⃣ **Escribe tu pregunta** en lenguaje natural.  
        4️⃣ La IA buscará los fragmentos más relevantes y generará una respuesta coherente basada en el contenido.  
        """
    )
    st.info("💡 Consejo: mientras más clara sea tu pregunta, más precisa será la respuesta.")
    st.caption("🔒 Tu información se procesa localmente; solo se usa la API para generar respuestas.")

ke = st.text_input("🔑 Ingresa tu Clave de OpenAI", type="password")
if ke:
    os.environ["OPENAI_API_KEY"] = ke
else:
    st.warning("⚠️ Debes ingresar tu clave de API de OpenAI para continuar.")

pdf = st.file_uploader("📚 Carga tu archivo PDF", type="pdf")

if pdf is not None and ke:
    try:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if not text.strip():
            st.error("🚫 No se pudo extraer texto del PDF. Verifica que el documento no contenga solo imágenes.")
        else:
            st.success(f"✅ Se extrajeron {len(text)} caracteres del documento.")
            st.info("🧩 Dividiendo el texto en fragmentos procesables...")

            splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=100, length_function=len)
            chunks = splitter.split_text(text)
            st.write(f"📄 Documento dividido en **{len(chunks)} fragmentos** para el análisis semántico.")

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            st.markdown("### 💭 Haz una pregunta sobre el documento:")
            user_question = st.text_area("✏️ Escribe tu pregunta aquí...")

            if user_question:
                with st.spinner("🔎 Buscando en la base de conocimiento..."):
                    docs = knowledge_base.similarity_search(user_question, k=4)
                st.success("Fragmentos relevantes encontrados ✅")

                llm = OpenAI(temperature=0, model_name="gpt-4o")
                chain = load_qa_chain(llm, chain_type="stuff")

                with st.spinner("🧠 Generando respuesta con IA..."):
                    response = chain.run(input_documents=docs, question=user_question)

                st.markdown("### 🗣️ Respuesta:")
                st.markdown(f"💡 {response}")

                with st.expander("📘 Fragmentos de texto utilizados en la respuesta"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Fragmento {i}:**")
                        st.write(d.page_content)

    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {e}")
elif pdf is not None and not ke:
    st.warning("⚠️ Por favor ingresa tu clave de API de OpenAI para continuar.")
else:
    st.info("📥 Carga un archivo PDF para comenzar tu análisis.")
