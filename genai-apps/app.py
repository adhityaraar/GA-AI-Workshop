import logging
import os
import tempfile
from pathlib import Path
import streamlit as st
import ollama
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings  # Changed from HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PIL import Image

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

header_text = 'Document Question Answering <span style="color: blue; font-family: Cormorant Garamond; font-size: 40px;">| Ollama</span>'
st.markdown(f'<h1 style="color: black;">{header_text}</h1>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # Update the path to your logo/image
    if Path('watsonxai.jpg').exists():
        image = Image.open('watsonxai.jpg')
        st.image(image, caption='Powered by Ollama')

    st.write("Configure model and parameters:")

    model_option = st.selectbox("Model Selected:", ["llama3.1:8b", "llama3.2", "llama2", "mistral"])
    max_tokens = st.number_input("Max Tokens:", 1, 4096, value=256)
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.7)
    
    st.markdown('''
    This app is an LLM-powered RAG built using:
    - [Ollama](https://ollama.ai/)
    - [HuggingFace](https://huggingface.co/)
    - [LangChain](https://python.langchain.com/docs/get_started/introduction)
    ''')

st.markdown('<div style="text-align: right;">Powered by <span style="color: darkblue;">Ollama</span></div>', unsafe_allow_html=True)

def get_ollama_response(model, context, query):
    """Get response from Ollama model"""
    prompt_template = """ 
        Anda adalah asisten yang membantu user dalam menjawab pertanyaan terkait suatu dokumen. Tugas utama Anda adalah menganalisis pertanyaan dari pengguna dan hasil query yang telah disediakan, kemudian memberikan jawaban yang jelas, tepat, dan bermanfaat.

        Pertanyaan: {query}
        Konteks: {context}

        Instruksi bagaimana cara menjawab:
        1. Pastikan jawaban Anda menyertakan penjelasan yang sesuai dengan konteks. 
        2. Jika data kosong, katakan bahwa Anda tidak dapat mendapatkan datanya sehingga pertanyaan tidak dapat dijawab dan coba untuk bertanya lebih spesifik terkait data transaksi.
        3. Hasil jawaban yang disediakan berisi data penting yang perlu dianalisis untuk memberikan jawaban yang akurat. Jika pertanyaan menyangkut jumlah finansial atau uang, formatkan angka tersebut dalam format mata uang Rupiah dengan desimal yang sesuai.
        4. Jika Anda tidak dapat memberikan jawaban, jawab dengan sopan bahwa Anda tidak dapat menjawabnya.
        5. Jika jawaban yang diberikan menggunakan huruf kapital semua, ubah formatnya mengikuti kaidah bahasa Indonesia yang benar.
        6. Jawaban harus menggunakan Bahasa Indonesia yang baik dan benar, hindari penggunaan tanda petik dan hindari penggunaan tanda baris baru.

        Jawaban: 
    """
    
    prompt = prompt_template.format(context=context, query=query)
    messages = [{'role': 'user', 'content': prompt}]
    
    response_stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            'temperature': temperature,
            'num_predict': max_tokens,
        }
    )
    
    for chunk in response_stream:
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']

@st.cache_data
def process_pdf(uploaded_files, chunk_size=200, chunk_overlap=20):
    """Process uploaded PDF files"""
    documents = []
    
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
            temp_file.write(bytes_data)
            filepath = temp_file.name
            
            with st.spinner('Processing the uploaded file...'):
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def create_embeddings(docs):

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory="chroma_storage"
        )
    return db

# File upload and processing
uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type=["pdf"])

if uploaded_file:
    docs = process_pdf(uploaded_file)
    # print("Documents uploaded:\n", docs)

    db = create_embeddings(docs)
else:
    st.error("Please upload your document(s)")

st.markdown('<hr style="border: 1px solid #f0f2f6;">', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about your documents!"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if user_question := st.chat_input("Send a message...", key="prompt"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Processing your question..."):
                try:
                    if 'db' in locals():
                        docs_search = db.similarity_search(user_question, k=5)
                        print("Documents found:\n", docs_search)

                        context = "\n".join([doc.page_content for doc in docs_search])
                        print("\n\nKnowledge context:'\n", context)
                        
                        placeholder = st.empty()
                        full_response = ''
                        
                        for response_chunk in get_ollama_response(model_option, context, user_question):
                            full_response += response_chunk
                            placeholder.markdown(full_response)
                            
                        message = {"role": "assistant", "content": full_response}
                        st.session_state.messages.append(message)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    response = "Sorry, an error occurred while processing your request."
                    st.session_state.messages.append({"role": "assistant", "content": response})