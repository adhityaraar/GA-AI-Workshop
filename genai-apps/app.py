import streamlit as st
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from pinecone import Pinecone
from PIL import Image
import fitz
import logging
import os
from dotenv import load_dotenv

load_dotenv()

hf_api_key = os.getenv("HF_KEY", None)
pinecone_api_key = os.getenv("PINECONE_KEY", None)
pinecone_environment = "gcp-starter"
index_name = "quickstart"

# Configure logging
logging.getLogger('llama_index').setLevel(logging.ERROR)

# Configure the embedding model with 384 dimensions
embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"  # This model produces 384-dimensional embeddings
)

# Configure Ollama LLM
llm = Ollama(model="llama3.2", 
             request_timeout=30.0, 
             max_tokens=300, 
             min_tokens=8,
             stop_sequences=["\n"])

st.set_page_config(
    page_title="Document Question Answering",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
header_text = 'Document QA with LlamaIndex <span style="color: blue; font-family: Cormorant Garamond; font-size: 40px;">| AI Assistant</span>'
st.markdown(f'<h1 style="color: black;">{header_text}</h1>', unsafe_allow_html=True)

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload a document to get started."}]

# def process_pdf(uploaded_file):
#     try:
#         bytes_data = uploaded_file.read()
#         with fitz.open(stream=bytes_data, filetype="pdf") as doc:
#             text = ""
#             for page in doc:
#                 text += page.get_text()

#         document = Document(text=text)
#         return document
    
#     except Exception as e:
#         st.error(f"Error processing PDF: {str(e)}")
#         return None

import tempfile
import fitz  # PyMuPDF
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader

def process_pdf(uploaded_file):
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary file path
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save uploaded file to temporary directory
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Use SimpleDirectoryReader to load the PDF
            reader = SimpleDirectoryReader(input_dir=temp_dir)
            documents = reader.load_data()

            print("There")
            print("document", documents)
            
            if documents and len(documents) > 0:
                return documents

            else:
                st.error("No documents were loaded from the PDF")
                return None
                
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def index_document(document, pinecone_index):
    try:
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            top_n=3,
            show_progress=True
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # Create index with embedding model and LLM directly
        index = VectorStoreIndex.from_documents(
            document,
            storage_context=storage_context,
            embed_model=embed_model,
            llm=llm,
            show_progress=True
        )

        print("Where")
        print("Indexing", index)
        return index
    
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")
        return None

def get_response_from_query(index, query):
    try:
        # Define the custom prompt template
        prompt_template = """ 
            Anda adalah asisten yang membantu user dalam menjawab pertanyaan terkait suatu dokumen. Tugas utama Anda adalah menganalisis pertanyaan dari pengguna dan hasil query yang telah disediakan, kemudian memberikan jawaban yang jelas, tepat, dan bermanfaat.

            Konteks: {context}

            Instruksi bagaimana cara menjawab:
            1. Pastikan jawaban Anda menyertakan penjelasan yang sesuai dengan konteks. 
            2. Jika data kosong, katakan bahwa Anda tidak dapat mendapatkan datanya sehingga pertanyaan tidak dapat dijawab dan coba untuk bertanya lebih spesifik terkait data transaksi.
            3. Hasil jawaban yang disediakan berisi data penting yang perlu dianalisis untuk memberikan jawaban yang akurat. Jika pertanyaan menyangkut jumlah finansial atau uang, formatkan angka tersebut dalam format mata uang Rupiah dengan desimal yang sesuai.
            4. Jika Anda tidak dapat memberikan jawaban, jawab dengan sopan bahwa Anda tidak dapat menjawabnya.
            5. Jika jawaban yang diberikan menggunakan huruf kapital semua, ubah formatnya mengikuti kaidah bahasa Indonesia yang benar.
            6. Jawaban harus menggunakan Bahasa Indonesia yang baik dan benar, hindari penggunaan tanda petik dan hindari penggunaan tanda baris baru.

            Pertanyaan: {query}
            Jawaban: 
        """

        query_engine = index.as_query_engine(
            streaming=True,
            embed_model=embed_model,
            llm=llm,
            prompt_template=prompt_template
        )
        
        # Get streaming response
        streaming_response = query_engine.query(query)
        
        return streaming_response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None
    
# Sidebar configuration
with st.sidebar:
    st.title("Configuration")
    
    st.markdown('''
    ### About
    This app uses:
    - LlamaIndex for document processing
    - Pinecone for vector storage
    - HuggingFace for embeddings (all-MiniLM-L6-v2)
    - Ollama (llama2) for LLM
    ''')

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pc.Index(index_name)

        if st.session_state.index is None:
            st.session_state.index = process_pdf(uploaded_file)
            if st.session_state.index:
                st.success("Document processed and indexed successfully!")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if query := st.chat_input("Ask a question about your document:"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            if st.session_state.index:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        index = index_document(st.session_state.index, pinecone_index)
                        if index:
                            response = get_response_from_query(index, query)
                            if response:
                                response_placeholder = st.empty()
                                full_response = ""
                                
                                # Handle streaming response
                                for text in response.response_gen:
                                    full_response += text
                                    response_placeholder.write(full_response)
                                
                                st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"Error connecting to Pinecone: {str(e)}")

else:
    if not uploaded_file:
        st.warning("Please upload a PDF document.")
