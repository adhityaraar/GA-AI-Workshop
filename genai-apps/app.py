import streamlit as st
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from pinecone import Pinecone
from PIL import Image
import fitz
import logging
import os
from dotenv import load_dotenv

load_dotenv()

hf_api_key = os.getenv("HF_KEY", None)
pinecone_api_key = os.getenv("PINECONE_KEY", None)
pinecone_environment = "gcp-starter" #os.getenv("PINECONE_ENV", None)
index_name = "quickstart" #os.getenv("PINECONE_INDEX", None)

# Configure logging
logging.getLogger('llama_index').setLevel(logging.ERROR)

# Streamlit page config
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

def initialize_llm(hf_api_key):
    try:
        # Initialize HuggingFace LLM
        # model_llm = "mistralai/Mixtral-8x22B-v0.1" #"mistralai/Mistral-7B-Instruct-v0.3"
        # model_llm = "meta-llama/Llama-3.1-70B-Instruct"
        # model_llm = "HuggingFaceH4/zephyr-7b-alpha"
        # model_llm = "meta-llama/Meta-Llama-3-70B"
        model_llm = "meta-llama/Llama-3-8B"

        prompt_template = """ 
            Anda adalah asisten yang membantu user dalam menjawab pertanyaan terkait suatu dokumen. Tugas utama Anda adalah menganalisis pertanyaan dari pengguna dan hasil query yang telah disediakan, kemudian memberikan jawaban yang jelas, tepat, dan bermanfaat.

            Konteks: {context}
            Pertanyaan: {query}

            Instruksi bagaimana cara menjawab:
            1. Pastikan jawaban Anda menyertakan penjelasan yang sesuai dengan konteks pertanyaan dan hasil query. 
            2. Jika data kosong, katakan bahwa Anda tidak dapat mendapatkan datanya sehingga pertanyaan tidak dapat dijawab dan coba untuk bertanya lebih spesifik terkait data transaksi.
            3. Hasil query yang disediakan berisi data penting yang perlu dianalisis untuk memberikan jawaban yang akurat. Jika pertanyaan menyangkut jumlah finansial atau uang, formatkan angka tersebut dalam format mata uang Rupiah dengan desimal yang sesuai.
            4. Jawaban harus disampaikan dengan deskriptif yang didapatkan dari hasil query dalam beberapa kalimat yang mudah dipahami oleh user.
            5. Jika Anda tidak dapat memberikan jawaban, jawab dengan sopan bahwa Anda tidak dapat menjawabnya.
            6. Jika jawaban yang diberikan menggunakan huruf kapital semua, ubah formatnya mengikuti kaidah bahasa Indonesia yang benar.
            7. Gunakan Bahasa Indonesia yang baik dan benar, hindari penggunaan tanda petik dan hindari penggunaan tanda baris baru.

            Jawaban: """

        llm = HuggingFaceInferenceAPI(
            model_name=model_llm,
            api_key=hf_api_key,
            max_new_tokens=512,
            temperature=0.1,
            stop_sequences=["\n", ""],
            headers={"Authorization": f"Bearer {hf_api_key}"},
            repetition_penalty=1,
        )       
        
        # Set the LLM as the default for LlamaIndex
        Settings.llm = llm
        Settings.chunk_size = 512
        Settings.chunk_overlap = 20
        
        # Configure prompt template in query engine
        Settings.query_engine_kwargs = {
            "prompt_template": prompt_template,
            "response_mode": "compact"
        }
        
        return True
    
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return False

def process_pdf(uploaded_file, pinecone_index):
    try:
        # Save uploaded file temporarily
        bytes_data = uploaded_file.read()
        with fitz.open(stream=bytes_data, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()

        # Create document
        document = Document(text=text)
        
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Setup vector store
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            show_progress=False
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # Create index
        with st.spinner('Processing document...'):
            index = VectorStoreIndex.from_documents(
                [document],
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=False,
                verbose=False
            )
            
        return index

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def get_clean_response(index, query):
    try:
        response = index.as_query_engine().query(query)
        return response.response if hasattr(response, 'response') else str(response)
    except Exception as e:
        return f"Error generating response: {str(e)}"
    
# def get_clean_response(index, query):
#     try:
#         # Create query engine with custom parameters
#         query_engine = index.as_query_engine(
#             response_synthesizer_kwargs={
#                 "response_mode": "compact",
#                 "text_qa_template": Settings.query_engine_kwargs["prompt_template"]
#             }
#         )
        
#         response = query_engine.query(query)
        
#         # Clean up the response
#         cleaned_response = str(response.response)
#         # Remove any "Query:" or "Answer:" prefixes
#         cleaned_response = cleaned_response.replace("Query:", "").replace("Answer:", "")
#         # Remove multiple newlines
#         cleaned_response = " ".join(cleaned_response.split())
#         # Remove any trailing questions
#         if "?" in cleaned_response:
#             cleaned_response = cleaned_response.split("?")[0] + "?"
            
#         return cleaned_response
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# Sidebar configuration
with st.sidebar:
    st.title("Configuration")
    
    # Add your logo/image here
    # image = Image.open('your_logo.jpg')
    # st.image(image, caption='Your Caption')
        
    st.markdown('''
    ### About
    This app uses:
    - LlamaIndex for document processing
    - Pinecone for vector storage
    - HuggingFace for embeddings and LLM
    ''')

# Main content area
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    # Initialize LLM
    if initialize_llm(hf_api_key):
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
        # pc = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pc.Index(index_name)
        
        # Process the document if not already processed
        if st.session_state.index is None:
            st.session_state.index = process_pdf(uploaded_file, pinecone_index)
            if st.session_state.index:
                st.success("Document processed successfully!")

        # Chat interface
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
                        response = get_clean_response(st.session_state.index, query)
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    if not uploaded_file:
        st.warning("Please upload a PDF document.")
    if not (pinecone_api_key and pinecone_environment and index_name and hf_api_key):
        st.warning("Please provide all required API credentials in the sidebar.")