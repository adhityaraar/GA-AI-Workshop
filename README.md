# Building your GenAI with Ollama!

Ensure you have the [Python](https://www.python.org/downloads/) and [Conda](https://docs.anaconda.com/miniconda/) installation before going to the step below!

# There are 3 stages:
1. Play around with your prompt! go to [genai-prompt](https://github.com/adhityaraar/GA-AI-Workshop/tree/main/genai-prompt)
2. Create simple chat with your own personality! go to [genai-chat](https://github.com/adhityaraar/GA-AI-Workshop/tree/main/genai-chat)
3. Building your apps with your own document! go to [genai-apps](https://github.com/adhityaraar/GA-AI-Workshop/tree/main/genai-apps)
4. For the data you can use [here](https://github.com/adhityaraar/GA-AI-Workshop/tree/main/data)

### Please do Ollama installation before jump to any stages!
1. Download the ollama model (linux/windows/mac) [here](https://ollama.com/)
2. You can find any models on [here](https://ollama.com/library)

# HuggingFace API Setup
1. Sign In or Sign Up on [here](https://huggingface.co/).
2. Go to Profile (avatar) and click on "Settings".
3. Go to "Access Token" and click on "New token".
4. Give a user-friendly name to the token and permission=write. Then click on generate token.
5. Copy the token and add it to your .env file. `HF_KEY=<your_new_huggingface_access_token>`
6. Model Request (Example: Go to [meta-llama card](https://huggingface.co/meta-llama) or [meta-llama-3.1-8b card](https://huggingface.co/meta-llama/Llama-3.1-8B)
   
# Pineccone API Setup
1. Sign In or Sign Up on [here](https://www.pinecone.io/).
2. Go to API Keys
3. Copy the token and add it to your .env file. `PINECONE_KEY=<your_pinecone_access_token>`

# Setup the code
1. Open your terminal or console window 
2. Command `Git clone https://github.com/adhityaraar/GA-AI-Workshop.git`
3. Command `cd genai-apps`.
4. Add your API key to .env
5. Run the app by running the command `streamlit run app.py`.
6. Put your documents on `/data` and start asking your queries!

### Contoh Pertanyaan:
Document: [Peraturan Perusahaan](https://github.com/adhityaraar/GA-AI-Workshop/blob/main/data/Peraturan_Perusahaan.pdf)
- Pertanyaan easy
    - Pada pukul berapa karyawan mulai bekerja?
    - Berapa usia pensiun karyawan?
    - Berapa lama masa percobaan karyawan baru?
    
    Apa yang dimaksud dengan cuti karyawan?
    
- Pertanyaan medium
    - Kapan gaji akan dibayarkan jika tanggal 25 jatuh pada hari Sabtu?
    - Apa yang harus dilakukan karyawan jika ingin menggunakan hak cuti tahunannya?
    - pakaian apa yang digunakan pada hari Selasa?
    
- Pertanyaan hard
    - Apa saja tunjangan yang dapat diterima karyawan yang telah bekerja minimal 12 bulan di perusahaan, dan bagaimana aturan terkait Tunjangan Hari Raya (THR) untuk karyawan yang berhenti bekerja paling lama 30 hari sebelum hari raya Idul Fitri?
    - Bagaimana perusahaan mengevaluasi performa karyawan baru selama masa percobaan, dan siapa yang terlibat dalam proses pengawasan dan penilaian?
    - Bicarakan mengenai jaminan sosial karyawan yang diberikan oleh perusahaan, termasuk hak karyawan yang meninggal dan hak keluarga karyawan yang meninggal dalam hal jaminan sosial.
