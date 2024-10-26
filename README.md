# Building  your GenAI Apps with Streamlit

Ensure you have the Python and Conda installation before going to the step below!

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
1. Please download the ollama model (linux/windows/mac) [here](https://ollama.com/)
2. You can find any models on [here](https://ollama.com/library)
3. Open your terminal or console window 
4. Command `Git clone https://github.com/adhityaraar/GA-AI-Workshop.git`
5. Command `cd genai-apps`.
6. Add your API key to .env
7. Run the app by running the command `streamlit run app.py`.
8. Put your documents on `/data` and start asking your queries!

### Contoh Pertanyaan:
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
