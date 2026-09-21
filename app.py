import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Medical Assistant Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------------
# 2. LOAD MODEL SBERT & DATASET MEDIS
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    medical_qa_data = [
        {"category": "Diabetes", "question": "apa itu diabetes", "answer": "🩺 Diabetes melitus adalah kondisi kronis di mana tubuh tidak dapat memproduksi atau menggunakan insulin dengan baik."},
        {"category": "Diabetes", "question": "gejala diabetes mellitus gula darah tinggi", "answer": "🩺 Gejala umum diabetes: Sering buang air kecil, rasa haus berlebihan, cepat lapar, penurunan BB tanpa sebab."},
        {"category": "Diabetes", "question": "cara mencegah diabetes", "answer": "🛡️ Pertahankan berat badan ideal, olahraga rutin, kurangi gula, dan berhenti merokok."},
        {"category": "Hipertensi", "question": "apa itu tekanan darah tinggi hipertensi", "answer": "🩺 Hipertensi adalah kondisi tekanan darah ≥140/90 mmHg secara konsisten."},
        {"category": "Hipertensi", "question": "gejala tekanan darah tinggi hipertensi sakit kepala pusing", "answer": "🩺 Gejala hipertensi: Sakit kepala belakang, pusing, penglihatan kabur, nyeri dada."},
        {"category": "Sakit Kepala", "question": "penyebab kepala pusing berputar vertigo", "answer": "🩺 Vertigo ditandai sensasi berputar. Umumnya disebabkan gangguan telinga dalam (BPPV)."},
        {"category": "Sakit Kepala", "question": "migrain penyebab dan pengobatan", "answer": "🩺 Migrain adalah sakit kepala berdenyut sebelah sisi. Redakan dengan istirahat di ruangan gelap."},
        {"category": "Asma", "question": "gejala serangan asma sesak napas mengi", "answer": "🫁 Gejala: Mengi (suara ngik-ngik), sesak napas, batuk, dan dada terasa berat."},
        {"category": "Jantung", "question": "gejala serangan jantung nyeri dada", "answer": "🚨 DARURAT! Nyeri dada seperti ditekan >20 menit, menjalar ke lengan kiri. Segera hubungi 119!"},
        {"category": "Maag", "question": "gejala maag gastritis sakit lambung nyeri ulu hati", "answer": "🩺 Nyeri/perih di ulu hati, mual, muntah, dan kembung."},
        {"category": "Demam", "question": "demam panas badan tinggi", "answer": "🌡️ Suhu tubuh ≥38°C. Kompres hangat, minum banyak air, dan minum Paracetamol."},
        {"category": "Pertolongan Pertama", "question": "cara menangani luka bakar", "answer": "🚨 Siram air mengalir dingin selama 20 menit. JANGAN oleskan pasta gigi!"},
        {"category": "Kesehatan Mental", "question": "cara mengatasi stres kecemasan anxiety", "answer": "🧠 Lakukan teknik pernapasan 4-7-8, meditasi, kurangi kafein, dan cerita ke orang terpercaya."}
    ]
    
    df = pd.DataFrame(medical_qa_data)
    embeddings = model.encode(df['question'].tolist(), convert_to_tensor=True)
    return model, df, embeddings

model, df, question_embeddings = load_model_and_data()

# ---------------------------------------------------------
# 3. SIDEBAR DASHBOARD (STATISTIK & CONTROL)
# ---------------------------------------------------------
st.sidebar.title("📊 Medical AI Dashboard")
st.sidebar.markdown("---")

# Status Sistem
st.sidebar.subheader("⚙️ System Status")
st.sidebar.success("Model: SBERT Multilingual")
st.sidebar.info(f"Total Database Q&A: {len(df)} Pasangan")

# Slider Thresholding
threshold = st.sidebar.slider(
    "Confidence Threshold:", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.35, 
    step=0.05,
    help="Batas minimum skor kemiripan semantik untuk memberikan jawaban."
)

# Grafik Distribusi Kategori
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Distribusi Kategori QA")
category_counts = df['category'].value_counts().reset_index()
category_counts.columns = ['Kategori', 'Jumlah']

fig = px.pie(
    category_counts, 
    values='Jumlah', 
    names='Kategori', 
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
st.sidebar.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 4. UTAMA (CHATBOT INTERFACE)
# ---------------------------------------------------------
st.title("🩺 Assistant Medical Chatbot")
st.write("Silakan ajukan pertanyaan seputar gejala atau informasi kesehatan dasar di bawah ini.")

# Session State untuk Riwayat Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya asisten medis AI. Ada yang bisa saya bantu terkait informasi kesehatan Anda?"}
    ]

# Tampilkan Pesan Sebelumnya
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Menerima Pesan Pengguna
if user_query := st.chat_input("Ketikkan pertanyaan medis Anda di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Proses SBERT Cosine Similarity
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    best_idx = int(np.argmax(scores.cpu()))
    best_score = float(scores[best_idx])

    # Logika Penentuan Jawaban
    if best_score < threshold:
        bot_response = "🤖 Maaf, saya belum memahami pertanyaan medis tersebut. Silakan hubungi layanan darurat **119** atau berkonsultasi langsung dengan dokter."
        matched_cat = "N/A"
    else:
        bot_response = df.iloc[best_idx]['answer']
        matched_cat = df.iloc[best_idx]['category']

    # Output Balasan Chatbot
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"🎯 **Similarity Score:** `{best_score:.4f}`")
        with col2:
            st.caption(f"🏷️ **Kategori:** `{matched_cat}`")

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
