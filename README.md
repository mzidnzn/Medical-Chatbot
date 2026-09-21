# 🩺 Medical Assistant NLP Chatbot (BERT & SBERT)

Aplikasi Chatbot Medis berbasis **Natural Language Processing (NLP)** yang dirancang untuk memberikan edukasi kesehatan dasar, panduan pertolongan pertama, serta triase awal secara cepat dan akurat.

---

## 📌 Fitur Utama

* **Semantic Search dengan SBERT:** Menggunakan arsitektur `paraphrase-multilingual-MiniLM-L12-v2` untuk memahami konteks dan *intent* pertanyaan medis pengguna dalam Bahasa Indonesia.
* **Interactive Streamlit Dashboard:** Antarmuka percakapan interaktif yang responsif dilengkapi statistik dataset dan kontrol sensitivitas (*threshold*).
* **Out-of-Scope Protection:** Dilengkapi dengan *Confidence Threshold* untuk mencegah jawaban sembarangan (*hallucination*) pada pertanyaan di luar domain medis.
* **Metadata Analytics:** Menampilkan skor kemiripan semantik (*Similarity Score*) dan kategori penyakit terkait pada setiap jawaban.

---

## 🛠️ Tech Stack & Library

* **Python 3.x**
* **Sentence-Transformers (SBERT)**
* **Streamlit** (UI Dashboard & Deployment)
* **NLTK & Scikit-Learn** (Text Preprocessing)
* **Pandas & NumPy** (Data Manipulation)
* **Plotly Express** (Visualisasi Data)

---

## 🔄 8 Tahap Pipeline NLP

1. **Text Normalization / Lowercasing**
2. **Noise Removal (Regex Filtering)**
3. **Tokenizing**
4. **Stopwords Removal** (Bahasa Indonesia & Inggris)
5. **Lemmatization**
6. **Vectorization** (SBERT Dense Embeddings)
7. **Similarity Calculation** (Cosine Similarity)
8. **Response Retrieval & Thresholding**

---

## 📂 Struktur Repository

```text
├── medical_chatbot.ipynb   # Notebook utama (Eksperimen 8 Tahap NLP & Quiz)
├── app.py                  # Script aplikasi Streamlit Dashboard
├── requirements.txt        # Daftar pustaka/library Python
└── README.md               # Dokumentasi proyek
