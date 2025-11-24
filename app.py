import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# ==========================================
# 1. KONFIGURASI TEMA KULINER 🌶️
# ==========================================
st.set_page_config(
    page_title="Nusantara Kitchen AI",
    page_icon="🥘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Kustom (Tema Warung/Tradisional)
st.markdown("""
<style>
    /* Latar Belakang Header */
    .main-header {
        font-family: 'Georgia', serif;
        font-size: 40px;
        font-weight: 700;
        color: #d35400; /* Warna Oranye Bata */
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px #eee;
    }
    .sub-header {
        font-size: 18px;
        color: #8d6e63; /* Coklat Kayu */
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
    }
    
    /* Desain Kartu Resep */
    .recipe-card {
        background-color: #fff8e1; /* Warna Krem Kuning */
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #ff6f00; /* Garis Oranye Terang */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .recipe-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Teks Resep (Dipaksa Hitam/Gelap) */
    .recipe-card p {
        color: #4e342e !important; /* Coklat Tua Gelap */
        font-size: 16px;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Badge Skor */
    .score-badge {
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
        margin-bottom: 10px;
        color: white;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA & MODEL (DAPUR AI)
# ==========================================
@st.cache_resource
def load_data():
    try:
        # MEMBACA FILE BERSIH
        # Pastikan nama file di GitHub Anda adalah 'resep_clean.csv'
        df = pd.read_csv('resep_clean.csv')
        
        # Pastikan kolom clean_text terisi
        if 'clean_text' not in df.columns:
             # Fallback jika user lupa cleaning: pakai full_text sementara
            df['clean_text'] = df['full_text'].astype(str).str.lower()
            
        df['clean_text'] = df['clean_text'].fillna('')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def build_model(df):
    # Stopwords Khusus Masakan (PENTING!)
    # Kita membuang kata kerja umum supaya mesin fokus ke BAHAN (Ayam, Santan, Kunyit, dll)
    stopwords_resep = [
        'dan', 'yang', 'di', 'itu', 'ini', 'ke', 'dari', 'ada', 'buat', 'yg', 'mau',
        'ga', 'gak', 'aku', 'sama', 'kalo', 'lagi', 'bisa', 'karena', 'jadi', 'apa',
        'tapi', 'suka', 'udah', 'banget', 'ya', 'dia', 'kita', 'untuk', 'dengan',
        'pada', 'atau', 'adalah', 'saya', 'mereka', 'kan', 'juga', 'aja', 'kalo',
        'kalau', 'langsung', 'banyak', 'tp', 'dr', 'bgt', 'sdh', 'udh', 'nih', 'sih',
        'kok', 'deh', 'masih', 'biar', 'tetap', 'pun', 'doang', 'nya',
        # Kata kerja masak umum
        'resep', 'cara', 'membuat', 'bikin', 'masak', 'enak', 'lezat', 'mantap', 'praktis',
        'simple', 'mudah', 'ala', 'khas', 'menu', 'makan', 'siang', 'malam', 'pagi',
        'video', 'tutorial', 'bumbu', 'dapur', 'sendiri', 'mari', 'yuk', 'cobain'
    ]
    
    tfidf = TfidfVectorizer(stop_words=stopwords_resep)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

# Load Data Awal
df = load_data()

# ==========================================
# 3. SIDEBAR (PROFIL CHEF)
# ==========================================
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>👨‍🍳 Chef AI</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><i>Asisten Masak Pribadimu</i></div>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3448/3448099.png", width=120)
    
    st.info(
        """
        **Bingung mau masak apa?**
        
        Ketik bahan yang tersisa di kulkasmu (misal: *Ayam, Santan, Pedas*), 
        dan biarkan Chef AI mencarikan resep tradisional yang cocok!
        """
    )
    st.write("---")
    st.write("🌶️ **Kategori:** Kuliner Nusantara")
    st.write("🤖 **Metode:** Content-Based Filtering")
    st.caption("Skripsi Mahasiswa SI - Amikom")

# ==========================================
# 4. HALAMAN UTAMA (DASHBOARD)
# ==========================================

st.markdown('<div class="main-header">🍲 Nusantara Kitchen AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Temukan Cita Rasa Warisan Leluhur Berdasarkan Bahanmu</div>', unsafe_allow_html=True)

if df is None:
    st.error("⚠️ FILE DATA HILANG! Pastikan Anda sudah upload file 'resep_clean.csv' ke GitHub.")
    st.stop()

# Bangun Model
df, cosine_sim = build_model(df)

# Kolom Pencarian
col1, col2 = st.columns([3, 1])
with col1:
    keyword = st.text_input("", placeholder="Punya bahan apa? (Contoh: Daging sapi, rendang, kuah santan)", label_visibility="collapsed")
with col2:
    cari_btn = st.button("Carikan Resep", type="primary", use_container_width=True)

# ==========================================
# 5. LOGIKA PENCARIAN & HASIL
# ==========================================
if cari_btn and keyword:
    keyword_lower = keyword.lower()
    
    with st.spinner('🔥 Mengulek bumbu & mencari resep...'):
        time.sleep(0.8) # Efek loading
        
        # Cari di data (di kolom clean_text)
        hasil = df[df['clean_text'].str.contains(keyword_lower)]
        
        if len(hasil) == 0:
            st.warning(f"Yah, Chef tidak menemukan resep dengan bahan **'{keyword}'**. Coba bahan lain ya!")
            st.info("💡 Tips: Coba kata kunci bahan dasar seperti 'ayam', 'tahu', 'tempe', 'sapi', 'ikan'.")
        else:
            # Ambil Patokan
            idx = hasil.index[0]
            tweet_patokan = df.iloc[idx]['full_text']
            
            st.success(f"Hore! Menemukan {len(hasil)} resep yang cocok.")
            
            # Tampilkan Patokan
            with st.expander("Lihat Resep Dasar (Basis Pencarian)", expanded=True):
                st.write(f"**Resep Terpilih:** \"{tweet_patokan}\"")

            # Hitung Skor
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_5 = sim_scores[1:6]

            st.markdown("### 🍛 5 Rekomendasi Menu Serupa:")
            
            for i, skor in top_5:
                resep_hasil = df.iloc[i]['full_text']
                persen = skor * 100
                
                # Logika Warna Badge
                if persen > 30:
                    bg_color = "#2e7d32" # Hijau Tua
                    label = "Sangat Pas"
                elif persen > 15:
                    bg_color = "#f57c00" # Oranye
                    label = "Mirip"
                else:
                    bg_color = "#c62828" # Merah
                    label = "Variasi Unik"

                # Tampilkan Kartu
                st.markdown(f"""
                <div class="recipe-card">
                    <span class="score-badge" style="background-color: {bg_color};">
                        {label} ({persen:.1f}%)
                    </span>
                    <p>"{resep_hasil}"</p>
                </div>
                """, unsafe_allow_html=True)

st.write("")
st.markdown("<center style='color: #888; font-size: 12px; margin-top: 50px;'>Dibuat dengan ❤️ & 🌶️ untuk Skripsi</center>", unsafe_allow_html=True)
