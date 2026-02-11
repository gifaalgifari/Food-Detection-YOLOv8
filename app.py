import streamlit as st
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import gdown

# =========================
# 1. Load YOLOv8 Model
# =========================
model_path = "best.pt"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1uuPFXOXnI-tcjTPAshlRKXA0s0y09eDs"
    gdown.download(url, model_path, quiet=False)
model = YOLO(model_path)

# =========================
# 2. Page Config
# =========================
st.set_page_config(page_title="Deteksi Makanan Bergizi", page_icon="", layout="centered")

st.title(" Food Detection & Nutrition Info")
st.markdown("""
Semua informasi nutrisi diambil dari  [**Fatsecret Indonesia**](https://www.fatsecret.co.id/kalori-gizi/search), dan perhitungan berdasarkan **per porsi standar**.
""")


# =========================
# 3. Load Nutrition CSV
# =========================
@st.cache_data
def load_nutrition_data():
    df = pd.read_csv("nutrition_data.csv")
    df.columns = df.columns.str.strip()
    return df

nutrition_df = load_nutrition_data()

# =========================
# 4. Penjelasan Makanan
# =========================
food_description = {
    "Ayam Goreng": """
**Ayam Goreng** merupakan sumber protein yang membantu pembentukan otot dan menjaga kekebalan tubuh. 
Kandungan lemaknya memberikan energi tambahan, namun konsumsi berlebihan dapat meningkatkan kolesterol. 
Porsi ideal: **1 potong (80–100 g)**.
""",
    "Nasi Putih": """
**Nasi Putih** adalah sumber karbohidrat utama untuk energi cepat. 
Cocok sebagai makanan pokok, namun konsumsi berlebihan dapat meningkatkan gula darah. 
Porsi ideal: **1 centong (100–150 g)**.
""",
    "Nasi Goreng": """
**Nasi Goreng** mengandung karbohidrat, lemak, dan sedikit protein. 
Karbohidratnya memberi energi, sementara minyak menambah kalori harian.
Porsi ideal: **1 piring (200–250 g)**.
""",
    "Kentang Goreng": """
**Kentang Goreng** kaya karbohidrat dan memberikan energi cepat, namun tinggi lemak karena proses penggorengan. 
Ideal dikonsumsi sesekali. Porsi ideal: **100 g**.
""",
    "Rendang": """
**Rendang** kaya protein dan zat besi yang baik untuk pembentukan sel darah merah. 
Namun kandungan lemaknya cukup tinggi sehingga konsumsi perlu dibatasi. 
Porsi ideal: **1 potong (80 g)**.
""",
    "Telur": """
**Telur** mengandung protein berkualitas tinggi, vitamin D, dan kolin yang baik untuk otak. 
Dapat dikonsumsi harian. Porsi ideal: **1–2 butir per hari**.
""",
    "Ikan": """
**Ikan** kaya protein dan omega-3 yang baik untuk jantung dan otak. 
Sumber nutrisi rendah lemak dan direkomendasikan untuk konsumsi rutin. 
Porsi ideal: **100 g per porsi**.
""",
    "Bakso": """
**Bakso** memiliki protein cukup tinggi namun bisa mengandung natrium tinggi. 
Baik dikonsumsi dalam jumlah wajar. Porsi ideal: **5 butir sedang**.
""",
    "Bubur Ayam": """
**Bubur Ayam** menyediakan karbohidrat dan sedikit protein. 
Ringan di pencernaan namun bisa tinggi natrium jika memakai banyak kecap/kaldu. 
Porsi ideal: **1 mangkuk (250 g)**.
""",
    "Sate": """
**Sate** kaya protein hewani dan vitamin B12. 
Namun saus kacang menambah kalori cukup banyak. 
Porsi ideal: **5 tusuk + sedikit bumbu**.
""",
    "Steik Daging": """
**Steak Daging** tinggi protein dan zat besi, baik untuk pembentukan otot. 
Batasi konsumsi bagian berlemak. Porsi ideal: **100–150 g**.
""",
    "Ayam Pop": """
**Ayam Pop** adalah ayam rendah minyak, tetap memberi protein baik dengan lemak yang lebih rendah dari ayam goreng. 
Porsi ideal: **1 potong (80–100 g)**.
""",
    "Tumis Kangkung": """
**Tumis Kangkung** kaya serat, zat besi, dan vitamin. 
Baik untuk pencernaan dan kesehatan darah. 
Porsi ideal: **1 piring sayur (100–150 g)**.
""",
    "Salad Sayuran": """
**Salad Sayuran** sangat rendah kalori namun tinggi vitamin, mineral, dan serat. 
Ideal untuk diet dan kesehatan pencernaan. 
Porsi ideal: **1 mangkuk (120–150 g)**.
"""
}

# =========================
# 5. Upload Gambar
# =========================
uploaded_file = st.file_uploader("Unggah gambar makanan", type=["jpg", "jpeg", "png"])

# =========================
# 6. Detection Process
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    st.write("Sedang mendeteksi makanan...")
    results = model.predict(image, conf=0.5)

    result_img = results[0].plot()
    st.image(result_img, caption="Hasil Deteksi YOLOv8", use_container_width=True)

    print(results[0].boxes)
    print(results[0].names)

    detected_labels = list(set([results[0].names[int(box.cls)] for box in results[0].boxes]))
    
    
    if not detected_labels :
        st.subheader("No Detection (Tidak ada dalam kelas) ")
    else :
        st.subheader("Jenis Makanan Terdeteksi:")
        for label in detected_labels:
            st.markdown(f"- **{label}**")

        # =========================
        # 7. Nutrition Cards
        # =========================
        st.markdown(
                """
                <div style='background:#f8f9fa; padding:5px; border-radius:12px;  margin-top:20px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
                """, 
                unsafe_allow_html=True
            )
        st.subheader("Informasi Gizi & Penjelasan:")

        for label in detected_labels:
            data_row = nutrition_df[nutrition_df['food_name'].str.lower() == label.lower()]

            
            st.divider()

            st.markdown(f"### Nama Makanan: {label}")

            # Info Gizi
            if not data_row.empty:
                row = data_row.iloc[0]

                st.markdown(f"""
                    <div style='display:flex; flex-direction:column; gap:6px; margin-left:10px;'>
                        <div style='display:flex; justify-content:space-between; width:260px;'>
                            <span><b>Kalori</b></span><span>{row['calories']}</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; width:260px;'>
                            <span><b>Protein</b></span><span>{row['protein']}</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; width:260px;'>
                            <span><b>Lemak</b></span><span>{row['fat']}</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; width:260px;'>
                            <span><b>Karbohidrat</b></span><span>{row['carbs']}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Informasi gizi belum tersedia di file CSV.")

            # Penjelasan
            st.markdown("#### Penjelasan:")
            st.markdown(food_description.get(label, "Deskripsi belum tersedia."))

            st.markdown("</div>", unsafe_allow_html=True)
            

else:
    st.info("Silakan unggah gambar makanan untuk mulai mendeteksi.")

st.markdown("---")
st.caption("OBJECT DETECTION YOLOv8 DENGAN SEMI-SUPERVISED LEARNING")
