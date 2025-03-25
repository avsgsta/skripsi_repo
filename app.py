import streamlit as st
import time
import pandas as pd
import torch
import pickle
import re
import os
import gdown
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import AutoTokenizer
import plotly.express as px
from urllib.parse import urlparse, urlunparse

st.set_page_config(page_title="Detection Fake Review Tokopedia", page_icon="🛒", layout="wide")

# 🔗 Ganti dengan ID file Google Drive kamu
FILE_ID = "15NHe3e95pLEmFEATvwbRK6_7dYcz-Vrv"
MODEL_PATH = "model_optimized.pkl"
MODEL_TOKENIZER = "cahya/bert-base-indonesian-522M"

# 🔄 Download Model jika belum ada
if not os.path.exists(MODEL_PATH):
    print("🚀 Mengunduh model dari Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# ✅ Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

model.to(device)
model.eval()

def predict_review_label(review, image_url=None):
    if is_emoji_only(review):
        return "Fake"
    if image_url:
        return "Real"
    
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "Fake" if prediction == 1 else "Real"

def is_emoji_only(text):
    emoji_pattern = re.compile(r"^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]+$") 
    return bool(emoji_pattern.match(text))

def clean_url(url):
    """Menghapus parameter tracking dari URL (seperti ?utm_source=...)"""
    parsed_url = urlparse(url)
    cleaned_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "", ""))
    return cleaned_url

def format_review_url(url):
    """Menambahkan '/review' jika belum ada dalam URL."""
    url = clean_url(url)  # 🔥 Bersihkan URL terlebih dahulu
    if "/review" not in url:
        return url.rstrip("/") + "/review"
    return url

st.title("🛍️ Detection Fake Review Tokopedia 🤖")

url_input = st.text_input("🔗 Masukkan URL produk Tokopedia:")
start_button = st.button("🚀 Mulai Deteksi")

if start_button and url_input:
    formatted_url = format_review_url(url_input)  # ✅ Format URL sebelum scraping
    st.write(f"🔗 **Mengakses Halaman:** [{formatted_url}]({formatted_url})")
    st.write("⏳ **Sedang memproses, harap tunggu...**")
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0")

    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(formatted_url)
    time.sleep(3)

    data = []
    max_reviews = 300
    reviews_scraped = 0
    page_number = 1

    def scroll_down():
        """Scroll ke bawah untuk memuat lebih banyak konten sebelum parsing."""
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

    while reviews_scraped < max_reviews:
        status_placeholder.markdown(f"**📄 Mengambil halaman {page_number}...**")
        
        scroll_down()  # ⬇️ Scroll ke bawah sebelum mengambil data
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.findAll('div', attrs={'class': 'css-1k41fl7'})

        if not containers:
            break  

        for container in containers:
            if reviews_scraped >= max_reviews:
                break
            try:
                user = container.find('span', class_='name').text
                review = container.find('span', attrs={'data-testid': 'lblItemUlasan'}).text
                rating_stars = container.find_all('svg', attrs={'fill': 'var(--YN300, #FFD45F)'} )
                rating = len(rating_stars)
                image_tag = container.find('img', attrs={'data-testid': 'imgItemPhotoulasan'})
                image_url = image_tag['src'] if image_tag else None

                label = predict_review_label(review, image_url)
                data.append((user, review, rating, label, image_url))
                reviews_scraped += 1
            except AttributeError:
                continue

        try:
            next_button = WebDriverWait(driver, 7).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))
            )
            if next_button.is_enabled():
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(5)
                page_number += 1
            else:
                break
        except Exception:
            break  

        progress_bar.progress(reviews_scraped / max_reviews)

    driver.quit()
    progress_bar.empty()

    if data:
        df = pd.DataFrame(data, columns=["Nama User", "Ulasan", "Rating Bintang", "Kategori", "Gambar Ulasan"])
        st.session_state["scraped_data"] = df
        st.session_state["scraping_done"] = True
    else:
        st.error("❌ Tidak ada ulasan yang ditemukan.")

if st.session_state.get("scraping_done"):
    df = st.session_state["scraped_data"]
    real_count = df[df["Kategori"] == "Real"].shape[0]
    fake_count = df[df["Kategori"] == "Fake"].shape[0]
    total_reviews = real_count + fake_count
    real_percentage = (real_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    fig = px.pie(
        names=["Real", "Fake"],
        values=[real_count, fake_count],
        title="Distribusi Ulasan Real vs Fake",
        color_discrete_sequence=["blue", "red"]
    )
    
    st.subheader("📊 Ringkasan Analisis")
    st.plotly_chart(fig)
    st.dataframe(df)
    
    if real_percentage >= 70:
        st.success(f"✅ Produk ini **layak dibeli** (Real Reviews: {real_percentage:.2f}%)")
    elif 50 <= real_percentage < 70:
        st.warning(f"⚠️ Produk ini **perlu dipertimbangkan** (Real Reviews: {real_percentage:.2f}%)")
    else:
        st.error(f"❌ Produk ini **tidak layak dibeli** (Real Reviews: {real_percentage:.2f}%)")
    
    csv_file = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Unduh CSV", data=csv_file, file_name="tokopedia_reviews.csv", mime="text/csv")
