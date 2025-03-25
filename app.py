import streamlit as st
import time
import pandas as pd
import torch
import pickle
import re
import os
import requests
import undetected_chromedriver as uc
from io import BytesIO
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import AutoTokenizer
import plotly.express as px
from urllib.parse import urlparse, urlunparse
import asyncio  # Hindari event loop error di Python 3.12+

st.set_page_config(page_title="Detection Fake Review Tokopedia", page_icon="🛒", layout="wide")

# 🔗 Google Drive File ID untuk Model
FILE_ID = "15NHe3e95pLEmFEATvwbRK6_7dYcz-Vrv"
MODEL_TOKENIZER = "cahya/bert-base-indonesian-522M"

# 🔄 Load model langsung dari Google Drive tanpa menyimpan ke disk
@st.cache_resource
def load_model_from_drive():
    GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    response = requests.get(GDRIVE_URL, stream=True)
    response.raise_for_status()
    model_data = BytesIO(response.content)
    return pickle.load(model_data)

# ✅ Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)

try:
    model = load_model_from_drive()
    model.to(device)
    model.eval()
    st.success("✅ Model berhasil dimuat dari Google Drive!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")

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

    options = uc.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0")

    # ✅ Gunakan undetected_chromedriver untuk menghindari deteksi bot
    try:
        driver = uc.Chrome(options=options)
        driver.get(formatted_url)
        time.sleep(3)
    except Exception as e:
        st.error(f"❌ Gagal menjalankan Selenium: {e}")
        driver.quit()

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
                rating_stars = container.find_all('svg', attrs={'fill': 'var(--YN300, #FFD45F)'})
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
    st.subheader("📊 Ringkasan Analisis")
    st.dataframe(df)
    
    csv_file = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Unduh CSV", data=csv_file, file_name="tokopedia_reviews.csv", mime="text/csv")
