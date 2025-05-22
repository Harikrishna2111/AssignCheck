import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer, util
import tempfile
import os
import torch

st.set_page_config(page_title="OCR Similarity Checker", layout="centered")

st.title("🧠 OCR Text Similarity App")
st.write("Upload two handwritten assignments (PDFs or images), and we'll compare their similarity using OCR and NLP magic!")

# Load models only once
@st.cache_resource
def load_models():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return processor, trocr, embedder

processor, trocr, embedder = load_models()

def convert_pdf_to_images(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name
    return convert_from_path(tmp_path)

def extract_text_from_image(image):
    image = image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = trocr.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def get_text(file):
    if file.type == "application/pdf":
        images = convert_pdf_to_images(file)
        text = ""
        for img in images:
            text += extract_text_from_image(img) + " "
        return text.strip()
    else:
        image = Image.open(file)
        return extract_text_from_image(image)

def compute_similarity(text1, text2):
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

# Upload UI
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload File 1", type=["pdf", "png", "jpg", "jpeg"])
with col2:
    file2 = st.file_uploader("Upload File 2", type=["pdf", "png", "jpg", "jpeg"])

if file1 and file2:
    with st.spinner("🔍 Extracting and comparing..."):
        text1 = get_text(file1)
        text2 = get_text(file2)
        similarity = compute_similarity(text1, text2)

    st.subheader("📝 Extracted Texts:")
    with st.expander("File 1 Text"):
        st.write(text1)
    with st.expander("File 2 Text"):
        st.write(text2)

    st.subheader("📊 Similarity Score:")
    st.metric("Cosine Similarity", f"{similarity:.2f}")

    if similarity > 0.9:
        st.warning("⚠️ Very high similarity — might be a copy?")
    elif similarity > 0.7:
        st.info("🧐 Moderate similarity — maybe some common parts.")
    else:
        st.success("✅ Low similarity — looks original!")

