from ocr import extract_text
from similarity import get_similarity
from utils import clean_text

# Input paths
img1 = "sample_inputs/assignment1.png"
img2 = "sample_inputs/assignment2.png"

# OCR Extraction
print("📄 Extracting text from image 1...")
text1 = extract_text(img1)
print("✅ Done!\n")

print("📄 Extracting text from image 2...")
text2 = extract_text(img2)
print("✅ Done!\n")

# Clean Text
text1_clean = clean_text(text1)
text2_clean = clean_text(text2)

# Similarity Score
print("🔍 Calculating similarity...")
score = get_similarity(text1_clean, text2_clean)
print(f"\n📊 Similarity Score: {score:.2f} (0 = different, 1 = identical)")
