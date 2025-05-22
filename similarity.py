from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def get_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)
    return similarity.item()
