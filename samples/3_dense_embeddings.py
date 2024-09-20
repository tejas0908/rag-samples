from sentence_transformers import SentenceTransformer
text = ["the students opened their textbooks", "each student has five textbooks"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(text)
print(embeddings.shape)
print(embeddings)
