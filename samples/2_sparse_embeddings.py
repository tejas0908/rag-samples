from sklearn.feature_extraction.text import CountVectorizer

# Training the model
text = ["the students opened their textbooks", "each student has five textbooks"]
vectorizer = CountVectorizer()
vectorizer.fit(text)

# Show Vocabulary
print(sorted(vectorizer.vocabulary_))

print('\n')

# Calculate Embeddings
vector = vectorizer.transform(text)
print(vector.toarray())
