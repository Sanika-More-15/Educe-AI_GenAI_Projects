from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

with open("rag doc.txt","r", encoding="utf-8") as file:
    documents = file.read()

chunks = [chunk.strip() for chunk in documents.split("\n\n") if chunk.strip()]  

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(chunk)

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(chunks)

query = input("\nAsk a question:")

query_embedding = model.encode([query])

similarities = cosine_similarity(
    query_embedding,
    embeddings
)

best_score = -1
best_chunk = ""

for i, score in enumerate(similarities[0]):
    if score > best_score:
        best_score = score
        best_chunk = chunks[i]

print("\n--- Answer ---")
print(best_chunk)

