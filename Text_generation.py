from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
prompt = input("Enter a prompt: ")

result = generator(
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50
)

generated_text = result[0]["generated_text"]

print("\n-----Generated Text-----")
print(generated_text)

# Installed transformers and pytorch "pip install transformers torch"

# mention the library used :
# library : Hugging Face Transformers, 1st I import the transformers library from hugging face
# then I used Hugging Face transformers pipeline to load GPT-2 for text generation.

