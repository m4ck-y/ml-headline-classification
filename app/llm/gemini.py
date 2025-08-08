from google import genai

client = genai.Client(api_key="AIzaSyAYNp3GDcwcR0HOidqRnBcFfBdeAD-s8js")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)

response = client.models.embed_content(
    model='text-embedding-004',
    contents='why is the sky blue?',
)
print(response)


# pip install chromadb google-generativeai #TODO: DOCS