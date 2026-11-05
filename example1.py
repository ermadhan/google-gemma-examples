import os
from google import genai

# Initialize the client. It will automatically look for the GEMINI_API_KEY environment variable.
client = genai.Client()

# Generate text using a Gemma model (e.g., Gemma 3 or Gemma 4)
response = client.models.generate_content(
    model='gemini-3.5-flash-lite',
    contents='Write a quick Python script to reverse a string.',
)

print(response.text)
