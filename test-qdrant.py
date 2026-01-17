import os
from dotenv import load_dotenv
from pathlib import Path
from google import genai

# Load .env explicitly
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Use a model that is confirmed available
response = client.models.generate_content(
    model="models/gemini-flash-latest",
    contents="Explain Ohm's Law in one sentence."
)

print("\n🤖 Gemini says:\n")
print(response.text)
