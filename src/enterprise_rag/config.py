import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_HUGGINGFACE_API_KEY = os.environ.get("CHROMA_HUGGINGFACE_API_KEY")