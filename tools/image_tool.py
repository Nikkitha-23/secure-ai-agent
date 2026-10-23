"""
tools/image_tool.py — Image Analysis Tool
------------------------------------------
Capabilities:
  1. Vision Analysis  → Groq LLaVA (what's in the image?)
  2. OCR              → Groq LLaVA (extract text from image)
  3. Diagram Analysis → Groq LLaVA (explain charts/flowcharts)

Flow:
  User sends image (base64 or file path)
       ↓
  is_image_intent() detects
       ↓
  image → base64 encode
       ↓
  Groq Vision API (llava-v1.5-7b-4096-preview)
       ↓
  Answer
"""

import os
import re
import base64
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Intent Detection ──────────────────────────────────────────────────────────

IMAGE_KEYWORDS = {
    # General vision
    "what is in this image", "what's in this image", "describe this image",
    "analyze this image", "what do you see", "look at this",
    # OCR
    "read this image", "extract text", "what text", "ocr", "read the text",
    "text in this image", "words in this image",
    # Diagram
    "explain this diagram", "explain this chart", "explain this graph",
    "what does this diagram", "analyze this chart", "read this chart",
    "explain this flowchart", "what is this flowchart",
    # General
    "image", "photo", "picture", "screenshot",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def is_image_intent(query: str, has_image: bool = False) -> bool:
    """
    Return True if:
    - User uploaded an image (has_image=True), OR
    - Query mentions image-related keywords
    """
    if has_image:
        return True
    q = query.lower()
    for kw in IMAGE_KEYWORDS:
        if kw in q:
            logger.debug(f"🖼️ Image keyword match: '{kw}'")
            return True
    return False


# ── Image Loader ──────────────────────────────────────────────────────────────

def load_image_as_base64(image_input) -> tuple[str, str]:
    """
    Accepts:
      - File path (str)
      - Raw bytes
      - Already base64 string

    Returns: (base64_string, media_type)
    """
    # Already base64
    if isinstance(image_input, str) and not os.path.exists(image_input):
        # Assume it's already base64
        media_type = "image/jpeg"
        return image_input, media_type

    # File path
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        ext = path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp"
        }
        media_type = media_type_map.get(ext, "image/jpeg")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64, media_type

    # Raw bytes
    if isinstance(image_input, bytes):
        b64 = base64.b64encode(image_input).decode("utf-8")
        return b64, "image/jpeg"

    raise ValueError(f"Unsupported image input type: {type(image_input)}")


# ── Sub-tool: Groq Vision ─────────────────────────────────────────────────────

def analyze_with_groq_vision(image_input, query: str) -> str:
    """
    Send image + query to Groq Vision API.
    Uses llava-v1.5-7b-4096-preview model.
    """
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Load image
        b64_image, media_type = load_image_as_base64(image_input)

        # Build prompt based on query type
        q = query.lower()

        if any(w in q for w in ["text", "read", "ocr", "extract", "words"]):
            system_prompt = "You are an OCR expert. Extract ALL text visible in the image exactly as it appears. Preserve formatting where possible."
        elif any(w in q for w in ["diagram", "flowchart", "chart", "graph", "architecture"]):
            system_prompt = "You are a diagram analysis expert. Explain the structure, components, and flow shown in the image clearly and concisely."
        else:
            system_prompt = "You are a vision analysis expert. Describe and analyze the image thoroughly, noting key elements, colors, text, and context."

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": query if query else "Describe this image in detail."
                        }
                    ]
                }
            ],
            max_tokens=1024,
            temperature=0
        )

        result = response.choices[0].message.content.strip()
        logger.info(f"🖼️ Vision result: {result[:80]}...")
        return result

    except Exception as e:
        logger.error(f"❌ Groq Vision error: {e}")
        return f"Image analysis failed: {e}"


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_image_tool(query: str, image_input=None) -> str:
    """
    Main entry point called from main.py / agent_loop.py

    Args:
        query      : User's question about the image
        image_input: File path, bytes, or base64 string

    Returns:
        Analysis result as string
    """
    if image_input is None:
        return "No image provided. Please upload an image to analyze."

    logger.info(f"🖼️ Image Tool triggered | query: {query[:60]}")
    return analyze_with_groq_vision(image_input, query)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🖼️ Image Tool — Intent Detection Test:\n")

    test_queries = [
        ("what is in this image?", False),
        ("extract text from this screenshot", False),
        ("explain this flowchart", False),
        ("what is RAG?", False),
        ("describe the photo", False),
        ("mean of 4 8 15", False),
    ]

    for query, has_img in test_queries:
        result = is_image_intent(query, has_img)
        icon = "✅" if result else "❌"
        print(f"{icon} is_image_intent('{query}') → {result}")