import os
import sys
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageEnhance, ImageOps
from dotenv import load_dotenv
from google import genai
from google.genai import types
from io import BytesIO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to your .env file."
    )

client = genai.Client(api_key=API_KEY)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Instructions for historical Tamil OCR
# --------------------------------------------------


OCR_PROMPT = """
You are an expert in reading historical handwritten Tamil documents,
including old Tamil legal deeds, property records, and sale agreements.

TASK:
Transcribe the entire document faithfully in Tamil.

STRICT RULES:
1. Read the entire page from top to bottom and right to left within
   each line, following normal Tamil writing direction.
2. Preserve the original Tamil wording, spelling, names, dates,
   amounts, addresses, and legal terminology.
3. Do not translate, summarize, modernize, or rewrite the document.
4. Do not omit headings, stamps, signatures, marginal notes, or
   text near the edges.
5. Preserve the original line breaks and paragraph boundaries
   as closely as possible.
6. Keep document headings, dates, names, and monetary amounts
   on separate lines when they appear that way in the image.
7. Never invent text to complete an unclear sentence.
8. Use [தெளிவில்லை] only for text that genuinely cannot be read.
9. If a word is partially readable, transcribe the readable portion
   and mark only the unclear portion.
10. Preserve the distinction between numerals, dates, and currency.
11. Do not add explanations, translations, markdown, or commentary.

Return only the complete Tamil transcription, preserving the
document's layout as closely as possible.
"""



# --------------------------------------------------
# Prepare an image for OCR
# --------------------------------------------------

def prepare_image(image: Image.Image) -> bytes:
    """Improve readability without aggressively altering the page."""

    image = ImageOps.exif_transpose(image).convert("RGB")

    # Upscale small scans to improve character readability.
    max_dimension = max(image.size)

    if max_dimension < 1800:
        scale = 1800 / max_dimension
        new_size = (
            int(image.width * scale),
            int(image.height * scale),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Mild contrast enhancement.
    image = ImageEnhance.Contrast(image).enhance(1.2)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


# --------------------------------------------------
# Extract Tamil text from one image
# --------------------------------------------------

def extract_tamil(image_bytes: bytes) -> str:
    """Send an image to Gemini and return its Tamil transcription."""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_text(text=OCR_PROMPT),
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0,
        ),
    )

    text = response.text

    if not text or not text.strip():
        raise RuntimeError("Gemini returned an empty transcription.")

    return text.strip()


# --------------------------------------------------
# Process an image file
# --------------------------------------------------

def process_image(file_path: Path) -> list[str]:
    with Image.open(file_path) as image:
        image_bytes = prepare_image(image.copy())

    print(f"Extracting Tamil from {file_path.name}...")

    text = extract_tamil(image_bytes)

    output_file = OUTPUT_DIR / f"{file_path.stem}.txt"
    output_file.write_text(text + "\n", encoding="utf-8")

    print(f"Saved: {output_file}")
    return [text]


# --------------------------------------------------
# Process every page of a PDF
# --------------------------------------------------

def process_pdf(file_path: Path) -> list[str]:
    document = fitz.open(file_path)
    page_texts = []

    try:
        for page_number, page in enumerate(document, start=1):
            print(
                f"Processing page {page_number}/{len(document)}..."
            )

            # Render the PDF page at higher resolution.
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2.5, 2.5),
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )

            image_bytes = prepare_image(image)
            text = extract_tamil(image_bytes)

            page_texts.append(text)

            # Save each page independently.
            page_file = (
                OUTPUT_DIR
                / f"{file_path.stem}_page_{page_number:03d}.txt"
            )

            page_file.write_text(
                text + "\n",
                encoding="utf-8",
            )

            print(f"Saved: {page_file}")

    finally:
        document.close()

    # Combine the complete document into one text file.
    combined_file = OUTPUT_DIR / f"{file_path.stem}_complete.txt"

    combined_text = "\n\n".join(
        f"--- பக்கம் {i} ---\n{text}"
        for i, text in enumerate(page_texts, start=1)
    )

    combined_file.write_text(
        combined_text + "\n",
        encoding="utf-8",
    )

    print(f"\nComplete transcription saved: {combined_file}")

    return page_texts


# --------------------------------------------------
# Main entry point
# --------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python main.py <image_or_pdf_path>\n"
            "Example: python main.py old_tamil_page.jpg\n"
            "Example: python main.py old_tamil_document.pdf"
        )
        return

    file_path = Path(sys.argv[1])

    if not file_path.is_file():
        print(f"File not found: {file_path}")
        return

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        process_pdf(file_path)

    elif extension in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        process_image(file_path)

    else:
        print("Unsupported file type. Use an image or PDF.")


if __name__ == "__main__":
    main()