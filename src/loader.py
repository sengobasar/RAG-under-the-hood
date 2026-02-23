import pytesseract
from pdf2image import convert_from_path
import re
import cv2
import numpy as np
import unicodedata


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POPPLER_PATH = r"C:\Users\bowjo\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"


# ---------------- IMAGE PREPROCESS ---------------- #

def preprocess_image(pil_image):

    img = np.array(pil_image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.fastNlMeansDenoising(gray, None, 25, 7, 21)

    gray = cv2.equalizeHist(gray)

    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 3
    )

    return gray


# ---------------- CLEAN TEXT ---------------- #

def clean_text(text):

    text = unicodedata.normalize("NFKC", text)

    text = "".join(c for c in text if c.isprintable())

    text = re.sub(r"[•◦■◆●►▶※©®™|_=~^`]", "", text)

    text = re.sub(r"\b[a-zA-Z]{1,2}\b", "", text)

    text = re.sub(r"[^\w\s।.!?,]{3,}", "", text)

    text = re.sub(r"\b\d+\b", "", text)

    text = re.sub(r"\.{2,}", ".", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------- LOADER ---------------- #

def load_pdf(path):

    print("Running OCR (Hindi + English)...")

    pages = convert_from_path(
        path,
        dpi=400,
        poppler_path=POPPLER_PATH
    )

    full_text = ""

    tess_config = "--oem 3 --psm 4 -c preserve_interword_spaces=1"

    for i, page in enumerate(pages):

        print(f"OCR page {i+1}/{len(pages)}")

        img = preprocess_image(page)

        raw = pytesseract.image_to_string(
            img,
            lang="hin+eng",
            config=tess_config
        )

        cleaned = clean_text(raw)

        if len(cleaned) > 50:
            full_text += cleaned + "\n\n"

    return full_text