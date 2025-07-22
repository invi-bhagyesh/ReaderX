import fitz  # PyMuPDF
import os
import json
import re

# === Phase 1: Configuration & Preprocessing ===
PAGES_TO_SCAN = 2
TOP_MARGIN_THRESHOLD = 0.30
CENTERING_TOLERANCE = 0.10
MIN_WORD_COUNT = 1
MAX_WORD_COUNT = 25

INPUT_FOLDER = "pdfs"
OUTPUT_FOLDER = "output_jsons"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def is_bold(font_name):
    return any(weight in font_name.lower() for weight in ["bold", "black", "heavy", "demi"])

def is_all_caps(text):
    return text.isupper()

def is_title_case(text):
    return text.istitle()

def is_page_number(text):
    return text.strip().isdigit()

def ends_with_period(text):
    return text.strip().endswith(".")

def extract_candidate_blocks(doc):
    candidates = []
    for page_index in range(min(PAGES_TO_SCAN, len(doc))):
        page = doc[page_index]
        page_width = page.rect.width
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    words = text.split()
                    if len(words) < MIN_WORD_COUNT or len(words) > MAX_WORD_COUNT:
                        continue

                    x0, y0, x1, y1 = span["bbox"]
                    x_center = (x0 + x1) / 2

                    candidates.append({
                        "text": text,
                        "font_size": span["size"],
                        "font_name": span["font"],
                        "is_bold": is_bold(span["font"]),
                        "y_position": y0,
                        "x_center": x_center,
                        "page_width": page_width,
                        "page_height": page.rect.height,
                        "page_number": page_index + 1,
                        "bbox": span["bbox"],
                    })

    return candidates

def score_candidates(candidates):
    if not candidates:
        return []

    max_font_size = max(c["font_size"] for c in candidates)
    scored = []

    for c in candidates:
        score = 0

        # Font Size Score (40 pts)
        score += (c["font_size"] / max_font_size) * 40

        # Position Score (25 pts)
        if c["y_position"] < c["page_height"] * TOP_MARGIN_THRESHOLD:
            score += 25

        # Boldness Score (15 pts)
        if c["is_bold"]:
            score += 15

        # Centering Score (10 pts)
        page_center = c["page_width"] / 2
        tolerance = c["page_width"] * CENTERING_TOLERANCE
        if abs(c["x_center"] - page_center) < tolerance:
            score += 10

        # Case Style Score (5 pts)
        if is_all_caps(c["text"]) or is_title_case(c["text"]):
            score += 5

        # Punctuation Penalty (-10 pts)
        if ends_with_period(c["text"]):
            score -= 10

        c["score"] = score
        scored.append(c)

    return sorted(scored, key=lambda x: x["score"], reverse=True)

def try_subtitle_merge(winner, candidates):
    winner_y1 = winner["bbox"][3]
    winner_page = winner["page_number"]
    subtitle_candidates = [
        c for c in candidates
        if c["page_number"] == winner_page and
           c["y_position"] > winner_y1 and
           abs(c["x_center"] - winner["x_center"]) < 20 and
           c["font_size"] <= winner["font_size"] and
           c["y_position"] - winner_y1 < 40
    ]

    if not subtitle_candidates:
        return winner["text"]

    top5_candidates = set(c["text"] for c in candidates[:5])
    for sub in subtitle_candidates:
        if sub["text"] in top5_candidates:
            return f"{winner['text']} {sub['text']}"
    return winner["text"]

import os
import fitz  # PyMuPDF
import json
from collections import defaultdict

FOLDER_PATH = "pdfs"
OUTPUT_FILE = "output.json"

def extract_title(page):
    """Extract boldest and largest text from first page as title."""
    blocks = page.get_text("dict")["blocks"]
    max_font = 0
    title = "Untitled"

    for block in blocks:
        for line in block.get("lines", []):
            for span in line["spans"]:
                font_size = span["size"]
                is_bold = "bold" in span["font"].lower()
                if font_size > max_font and is_bold:
                    max_font = font_size
                    title = span["text"].strip()
    return title


def extract_headings(doc):
    """Extract headings based on font size, boldness, and numbering patterns."""
    headings = []

    # Collect all spans with styles
    span_data = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text or len(text) < 3:
                        continue
                    span_data.append({
                        "text": text,
                        "font": span["font"],
                        "size": span["size"],
                        "bold": "bold" in span["font"].lower(),
                        "page": page_num + 1
                    })

    # Find most common font sizes to classify headings
    size_count = defaultdict(int)
    for span in span_data:
        size_count[round(span["size"])] += 1

    sorted_sizes = sorted(size_count.items(), key=lambda x: -x[0])  # largest to smallest
    levels = {}
    if sorted_sizes:
        if len(sorted_sizes) > 0:
            levels[sorted_sizes[0][0]] = "H1"
        if len(sorted_sizes) > 1:
            levels[sorted_sizes[1][0]] = "H2"
        if len(sorted_sizes) > 2:
            levels[sorted_sizes[2][0]] = "H3"

    for span in span_data:
        size_rounded = round(span["size"])
        level = levels.get(size_rounded)
        if level:
            headings.append({
                "level": level,
                "text": span["text"],
                "page": span["page"]
            })

    return headings


def process_pdf(filepath):
    doc = fitz.open(filepath)
    title = extract_title(doc[0])
    outline = extract_headings(doc)
    return {
        "title": title,
        "outline": outline
    }


def main():
    results = {}
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".pdf"):
            path = os.path.join(FOLDER_PATH, filename)
            print(f"Processing {filename}...")
            try:
                results[filename] = process_pdf(path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
