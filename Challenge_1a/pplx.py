import fitz  # PyMuPDF
import os
import json
import re
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class TextSpan:
    text: str
    font_name: str
    font_size: float
    is_bold: bool
    is_italic: bool
    bbox: Tuple[float, float, float, float]
    page_num: int
    x_center: float
    y_position: float
    line_height: float

class DecisionTreeHeadingClassifier:
    def __init__(self):
        """Decision tree with optimized rules for heading classification."""
        self.rules = [
            {
                'condition': lambda f: f['font_size_ratio'] > 1.3 and f['is_bold'],
                'weight': 0.9,
                'description': 'Large bold text'
            },
            {
                'condition': lambda f: f['pattern_match'] and f['font_size_ratio'] > 1.1,
                'weight': 0.85,
                'description': 'Numbered/lettered headings'
            },
            {
                'condition': lambda f: f['position_score'] > 0.8 and f['capitalization'] > 0.8,
                'weight': 0.8,
                'description': 'Top positioned, well capitalized'
            },
            {
                'condition': lambda f: f['center_alignment'] > 0.7 and f['font_size_ratio'] > 1.2,
                'weight': 0.75,
                'description': 'Centered large text'
            },
            {
                'condition': lambda f: f['font_size_ratio'] > 1.5,
                'weight': 0.7,
                'description': 'Very large text'
            },
            {
                'condition': lambda f: f['pattern_match'] and f['position_score'] > 0.6,
                'weight': 0.65,
                'description': 'Structured headings'
            }
        ]
        
        # Heading patterns
        self.heading_patterns = [
            r'^\d+\.?\s+',  # 1. or 1 
            r'^[A-Z]\.\s+',  # A. 
            r'^[IVX]+\.?\s+',  # Roman numerals
            r'^Chapter\s+\d+',  # Chapter 1
            r'^Section\s+\d+',  # Section 1
            r'^Part\s+\d+',  # Part 1
            r'^\d+\.\d+\.?\s+',  # 1.1.
            r'^\(\d+\)\s+',  # (1)
            r'^[a-z]\)\s+',  # a)
        ]
    
    def extract_features(self, span: TextSpan, hierarchy: Dict, body_font_size: float, page_width: float) -> Dict[str, float]:
        """Extract normalized features for decision tree."""
        features = {}
        
        # Font size ratio (normalized)
        features['font_size_ratio'] = min(span.font_size / body_font_size, 2.0)
        
        # Boolean features (0/1)
        features['is_bold'] = 1.0 if span.is_bold else 0.0
        features['pattern_match'] = 1.0 if self.matches_heading_pattern(span.text) else 0.0
        
        # Position score (0-1, higher for top of page)
        features['position_score'] = max(0, 1.0 - (span.y_position / 800))  # Assume max height
        
        # Length score (optimal for headings)
        word_count = len(span.text.split())
        features['length_score'] = 1.0 if 2 <= word_count <= 12 else 0.5
        
        # Capitalization score
        if span.text.isupper():
            features['capitalization'] = 1.0
        elif span.text.istitle():
            features['capitalization'] = 0.8
        else:
            features['capitalization'] = 0.3
        
        # Center alignment (0-1)
        page_center = page_width / 2
        distance_ratio = abs(span.x_center - page_center) / (page_width / 2)
        features['center_alignment'] = max(0, 1.0 - distance_ratio)
        
        return features
    
    def matches_heading_pattern(self, text: str) -> bool:
        """Check if text matches heading patterns."""
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in self.heading_patterns)
    
    def predict(self, span: TextSpan, hierarchy: Dict, body_font_size: float, page_width: float) -> Dict:
        """Predict using decision tree rules."""
        features = self.extract_features(span, hierarchy, body_font_size, page_width)
        
        max_confidence = 0.0
        triggered_rule = None
        
        # Apply decision tree rules
        for rule in self.rules:
            if rule['condition'](features):
                if rule['weight'] > max_confidence:
                    max_confidence = rule['weight']
                    triggered_rule = rule['description']
        
        return {
            'is_heading': max_confidence > 0.6,
            'confidence': max_confidence,
            'rule_triggered': triggered_rule,
            'features': features
        }

class EnhancedPDFHeadingExtractor:
    def __init__(self):
        # Configuration
        self.MIN_HEADING_LENGTH = 3
        self.MAX_HEADING_LENGTH = 200
        self.TOP_MARGIN_THRESHOLD = 0.4
        self.CENTERING_TOLERANCE = 0.15
        self.MIN_FONT_SIZE_RATIO = 1.1  # Headings should be at least 10% larger
        
        # Initialize NLP models
        self.nlp = None
        self.classifier = None
        self.decision_tree = None
        self.stop_words = set(stopwords.words('english'))
        
        # Load lightweight models
        self._load_models()
        
        # Heading patterns
        self.heading_patterns = [
            r'^\d+\.?\s+',  # 1. or 1 
            r'^[A-Z]\.\s+',  # A. 
            r'^[IVX]+\.?\s+',  # Roman numerals
            r'^Chapter\s+\d+',  # Chapter 1
            r'^Section\s+\d+',  # Section 1
            r'^Part\s+\d+',  # Part 1
            r'^\d+\.\d+\.?\s+',  # 1.1.
            r'^\(\d+\)\s+',  # (1)
            r'^[a-z]\)\s+',  # a)
        ]
    
    def _load_models(self):
        """Load lightweight models and decision tree classifier."""
        try:
            # Use blank spaCy model (minimal size)
            import spacy.blank
            self.nlp = spacy.blank("en")
        except:
            self.nlp = None
        
        # Skip transformer - too large
        self.classifier = None
        
        # Add decision tree classifier
        self.decision_tree = DecisionTreeHeadingClassifier()
        print("Loaded decision tree classifier for improved accuracy")
        
    def extract_text_spans(self, doc: fitz.Document) -> List[TextSpan]:
        """Extract all text spans with their properties."""
        spans = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    line_height = line["bbox"][3] - line["bbox"][1]
                    
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text or len(text) < 2:
                            continue
                        
                        # Calculate properties
                        bbox = span["bbox"]
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_position = bbox[1]
                        
                        # Font analysis
                        font_name = span["font"].lower()
                        is_bold = any(weight in font_name for weight in 
                                    ["bold", "black", "heavy", "demi", "medium"])
                        is_italic = "italic" in font_name or "oblique" in font_name
                        
                        spans.append(TextSpan(
                            text=text,
                            font_name=span["font"],
                            font_size=span["size"],
                            is_bold=is_bold,
                            is_italic=is_italic,
                            bbox=bbox,
                            page_num=page_num + 1,
                            x_center=x_center,
                            y_position=y_position,
                            line_height=line_height
                        ))
        
        return spans
    
    def analyze_font_hierarchy(self, spans: List[TextSpan]) -> Tuple[Dict[float, int], float]:
        """Analyze font sizes to determine heading hierarchy."""
        font_sizes = [span.font_size for span in spans]
        font_counter = Counter(round(size, 1) for size in font_sizes)
        
        # Get most common font size (likely body text)
        body_font_size = font_counter.most_common(1)[0][0]
        
        # Identify distinct font sizes for headings
        unique_sizes = sorted(set(round(size, 1) for size in font_sizes), reverse=True)
        heading_sizes = [size for size in unique_sizes 
                        if size >= body_font_size * self.MIN_FONT_SIZE_RATIO]
        
        # Assign hierarchy levels
        hierarchy = {}
        for i, size in enumerate(heading_sizes[:6]):  # Max 6 levels
            hierarchy[size] = i + 1
        
        return hierarchy, body_font_size
    
    def calculate_heading_score(self, span: TextSpan, hierarchy: Dict[float, int], 
                              body_font_size: float, page_width: float, 
                              page_height: float) -> float:
        """Calculate likelihood score for a span being a heading."""
        score = 0.0
        
        # Font size score (0-40 points)
        if round(span.font_size, 1) in hierarchy:
            level = hierarchy[round(span.font_size, 1)]
            score += max(0, 40 - (level - 1) * 5)
        
        # Bold/formatting score (0-20 points)
        if span.is_bold:
            score += 15
        if span.is_italic:
            score += 5
        
        # Position score (0-20 points)
        # Higher score for text in upper portion of page
        relative_y = span.y_position / page_height
        if relative_y < self.TOP_MARGIN_THRESHOLD:
            score += 20
        elif relative_y < 0.7:
            score += 10
        
        # Centering score (0-15 points)
        page_center = page_width / 2
        distance_from_center = abs(span.x_center - page_center)
        if distance_from_center < page_width * self.CENTERING_TOLERANCE:
            score += 15
        
        # Length score (0-10 points)
        word_count = len(span.text.split())
        if 2 <= word_count <= 12:
            score += 10
        elif word_count <= 20:
            score += 5
        
        # Pattern matching score (0-15 points)
        if self.matches_heading_pattern(span.text):
            score += 15
        
        # Capitalization score (0-10 points)
        if span.text.isupper():
            score += 10
        elif span.text.istitle():
            score += 8
        elif span.text[0].isupper():
            score += 5
        
        # Punctuation penalties
        if span.text.endswith('.') and not self.matches_heading_pattern(span.text):
            score -= 10
        if span.text.endswith(','):
            score -= 15
        
        # Content quality score using NLP (0-10 points)
        if self.nlp:
            score += self.analyze_content_quality(span.text)
        
        return score
    
    def calculate_hybrid_heading_score(self, span: TextSpan, hierarchy: Dict[float, int], 
                                     body_font_size: float, page_width: float, 
                                     page_height: float) -> float:
        """Hybrid approach: combine rule-based scoring with decision tree."""
        
        # Get original rule-based score (0-100)
        rule_score = self.calculate_heading_score(span, hierarchy, body_font_size, page_width, page_height)
        
        # Get decision tree prediction (0-1)
        tree_prediction = self.decision_tree.predict(span, hierarchy, body_font_size, page_width)
        tree_confidence = tree_prediction['confidence']
        
        # Combine scores with weighted average
        # 70% rule-based, 30% decision tree
        alpha = 0.7
        hybrid_score = alpha * rule_score + (1 - alpha) * tree_confidence * 100
        
        # Boost score if decision tree is highly confident
        if tree_confidence > 0.8:
            hybrid_score *= 1.1  # 10% boost for high-confidence predictions
        
        # Add bonus for rule triggering
        if tree_prediction['rule_triggered'] and tree_confidence > 0.7:
            hybrid_score += 5  # Small bonus for rule-based detection
        
        return min(hybrid_score, 100)  # Cap at 100
    
    def matches_heading_pattern(self, text: str) -> bool:
        """Check if text matches common heading patterns."""
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in self.heading_patterns)
    
    def analyze_content_quality(self, text: str) -> float:
        """Use NLP to assess if text looks like a heading."""
        try:
            doc = self.nlp(text)
            score = 0.0
            
            # Check for proper nouns (good for headings)
            proper_nouns = [token for token in doc if token.pos_ == "PROPN"]
            score += min(len(proper_nouns) * 2, 5)
            
            # Check for stop words (fewer is better for headings)
            stop_word_count = sum(1 for token in doc if token.text.lower() in self.stop_words)
            if len(doc) > 0 and stop_word_count / len(doc) < 0.3:
                score += 3
            
            # Check for verbs (fewer is better for headings)
            verb_count = sum(1 for token in doc if token.pos_ in ["VERB", "AUX"])
            if verb_count == 0:
                score += 2
            
            return min(score, 10)
        except:
            return 0
    
    def merge_multiline_headings(self, candidates: List[TextSpan]) -> List[TextSpan]:
        """Merge headings that span multiple lines."""
        merged = []
        i = 0
        
        while i < len(candidates):
            current = candidates[i]
            
            # Look for continuation on next line
            if i + 1 < len(candidates):
                next_span = candidates[i + 1]
                
                # Check if next span is likely a continuation
                if (next_span.page_num == current.page_num and
                    abs(next_span.x_center - current.x_center) < 50 and
                    next_span.y_position - current.bbox[3] < current.line_height * 1.5 and
                    abs(next_span.font_size - current.font_size) < 1):
                    
                    # Merge the spans
                    merged_text = f"{current.text} {next_span.text}"
                    merged_span = TextSpan(
                        text=merged_text,
                        font_name=current.font_name,
                        font_size=current.font_size,
                        is_bold=current.is_bold,
                        is_italic=current.is_italic,
                        bbox=current.bbox,
                        page_num=current.page_num,
                        x_center=current.x_center,
                        y_position=current.y_position,
                        line_height=current.line_height
                    )
                    merged.append(merged_span)
                    i += 2  # Skip next span as it's been merged
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def extract_title(self, doc: fitz.Document) -> str:
        """Extract document title from first page."""
        if not doc:
            return "Untitled"
        
        # Create a new document with just the first page
        first_page_doc = fitz.open()
        first_page_doc.insert_pdf(doc, from_page=0, to_page=0)
        
        spans = self.extract_text_spans(first_page_doc)
        first_page_doc.close()
        
        if not spans:
            return "Untitled"
        
        # Filter spans from first page only
        first_page_spans = [s for s in spans if s.page_num == 1]
        
        # Get page dimensions
        page = doc[0]
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Analyze font hierarchy
        hierarchy, body_font_size = self.analyze_font_hierarchy(first_page_spans)
        
        # Score candidates for title
        title_candidates = []
        for span in first_page_spans:
            if (self.MIN_HEADING_LENGTH <= len(span.text) <= self.MAX_HEADING_LENGTH and
                span.y_position < page_height * 0.3):  # Upper 30% of page
                
                score = self.calculate_hybrid_heading_score(
                    span, hierarchy, body_font_size, page_width, page_height)
                title_candidates.append((span, score))
        
        if title_candidates:
            # Return highest scoring candidate
            best_candidate = max(title_candidates, key=lambda x: x[1])
            return best_candidate[0].text
        
        return "Untitled"
    
    def extract_headings(self, doc: fitz.Document) -> List[Dict]:
        """Extract all headings using hybrid approach."""
        spans = self.extract_text_spans(doc)
        if not spans:
            return []
        
        # Analyze font hierarchy
        hierarchy, body_font_size = self.analyze_font_hierarchy(spans)
        
        # Filter potential heading candidates
        candidates = []
        for span in spans:
            if (self.MIN_HEADING_LENGTH <= len(span.text) <= self.MAX_HEADING_LENGTH and
                span.font_size >= body_font_size * self.MIN_FONT_SIZE_RATIO):
                
                page = doc[span.page_num - 1]
                page_width = page.rect.width
                page_height = page.rect.height
                
                # Use hybrid scoring instead of rule-based only
                score = self.calculate_hybrid_heading_score(
                    span, hierarchy, body_font_size, page_width, page_height)
                
                if score > 35:  # Slightly higher threshold for hybrid approach
                    candidates.append((span, score))
        
        # Sort by score and select top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_spans = [candidate[0] for candidate in candidates[:50]]
        
        # Merge multiline headings
        merged_spans = self.merge_multiline_headings(selected_spans)
        
        # Convert to output format with confidence scores
        headings = []
        candidate_dict = {id(span): score for span, score in candidates}
        
        for span in merged_spans:
            level = hierarchy.get(round(span.font_size, 1), 6)
            confidence = candidate_dict.get(id(span), 0)
            
            headings.append({
                "level": f"H{min(level, 6)}",
                "text": span.text,
                "page": span.page_num,
                "confidence": confidence
            })
        
        # Sort by page and position
        headings.sort(key=lambda x: (x["page"], 0))  # Removed y_position as it's not in dict
        
        return headings
    
    def process_pdf(self, filepath: str) -> Dict:
        """Process a PDF file and extract title and headings."""
        try:
            doc = fitz.open(filepath)
            title = self.extract_title(doc)
            headings = self.extract_headings(doc)
            doc.close()
            
            return {
                "title": title,
                "headings": headings,
                "total_pages": len(fitz.open(filepath)),
                "status": "success"
            }
        except Exception as e:
            return {
                "title": "Error",
                "headings": [],
                "total_pages": 0,
                "status": "error",
                "error": str(e)
            }

def main():
    """Main function to process PDFs from input directory."""
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = EnhancedPDFHeadingExtractor()
    
    # Process all PDF files
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    for filename in pdf_files:
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + '.json'
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing {filename}...")
        
        try:
            result = extractor.process_pdf(input_path)
            
            # Save individual result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            if result["status"] == "success":
                print(f"  ✓ Generated {output_filename}")
                print(f"  ✓ Title: {result['title']}")
                print(f"  ✓ Found {len(result['headings'])} headings")
            else:
                print(f"  ✗ Error: {result['error']}")
                
        except Exception as e:
            print(f"  ✗ Failed to process {filename}: {e}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
