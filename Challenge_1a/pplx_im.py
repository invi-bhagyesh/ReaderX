import fitz  # PyMuPDF
import os
import json
import re
import statistics
import signal
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
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

class EnhancedDecisionTreeClassifier:
    def __init__(self):
        """Enhanced decision tree with sophisticated rules and comprehensive multilingual support."""
        self.rules = [
            # High confidence rules
            {
                'condition': lambda f: f['font_size_ratio'] > 1.4 and f['is_bold'] and f['position_score'] > 0.7,
                'weight': 0.95,
                'description': 'Large bold text in prime position'
            },
            {
                'condition': lambda f: f['pattern_match'] > 0.8 and f['font_size_ratio'] > 1.2 and f['length_score'] > 0.8,
                'weight': 0.92,
                'description': 'Well-structured numbered headings'
            },
            # Context-aware rules
            {
                'condition': lambda f: f['font_size_ratio'] > 1.3 and f['center_alignment'] > 0.8 and f['capitalization'] > 0.9,
                'weight': 0.88,
                'description': 'Centered, capitalized titles'
            },
            {
                'condition': lambda f: f['font_size_ratio'] > 1.3 and f['is_bold'],
                'weight': 0.85,
                'description': 'Large bold text'
            },
            {
                'condition': lambda f: f['pattern_match'] > 0.7 and f['font_size_ratio'] > 1.1,
                'weight': 0.82,
                'description': 'Numbered/lettered headings'
            },
            {
                'condition': lambda f: f['position_score'] > 0.8 and f['capitalization'] > 0.8,
                'weight': 0.78,
                'description': 'Top positioned, well capitalized'
            },
            # East Asian specific rules
            {
                'condition': lambda f: f['pattern_match'] > 0.9 and f['font_size_ratio'] > 1.05,
                'weight': 0.88,
                'description': 'East Asian structured headings'
            },
            # Edge case handlers
            {
                'condition': lambda f: f['font_size_ratio'] > 1.6,
                'weight': 0.8,
                'description': 'Oversized text likely heading'
            },
            {
                'condition': lambda f: f['center_alignment'] > 0.7 and f['font_size_ratio'] > 1.2,
                'weight': 0.72,
                'description': 'Centered large text'
            },
            {
                'condition': lambda f: f['pattern_match'] > 0.5 and f['position_score'] > 0.6,
                'weight': 0.68,
                'description': 'Structured headings'
            }
        ]
        
        # Comprehensive multilingual heading patterns with East Asian support
        self.multilingual_patterns = {
            'universal': [
                r'^\d+\.?\s*',  # 1. or 1 (more flexible spacing)
                r'^\d+\.\d+\.?\s*',  # 1.1.
                r'^\(\d+\)\s*',  # (1)
                r'^[A-Z]\.\s*',  # A.
                r'^[IVX]+\.?\s*',  # Roman numerals
                r'^[a-z]\)\s*',  # a)
            ],
            # Western languages
            'english': [r'^Chapter\s+\d+', r'^Section\s+\d+', r'^Part\s+\d+', r'^Appendix\s+[A-Z]'],
            'spanish': [r'^Capítulo\s+\d+', r'^Sección\s+\d+', r'^Parte\s+\d+', r'^Apéndice\s+[A-Z]'],
            'french': [r'^Chapitre\s+\d+', r'^Section\s+\d+', r'^Partie\s+\d+', r'^Annexe\s+[A-Z]'],
            'german': [r'^Kapitel\s+\d+', r'^Abschnitt\s+\d+', r'^Teil\s+\d+', r'^Anhang\s+[A-Z]'],
            'portuguese': [r'^Capítulo\s+\d+', r'^Seção\s+\d+', r'^Parte\s+\d+', r'^Apêndice\s+[A-Z]'],
            'italian': [r'^Capitolo\s+\d+', r'^Sezione\s+\d+', r'^Parte\s+\d+', r'^Appendice\s+[A-Z]'],
            
            # East Asian languages
            'chinese': [
                r'^第[一二三四五六七八九十\d]+章',  # 第一章, 第2章
                r'^第[一二三四五六七八九十\d]+节',  # 第一节, 第2节
                r'^第[一二三四五六七八九十\d]+部分', # 第一部分
                r'^第[一二三四五六七八九十\d]+条',  # 第一条
                r'^[一二三四五六七八九十]+[、．.]',  # 一、 二. 三．
                r'^\d+[、．.]',  # 1、 2. 3．
                r'^[\（\(][一二三四五六七八九十\d]+[\）\)]',  # （一） (1)
                r'^附录[一二三四五六七八九十A-Z\d]+',  # 附录A
            ],
            'japanese': [
                r'^第[一二三四五六七八九十\d]+章',  # 第一章, 第2章
                r'^第[一二三四五六七八九十\d]+節',  # 第一節
                r'^第[一二三四五六七八九十\d]+部',  # 第一部
                r'^第[一二三四五六七八九十\d]+項',  # 第一項
                r'^[一二三四五六七八九十]+[、．.]',  # 一、 二．
                r'^\d+[、．.]',  # 1、 2．
                r'^[\（\(][一二三四五六七八九十\d]+[\）\)]',  # （一） (1)
                r'^付録[一二三四五六七八九十A-Z\d]+',  # 付録A
            ],
            'korean': [
                r'^제[일이삼사오육칠팔구십\d]+장',  # 제1장, 제일장
                r'^제[일이삼사오육칠팔구십\d]+절',  # 제1절
                r'^제[일이삼사오육칠팔구십\d]+부',  # 제1부
                r'^제[일이삼사오육칠팔구십\d]+항',  # 제1항
                r'^[일이삼사오육칠팔구십]+[\.．]',  # 일. 이．
                r'^\d+[\.．]',  # 1. 2．
                r'^[\（\(][일이삼사오육칠팔구십\d]+[\）\)]',  # （일） (1)
                r'^부록\s*[일이삼사오육칠팔구십A-Z\d]+',  # 부록 가
            ],
            'thai': [
                r'^บทที่\s*\d+',  # บทที่ 1
                r'^ส่วนที่\s*\d+',  # ส่วนที่ 1
                r'^หมวดที่\s*\d+',  # หมวดที่ 1
                r'^ข้อที่\s*\d+',  # ข้อที่ 1
                r'^\d+\.',  # 1.
                r'^[\（\(]\d+[\）\)]',  # (1)
                r'^ภาคผนวก\s*[ก-ฮA-Z\d]+',  # ภาคผนวก ก
            ],
            'vietnamese': [
                r'^Chương\s+\d+',  # Chương 1
                r'^Phần\s+\d+',  # Phần 1
                r'^Mục\s+\d+',  # Mục 1
                r'^Tiết\s+\d+',  # Tiết 1
                r'^\d+\.',  # 1.
                r'^[\（\(]\d+[\）\)]',  # (1)
                r'^Phụ lục\s*[A-Z\d]+',  # Phụ lục A
            ],
            # Additional East Asian languages
            'hindi': [
                r'^अध्याय\s*\d+',  # अध्याय 1
                r'^भाग\s*\d+',  # भाग 1
                r'^\d+\.',  # 1.
                r'^[\（\(]\d+[\）\)]',  # (1)
            ],
            'arabic': [
                r'^الفصل\s*\d+',  # الفصل 1
                r'^الجزء\s*\d+',  # الجزء 1
                r'^\d+\.',  # 1.
                r'^[\（\(]\d+[\）\)]',  # (1)
            ]
        }
    
    def extract_features(self, span: TextSpan, hierarchy: Dict, body_font_size: float, page_width: float) -> Dict[str, float]:
        """Extract normalized features for decision tree with East Asian considerations."""
        features = {}
        
        # Font size ratio (normalized)
        features['font_size_ratio'] = min(span.font_size / body_font_size, 2.5)
        
        # Boolean features (0/1)
        features['is_bold'] = 1.0 if span.is_bold else 0.0
        features['pattern_match'] = self.detect_language_patterns(span.text)
        
        # Position score (0-1, higher for top of page)
        features['position_score'] = max(0, 1.0 - (span.y_position / 800))
        
        # Enhanced length score for different script types
        script_type = self.detect_script_type(span.text)
        if script_type in ['cjk', 'thai']:
            # Character-based counting for East Asian languages
            char_count = len([c for c in span.text if not c.isspace()])
            features['length_score'] = 1.0 if 2 <= char_count <= 20 else (0.8 if char_count <= 40 else 0.3)
        else:
            # Word-based counting for Western languages
            word_count = len(span.text.split())
            features['length_score'] = 1.0 if 2 <= word_count <= 12 else (0.8 if word_count <= 20 else 0.3)
        
        # Enhanced capitalization score with East Asian considerations
        if script_type == 'cjk':
            # For CJK, look for mixed case or special characters
            has_latin = bool(re.search(r'[A-Za-z]', span.text))
            if has_latin and span.text[0].isupper():
                features['capitalization'] = 0.8
            else:
                features['capitalization'] = 0.5  # Neutral for pure CJK
        else:
            # Original logic for Western scripts
            if span.text.isupper():
                features['capitalization'] = 1.0
            elif span.text.istitle():
                features['capitalization'] = 0.9
            elif span.text[0].isupper() and sum(1 for c in span.text if c.isupper()) >= 2:
                features['capitalization'] = 0.7
            else:
                features['capitalization'] = 0.3
        
        # Center alignment (0-1)
        page_center = page_width / 2
        distance_ratio = abs(span.x_center - page_center) / (page_width / 2)
        features['center_alignment'] = max(0, 1.0 - distance_ratio)
        
        return features
    
    def detect_script_type(self, text: str) -> str:
        """Detect the script type of the text for appropriate analysis."""
        # Count characters by script
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text))
        thai_chars = len(re.findall(r'[\u0e00-\u0e7f]', text))
        arabic_chars = len(re.findall(r'[\u0600-\u06ff\u0750-\u077f]', text))
        devanagari_chars = len(re.findall(r'[\u0900-\u097f]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len([c for c in text if not c.isspace() and not c.isdigit()])
        
        if total_chars == 0:
            return 'unknown'
        
        # Determine dominant script
        if cjk_chars / total_chars > 0.3:
            return 'cjk'
        elif thai_chars / total_chars > 0.3:
            return 'thai'
        elif arabic_chars / total_chars > 0.3:
            return 'arabic'
        elif devanagari_chars / total_chars > 0.3:
            return 'devanagari'
        elif latin_chars / total_chars > 0.3:
            return 'latin'
        else:
            return 'mixed'
    
    def detect_language_patterns(self, text: str) -> float:
        """Enhanced pattern detection with comprehensive multilingual support."""
        score = 0.0
        
        # Check universal patterns first (highest priority)
        for pattern in self.multilingual_patterns['universal']:
            if re.match(pattern, text, re.IGNORECASE):
                return 1.0
        
        # Check East Asian patterns (high priority)
        east_asian_langs = ['chinese', 'japanese', 'korean', 'thai', 'vietnamese', 'hindi', 'arabic']
        for lang in east_asian_langs:
            if lang in self.multilingual_patterns:
                for pattern in self.multilingual_patterns[lang]:
                    if re.match(pattern, text):  # No IGNORECASE for some scripts
                        return 0.95  # Slightly lower than universal but higher than Western
        
        # Check Western language patterns
        western_langs = ['english', 'spanish', 'french', 'german', 'portuguese', 'italian']
        for lang in western_langs:
            if lang in self.multilingual_patterns:
                for pattern in self.multilingual_patterns[lang]:
                    if re.match(pattern, text, re.IGNORECASE):
                        return 0.9
        
        return score
    
    @lru_cache(maxsize=1000)
    def cached_pattern_match(self, text: str) -> bool:
        """Cache pattern matching results for performance."""
        return self.detect_language_patterns(text) > 0.5
    
    def predict(self, span: TextSpan, hierarchy: Dict, body_font_size: float, page_width: float) -> Dict:
        """Predict using enhanced decision tree rules with multilingual support."""
        features = self.extract_features(span, hierarchy, body_font_size, page_width)
        
        max_confidence = 0.0
        triggered_rule = None
        
        # Apply decision tree rules
        for rule in self.rules:
            try:
                if rule['condition'](features):
                    if rule['weight'] > max_confidence:
                        max_confidence = rule['weight']
                        triggered_rule = rule['description']
            except:
                continue  # Skip problematic rules
        
        return {
            'is_heading': max_confidence > 0.6,
            'confidence': max_confidence,
            'rule_triggered': triggered_rule,
            'features': features
        }

class EnhancedPDFHeadingExtractor:
    def __init__(self):
        # Configuration - adjusted for East Asian support
        self.MIN_HEADING_LENGTH = 2  # Shorter for character-based languages
        self.MAX_HEADING_LENGTH = 200
        self.TOP_MARGIN_THRESHOLD = 0.4
        self.CENTERING_TOLERANCE = 0.15
        self.MIN_FONT_SIZE_RATIO = 1.03  # More sensitive for East Asian fonts
        
        # Initialize NLP models
        self.nlp = None
        self.classifier = None
        self.decision_tree = None
        self.stop_words = self._load_multilingual_stopwords()
        
        # Load lightweight models
        self._load_models()
        
        # Performance tracking
        self.processing_stats = {
            'total_spans_processed': 0,
            'headings_found': 0,
            'processing_time': 0
        }
    
    def _load_multilingual_stopwords(self):
        """Load stopwords for multiple languages including East Asian."""
        try:
            # Western languages with NLTK support
            western_languages = ['english', 'spanish', 'french', 'german', 'italian', 'portuguese']
            combined_stopwords = set()
            
            for lang in western_languages:
                try:
                    combined_stopwords.update(stopwords.words(lang))
                except:
                    pass
            
            # Add common East Asian stopwords/particles manually
            east_asian_stopwords = {
                # Chinese
                '的', '了', '和', '是', '在', '有', '我', '你', '他', '她', '它', '们',
                '这', '那', '中', '上', '下', '来', '去', '说', '对', '为', '与', '及',
                '一个', '可以', '就是', '也是', '但是', '因为', '所以', '如果', '虽然',
                
                # Japanese particles and common words
                'の', 'に', 'は', 'を', 'が', 'で', 'て', 'と', 'から', 'まで', 'も', 'や',
                'だ', 'である', 'です', 'ます', 'した', 'する', 'ある', 'いる', 'この', 'その',
                'という', 'として', 'について', 'において', 'による', 'のような',
                
                # Korean particles and common words
                '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', '의', '도',
                '은', '는', '한', '하는', '된', '되는', '있는', '없는', '그', '이것', '저것',
                '그리고', '하지만', '그러나', '따라서', '또한', '즉', '만약', '비록',
                
                # Thai common words
                'ที่', 'และ', 'ใน', 'เป็น', 'มี', 'จะ', 'ได้', 'แล้ว', 'ก็', 'จาก', 'ถึง',
                'ของ', 'กับ', 'โดย', 'ตาม', 'เพื่อ', 'หรือ', 'แต่', 'เพราะ', 'ถ้า',
                
                # Vietnamese common words
                'và', 'của', 'trong', 'là', 'có', 'được', 'với', 'để', 'từ', 'đến', 'này', 'đó',
                'một', 'các', 'những', 'cho', 'về', 'theo', 'như', 'khi', 'nếu', 'nhưng',
                
                # Hindi common words
                'और', 'का', 'के', 'की', 'में', 'से', 'को', 'है', 'हैं', 'था', 'थे', 'होगा',
                'यह', 'वह', 'इस', 'उस', 'एक', 'भी', 'तो', 'ही', 'पर', 'लिए',
                
                # Arabic common words
                'في', 'من', 'إلى', 'على', 'أن', 'هذا', 'هذه', 'ذلك', 'التي', 'الذي',
                'كان', 'كانت', 'يكون', 'تكون', 'قد', 'لقد', 'أو', 'لكن', 'إذا', 'عند'
            }
            
            combined_stopwords.update(east_asian_stopwords)
            
            return combined_stopwords if combined_stopwords else set()
        except:
            return set()
    
    def _load_models(self):
        """Load lightweight models and enhanced decision tree classifier."""
        try:
            # Use blank spaCy model (minimal size)
            import spacy.blank
            self.nlp = spacy.blank("en")
        except:
            self.nlp = None
        
        # Skip transformer - too large for 200MB constraint
        self.classifier = None
        
        # Add enhanced decision tree classifier
        self.decision_tree = EnhancedDecisionTreeClassifier()
        print("Loaded enhanced decision tree classifier with comprehensive multilingual support")
    
    def detect_script_type(self, text: str) -> str:
        """Detect the script type of the text for appropriate analysis."""
        return self.decision_tree.detect_script_type(text)
    
    def batch_process_spans(self, spans: List[TextSpan]) -> List[TextSpan]:
        """Process spans in optimized batches for performance."""
        candidates = []
        
        for span in spans:
            # Quick rejection filters for performance
            if (len(span.text) < self.MIN_HEADING_LENGTH or 
                len(span.text) > self.MAX_HEADING_LENGTH or
                span.text.isdigit() or
                span.text.count(' ') > 25 or  # Increased for East Asian text
                span.text.startswith('http') or
                len(span.text.strip()) != len(span.text)):  # Skip padded text
                continue
                
            candidates.append(span)
        
        return candidates[:250]  # Increased limit for multilingual support
    
    def extract_text_spans(self, doc: fitz.Document) -> List[TextSpan]:
        """Extract all text spans with their properties."""
        spans = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height
            
            try:
                blocks = page.get_text("dict")["blocks"]
            except:
                continue  # Skip problematic pages
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    line_height = line["bbox"][3] - line["bbox"][1]
                    
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text or len(text) < 1:  # More lenient for East Asian single characters
                            continue
                        
                        # Calculate properties
                        bbox = span["bbox"]
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_position = bbox[1]
                        
                        # Enhanced font analysis
                        font_name = span["font"].lower()
                        is_bold = any(weight in font_name for weight in 
                                    ["bold", "black", "heavy", "demi", "medium", "semibold"])
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
    
    def analyze_advanced_font_hierarchy(self, spans: List[TextSpan]) -> Tuple[Dict[float, int], Dict]:
        """Advanced font analysis with East Asian text considerations."""
        font_sizes = [span.font_size for span in spans]
        
        if not font_sizes:
            return {}, {'body_size': 12.0, 'mean': 12.0, 'median': 12.0, 'stdev': 0}
        
        # Statistical analysis
        font_stats = {
            'mean': statistics.mean(font_sizes),
            'median': statistics.median(font_sizes),
            'stdev': statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0
        }
        
        # Dynamic body text detection with script-aware adjustments
        font_counter = Counter(round(size, 1) for size in font_sizes)
        body_candidates = font_counter.most_common(3)
        
        # Choose body font
        if body_candidates:
            body_font_size = max(body_candidates, key=lambda x: x[1])[0]
        else:
            body_font_size = font_stats['median']
        
        # Detect if document contains significant East Asian content
        east_asian_spans = [
            span for span in spans[:100]  # Check first 100 spans
            if self.detect_script_type(span.text) in ['cjk', 'thai', 'arabic', 'devanagari']
        ]
        has_east_asian = len(east_asian_spans) > len(spans) * 0.1  # 10% threshold
        
        # Adjust font size ratio threshold for East Asian text
        min_ratio = 1.02 if has_east_asian else self.MIN_FONT_SIZE_RATIO
        
        # Create hierarchy with adjusted thresholds
        unique_sizes = sorted(set(round(size, 1) for size in font_sizes), reverse=True)
        hierarchy = {}
        
        for i, size in enumerate(unique_sizes):
            if size >= body_font_size * min_ratio:
                frequency = font_counter.get(size, 0)
                size_diff = size - body_font_size
                
                # More lenient frequency threshold for East Asian text
                freq_threshold = 0.15 if has_east_asian else 0.1
                size_threshold = 0.2 if has_east_asian else 0.5
                
                if frequency < len(spans) * freq_threshold and size_diff >= size_threshold:
                    hierarchy[size] = min(i + 1, 6)
        
        font_stats['body_size'] = body_font_size
        font_stats['has_east_asian'] = has_east_asian
        return hierarchy, font_stats
    
    def analyze_context(self, span: TextSpan, all_spans: List[TextSpan]) -> float:
        """Analyze surrounding text context for heading detection."""
        context_score = 0.0
        
        # Find nearby spans on same page
        nearby_spans = [s for s in all_spans 
                       if s.page_num == span.page_num 
                       and abs(s.y_position - span.y_position) < 100
                       and s != span]
        
        if not nearby_spans:
            return 0.1  # Isolated text gets small bonus
        
        # Check if surrounded by smaller text
        smaller_text_count = sum(1 for s in nearby_spans 
                               if s.font_size < span.font_size * 0.9)
        
        if smaller_text_count >= 2:
            context_score += 0.15
        
        # Check for whitespace isolation (heading characteristic)
        above_spans = [s for s in nearby_spans if s.y_position < span.y_position - 10]
        below_spans = [s for s in nearby_spans if s.y_position > span.bbox[3] + 10]
        
        if above_spans and below_spans:
            context_score += 0.1
        elif not above_spans or not below_spans:  # Top or bottom of page
            context_score += 0.05
        
        return context_score
    
    def calculate_enhanced_hybrid_score(self, span: TextSpan, hierarchy: Dict[float, int], 
                                      body_font_size: float, page_width: float, 
                                      page_height: float, all_spans: List[TextSpan]) -> float:
        """Enhanced hybrid scoring with context analysis and multilingual support."""
        
        # Get base rule-based score
        rule_score = self.calculate_heading_score(span, hierarchy, body_font_size, page_width, page_height)
        
        # Get enhanced decision tree prediction
        tree_prediction = self.decision_tree.predict(span, hierarchy, body_font_size, page_width)
        tree_confidence = tree_prediction['confidence']
        
        # Get context analysis
        context_bonus = self.analyze_context(span, all_spans) * 100
        
        # Combine scores with adaptive weighting
        if tree_confidence > 0.8:
            alpha = 0.6  # Trust decision tree more when highly confident
        else:
            alpha = 0.75  # Rely more on rule-based scoring
            
        hybrid_score = alpha * rule_score + (1 - alpha) * tree_confidence * 100
        
        # Add context bonus
        hybrid_score += context_bonus
        
        # Boost score for high-confidence predictions
        if tree_confidence > 0.9:
            hybrid_score *= 1.2  # Higher boost for very high confidence
        elif tree_confidence > 0.85:
            hybrid_score *= 1.15
        elif tree_confidence > 0.75:
            hybrid_score *= 1.08
        
        # Enhanced bonus for specific rule triggering
        if tree_prediction['rule_triggered']:
            if 'prime position' in tree_prediction['rule_triggered']:
                hybrid_score += 10
            elif 'East Asian' in tree_prediction['rule_triggered']:
                hybrid_score += 8
            elif 'numbered' in tree_prediction['rule_triggered']:
                hybrid_score += 6
            else:
                hybrid_score += 3
        
        return min(hybrid_score, 100)
    
    def calculate_heading_score(self, span: TextSpan, hierarchy: Dict[float, int], 
                              body_font_size: float, page_width: float, 
                              page_height: float) -> float:
        """Calculate likelihood score for a span being a heading with multilingual support."""
        score = 0.0
        
        # Enhanced font size scoring (0-40 points)
        if round(span.font_size, 1) in hierarchy:
            level = hierarchy[round(span.font_size, 1)]
            score += max(0, 40 - (level - 1) * 4)  # Less penalty for lower levels
        else:
            # Bonus for very large fonts even if not in hierarchy
            size_ratio = span.font_size / body_font_size
            if size_ratio > 1.3:
                score += 25
            elif size_ratio > 1.15:
                score += 15
            elif size_ratio > 1.05:  # More sensitive for East Asian
                score += 8
        
        # Enhanced formatting score (0-25 points)
        if span.is_bold:
            score += 18
        if span.is_italic:
            score += 5
        if span.is_bold and span.is_italic:
            score += 2  # Bonus for both
        
        # Position score (0-20 points)
        relative_y = span.y_position / page_height
        if relative_y < self.TOP_MARGIN_THRESHOLD:
            score += 20
        elif relative_y < 0.6:
            score += 12
        elif relative_y < 0.8:
            score += 6
        
        # Enhanced centering score (0-15 points)
        page_center = page_width / 2
        distance_from_center = abs(span.x_center - page_center)
        center_ratio = distance_from_center / (page_width / 2)
        
        if center_ratio < 0.1:  # Very centered
            score += 15
        elif center_ratio < 0.2:
            score += 10
        elif center_ratio < 0.3:
            score += 5
        
        # Improved length score (0-12 points) with script awareness
        script_type = self.detect_script_type(span.text)
        if script_type in ['cjk', 'thai']:
            char_count = len([c for c in span.text if not c.isspace()])
            if 2 <= char_count <= 15:
                score += 12
            elif char_count <= 30:
                score += 8
            elif char_count <= 50:
                score += 4
        else:
            word_count = len(span.text.split())
            if 2 <= word_count <= 8:
                score += 12
            elif word_count <= 15:
                score += 8
            elif word_count <= 25:
                score += 4
        
        # Enhanced pattern matching (0-18 points)
        pattern_score = self.decision_tree.detect_language_patterns(span.text)
        score += pattern_score * 18
        
        # Improved capitalization score (0-12 points)
        if script_type == 'cjk':
            # For CJK, neutral scoring
            score += 6
        else:
            if span.text.isupper():
                score += 10
            elif span.text.istitle():
                score += 12  # Title case is often better than all caps
            elif span.text[0].isupper():
                score += 6
        
        # Enhanced punctuation analysis with East Asian punctuation
        east_asian_periods = ['。', '．', '｡']
        east_asian_commas = ['，', '、', 'ヽ']
        
        if span.text.endswith('.') and not pattern_score:
            score -= 8
        elif any(span.text.endswith(p) for p in east_asian_periods) and not pattern_score:
            score -= 6  # Less penalty for East Asian periods
        if span.text.endswith(',') or any(span.text.endswith(p) for p in east_asian_commas):
            score -= 10
        if span.text.endswith(':') or span.text.endswith('：'):  # Including full-width colon
            score += 3
        
        # Content quality using enhanced NLP
        score += self.analyze_enhanced_content_quality(span.text)
        
        return max(0, score)  # Ensure non-negative
    
    def analyze_enhanced_content_quality(self, text: str) -> float:
        """Enhanced language-agnostic content analysis with East Asian support."""
        try:
            score = 0.0
            
            # Detect script type for better analysis
            script_type = self.detect_script_type(text)
            
            if script_type in ['latin', 'cyrillic']:
                # Western language analysis
                word_count = len(text.split())
                if 2 <= word_count <= 10:
                    score += 6
                elif word_count <= 15:
                    score += 4
                    
            elif script_type in ['cjk', 'thai']:
                # East Asian language analysis (character-based)
                char_count = len([c for c in text if not c.isspace()])
                if 2 <= char_count <= 20:  # Shorter for character-based languages
                    score += 6
                elif char_count <= 40:
                    score += 4
            
            # Universal capitalization patterns (works for mixed scripts)
            if text.isupper() and len(text) < 50:
                score += 3
            
            # Enhanced punctuation analysis with East Asian punctuation
            east_asian_periods = ['。', '．', '｡']
            east_asian_commas = ['，', '、', 'ヽ']
            
            # Headings usually don't end with periods
            if not any(text.endswith(p) for p in ['.', ',', ';'] + east_asian_periods + east_asian_commas):
                score += 3
                
            # Colons often indicate headings (including full-width)
            if text.endswith(':') or text.endswith('：'):
                score += 2
                
            # Number patterns (good for structured headings)
            if re.search(r'[\d一二三四五六七八九十]', text):
                score += 2
            
            # Stop word analysis (if available and for applicable scripts)
            if self.stop_words and script_type in ['latin', 'cjk', 'thai']:
                if script_type == 'latin':
                    words = text.lower().split()
                    stop_ratio = sum(1 for w in words if w in self.stop_words) / max(len(words), 1)
                else:
                    # For East Asian, check character-level stop words
                    chars = list(text)
                    stop_ratio = sum(1 for c in chars if c in self.stop_words) / max(len(chars), 1)
                
                if stop_ratio < 0.3:  # Headings have fewer stop words
                    score += 3
            
            return min(score, 12)
        except:
            return 0
    
    def matches_heading_pattern(self, text: str) -> bool:
        """Check if text matches common heading patterns."""
        return self.decision_tree.detect_language_patterns(text) > 0.5
    
    def merge_multiline_headings(self, candidates: List[TextSpan]) -> List[TextSpan]:
        """Enhanced multiline heading merging with East Asian support."""
        merged = []
        i = 0
        
        while i < len(candidates):
            current = candidates[i]
            merged_spans = [current]
            
            # Look for continuations on subsequent lines
            j = i + 1
            while j < len(candidates):
                next_span = candidates[j]
                
                # Enhanced continuation detection
                if (next_span.page_num == current.page_num and
                    abs(next_span.x_center - current.x_center) < 60 and
                    next_span.y_position - current.bbox[3] < current.line_height * 2 and
                    abs(next_span.font_size - current.font_size) < 1.5 and
                    next_span.is_bold == current.is_bold):
                    
                    # Additional check for script consistency
                    current_script = self.detect_script_type(current.text)
                    next_script = self.detect_script_type(next_span.text)
                    
                    if current_script == next_script or current_script == 'mixed' or next_script == 'mixed':
                        merged_spans.append(next_span)
                        current = next_span  # Update for next iteration
                        j += 1
                    else:
                        break
                else:
                    break
            
            # Create merged span
            if len(merged_spans) > 1:
                # Use appropriate separator based on script type
                script_type = self.detect_script_type(merged_spans[0].text)
                separator = " " if script_type in ['latin', 'arabic'] else ""
                
                merged_text = separator.join(span.text for span in merged_spans)
                merged_span = TextSpan(
                    text=merged_text,
                    font_name=merged_spans[0].font_name,
                    font_size=merged_spans[0].font_size,
                    is_bold=merged_spans[0].is_bold,
                    is_italic=merged_spans[0].is_italic,
                    bbox=merged_spans[0].bbox,
                    page_num=merged_spans[0].page_num,
                    x_center=merged_spans[0].x_center,
                    y_position=merged_spans[0].y_position,
                    line_height=merged_spans[0].line_height
                )
                merged.append(merged_span)
                i = j
            else:
                merged.append(current)
                i += 1
        
        return merged
    
    def extract_title(self, doc: fitz.Document) -> str:
        """Extract document title from first page with enhanced multilingual logic."""
        if not doc:
            return "Untitled"
        
        try:
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
            hierarchy, font_stats = self.analyze_advanced_font_hierarchy(first_page_spans)
            
            # Score candidates for title (look in top 40% of page)
            title_candidates = []
            for span in first_page_spans:
                # Adjust length limits based on script type
                script_type = self.detect_script_type(span.text)
                max_length = 60 if script_type in ['cjk', 'thai'] else 100
                
                if (self.MIN_HEADING_LENGTH <= len(span.text) <= max_length and
                    span.y_position < page_height * 0.4):
                    
                    score = self.calculate_enhanced_hybrid_score(
                        span, hierarchy, font_stats['body_size'], page_width, page_height, first_page_spans)
                    title_candidates.append((span, score))
            
            if title_candidates:
                # Return highest scoring candidate
                best_candidate = max(title_candidates, key=lambda x: x[1])
                return best_candidate[0].text
            
            return "Untitled"
        except:
            return "Untitled"
    
    def extract_headings(self, doc: fitz.Document) -> List[Dict]:
        """Extract all headings using enhanced hybrid approach with multilingual support."""
        spans = self.extract_text_spans(doc)
        if not spans:
            return []
        
        # Update processing stats
        self.processing_stats['total_spans_processed'] = len(spans)
        
        # Batch process for performance
        spans = self.batch_process_spans(spans)
        
        # Analyze font hierarchy with advanced statistics
        hierarchy, font_stats = self.analyze_advanced_font_hierarchy(spans)
        
        # Filter potential heading candidates
        candidates = []
        for span in spans:
            if (self.MIN_HEADING_LENGTH <= len(span.text) <= self.MAX_HEADING_LENGTH and
                span.font_size >= font_stats['body_size'] * self.MIN_FONT_SIZE_RATIO):
                
                try:
                    page = doc[span.page_num - 1]
                    page_width = page.rect.width
                    page_height = page.rect.height
                    
                    # Use enhanced hybrid scoring
                    score = self.calculate_enhanced_hybrid_score(
                        span, hierarchy, font_stats['body_size'], page_width, page_height, spans)
                    
                    # Adaptive threshold based on script type
                    script_type = self.detect_script_type(span.text)
                    threshold = 25 if script_type in ['cjk', 'thai'] else 30
                    
                    if score > threshold:
                        candidates.append((span, score))
                except:
                    continue  # Skip problematic spans
        
        # Sort by score and select top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_spans = [candidate[0] for candidate in candidates[:80]]  # Increased for multilingual
        
        # Enhanced multiline heading merging
        merged_spans = self.merge_multiline_headings(selected_spans)
        
        # Convert to competition format (outline array)
        outline = []
        candidate_dict = {id(span): score for span, score in candidates}
        
        for span in merged_spans:
            level = hierarchy.get(round(span.font_size, 1), 6)
            
            outline.append({
                "level": f"H{min(level, 6)}",
                "text": span.text,
                "page": span.page_num
            })
        
        # Sort by page and maintain document order
        outline.sort(key=lambda x: x["page"])
        
        # Update stats
        self.processing_stats['headings_found'] = len(outline)
        
        return outline
    
    def robust_process_pdf(self, filepath: str) -> Dict:
        """Enhanced PDF processing with competition-compliant output format."""
        start_time = __import__('time').time()
        
        try:
            # Validate file
            if not os.path.exists(filepath):
                return {"status": "error", "error": "File not found"}
            
            file_size = os.path.getsize(filepath)
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return {"status": "error", "error": "File too large"}
            
            if file_size == 0:
                return {"status": "error", "error": "Empty file"}
            
            doc = fitz.open(filepath)
            
            # Handle corrupted or empty PDFs
            if len(doc) == 0:
                doc.close()
                return {"status": "error", "error": "Empty or corrupted PDF"}
            
            # Process with timeout protection (8 seconds for 50-page requirement)
            def timeout_handler(signum, frame):
                raise TimeoutError("Processing timeout")
            
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(8)  # 8-second timeout
                
                title = self.extract_title(doc)
                outline = self.extract_headings(doc)  # Returns outline format
                
                processing_time = __import__('time').time() - start_time
                
                # Competition-compliant format
                result = {
                    "title": title,
                    "outline": outline
                }
                
            finally:
                signal.alarm(0)  # Cancel timeout
                doc.close()
                
            return result
            
        except TimeoutError:
            return {"status": "error", "error": "Processing timeout (>8 seconds)"}
        except Exception as e:
            return {"status": "error", "error": f"Processing failed: {str(e)[:200]}"}

def main():
    """Enhanced main function with comprehensive multilingual support."""
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
    
    print(f"Processing {len(pdf_files)} PDF files with enhanced multilingual classifier...")
    print(f"Supported languages: English, Spanish, French, German, Portuguese, Italian,")
    print(f"Chinese, Japanese, Korean, Thai, Vietnamese, Hindi, Arabic")
    print(f"Competition constraints: ≤10s per 50-page PDF, ≤200MB model, CPU-only")
    print("-" * 60)
    
    success_count = 0
    total_headings = 0
    
    for i, filename in enumerate(pdf_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + '.json'
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[{i}/{len(pdf_files)}] Processing {filename}...")
        
        try:
            result = extractor.robust_process_pdf(input_path)
            
            # Only save successful results in competition format
            if result.get("status") != "error":
                # Clean result for competition (remove error handling fields)
                clean_result = {
                    "title": result["title"],
                    "outline": result["outline"]
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(clean_result, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                total_headings += len(result["outline"])
                
                print(f"  ✓ Generated {output_filename}")
                print(f"  ✓ Title: {result['title'][:50]}{'...' if len(result['title']) > 50 else ''}")
                print(f"  ✓ Found {len(result['outline'])} headings")
            else:
                print(f"  ✗ Error: {result['error']}")
                
        except Exception as e:
            print(f"  ✗ Failed to process {filename}: {e}")
    
    print("-" * 60)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(pdf_files)} files")
    print(f"Total headings extracted: {total_headings}")
    print(f"Multilingual support: Full East Asian + Western language coverage")

if __name__ == "__main__":
    main()
