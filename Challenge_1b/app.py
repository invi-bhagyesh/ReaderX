import json
import fitz  # PyMuPDF
import spacy
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import sys
import logging
from dataclasses import dataclass
from functools import lru_cache
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SectionInfo:
    """Data class for section information"""
    title: str
    content: str
    page: int
    font_size: float = 0.0
    is_bold: bool = False
    position: int = 0

class PersonaDrivenDocumentAnalyzer:
    def __init__(self):
        """Initialize the document analyzer with required models"""
        self._initialize_models()
        self._section_cache = {}
        self._embedding_cache = {}
        
    def _initialize_models(self):
        """Initialize NLP models with error handling"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.error("SpaCy English model not found. Please install: python -m spacy download en_core_web_sm")
            sys.exit(1)
        
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {str(e)}")
            sys.exit(1)
    
    @lru_cache(maxsize=100)
    def _get_embedding(self, text: str) -> np.ndarray:
        """Cache embeddings to avoid recomputation"""
        return self.sentence_model.encode([text])[0]
    
    def extract_pdf_content(self, filename: str) -> Dict[str, Any]:
        """Enhanced PDF content extraction with better structure detection"""
        if not os.path.exists(filename):
            logger.error(f"File not found: {filename}")
            return {"sections": [], "full_text": ""}
            
        try:
            doc = fitz.open(filename)
            sections = []
            full_text = ""
            page_texts = []
            
            logger.info(f"Processing {len(doc)} pages from {filename}")
            
            # First pass: collect all text and potential headers
            potential_headers = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += text + "\n"
                page_texts.append(text)
                
                # Enhanced header detection
                headers = self._extract_headers_from_page(page, page_num + 1)
                potential_headers.extend(headers)
            
            doc.close()
            
            # Filter and validate headers
            valid_headers = self._validate_headers(potential_headers)
            
            if valid_headers:
                sections = self._extract_content_for_headers(full_text, valid_headers, page_texts)
            else:
                sections = self._create_intelligent_sections(full_text, page_texts)
            
            logger.info(f"Extracted {len(sections)} sections from {filename}")
            
            return {
                "sections": sections,
                "full_text": full_text.strip()
            }
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            return {"sections": [], "full_text": ""}
    
    def _extract_headers_from_page(self, page, page_num: int) -> List[SectionInfo]:
        """Extract potential headers with enhanced formatting analysis"""
        headers = []
        blocks = page.get_text("dict")
        
        for block_idx, block in enumerate(blocks.get("blocks", [])):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text_content = span.get("text", "").strip()
                    font_size = span.get("size", 0)
                    font_flags = span.get("flags", 0)
                    
                    if self._is_potential_header(text_content, font_size, font_flags):
                        headers.append(SectionInfo(
                            title=text_content,
                            content="",
                            page=page_num,
                            font_size=font_size,
                            is_bold=bool(font_flags & 2**4),
                            position=block_idx
                        ))
        
        return headers
    
    def _is_potential_header(self, text: str, font_size: float, font_flags: int) -> bool:
        """Enhanced header detection with multiple criteria"""
        if not text or len(text.strip()) < 3:
            return False
            
        # Size-based detection
        if font_size > 12:
            return True
            
        # Bold text detection
        if font_flags & 2**4:  # Bold flag
            return True
            
        # Pattern-based detection
        return self._is_section_header(text)
    
    def _is_section_header(self, text: str) -> bool:
        """Enhanced section header pattern matching"""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Academic paper sections
        academic_patterns = [
            r'^(abstract|introduction|literature\s+review|methodology|methods|results|discussion|conclusion|references|bibliography)$',
            r'^(related\s+work|background|experimental\s+setup|evaluation|future\s+work)$',
        ]
        
        # Numbered sections
        numbered_patterns = [
            r'^\d+\.?\s+[A-Z][a-zA-Z\s]+',
            r'^(\d+\.)*\d+\s+[A-Z][a-zA-Z\s]+',
            r'^Chapter\s+\d+',
            r'^Section\s+\d+',
        ]
        
        # All caps (but reasonable length)
        if (text_clean.isupper() and 5 <= len(text_clean) <= 50 and 
            not any(char.isdigit() for char in text_clean)):
            return True
        
        # Check patterns
        all_patterns = academic_patterns + numbered_patterns
        return any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in all_patterns)
    
    def _validate_headers(self, headers: List[SectionInfo]) -> List[SectionInfo]:
        """Filter out false positive headers"""
        if not headers:
            return []
        
        # Remove duplicates and very short headers
        unique_headers = []
        seen_titles = set()
        
        for header in headers:
            title_clean = header.title.strip().lower()
            if (title_clean not in seen_titles and 
                len(title_clean) >= 3 and 
                len(title_clean) <= 100):
                unique_headers.append(header)
                seen_titles.add(title_clean)
        
        # Sort by page and position
        unique_headers.sort(key=lambda x: (x.page, x.position))
        
        return unique_headers
    
    def _extract_content_for_headers(self, full_text: str, headers: List[SectionInfo], page_texts: List[str]) -> List[Dict]:
        """Extract content between headers more accurately"""
        sections = []
        text_parts = full_text.split('\n')
        
        for i, header in enumerate(headers):
            content_lines = []
            header_found = False
            next_header_title = headers[i + 1].title.lower() if i + 1 < len(headers) else None
            
            for line in text_parts:
                line_clean = line.strip()
                
                # Start collecting after finding header
                if header.title.lower() in line_clean.lower():
                    header_found = True
                    continue
                
                # Stop at next header
                if (header_found and next_header_title and 
                    next_header_title in line_clean.lower()):
                    break
                
                # Collect content
                if header_found and line_clean:
                    content_lines.append(line_clean)
                    
                    # Limit content length
                    if len('\n'.join(content_lines)) > 2000:
                        break
            
            content = '\n'.join(content_lines).strip()
            if content:  # Only add sections with content
                sections.append({
                    "title": header.title,
                    "page": header.page,
                    "content": content
                })
        
        return sections
    
    def _create_intelligent_sections(self, full_text: str, page_texts: List[str]) -> List[Dict]:
        """Create sections using intelligent text segmentation"""
        # Try paragraph-based segmentation first
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 100]
        
        if len(paragraphs) >= 3:
            sections = []
            for i, para in enumerate(paragraphs[:8]):  # Limit to 8 sections
                sections.append({
                    "title": f"Section {i+1}",
                    "page": min(i + 1, len(page_texts)),
                    "content": para
                })
            return sections
        
        # Fallback to page-based segmentation
        sections = []
        for i, page_text in enumerate(page_texts[:5]):  # Limit to 5 pages
            if len(page_text.strip()) > 200:
                sections.append({
                    "title": f"Page {i+1}",
                    "page": i + 1,
                    "content": page_text.strip()[:1500]  # Limit content
                })
        
        return sections
    
    def calculate_relevance_score(self, section: Dict, persona: str, job_task: str, doc_title: str) -> float:
        """Enhanced relevance scoring with caching"""
        try:
            # Create text representations
            section_text = f"{section['title']} {section['content'][:500]}"  # Limit for performance
            persona_job_text = f"{persona} {job_task} {doc_title}"
            
            # Get cached embeddings
            section_embedding = self._get_embedding(section_text)
            persona_job_embedding = self._get_embedding(persona_job_text)
            
            # Semantic similarity (45%)
            semantic_score = cosine_similarity(
                section_embedding.reshape(1, -1), 
                persona_job_embedding.reshape(1, -1)
            )[0][0]
            
            # Keyword overlap (25%)
            keyword_score = self._calculate_keyword_overlap(section_text, persona_job_text)
            
            # Content type relevance (20%)
            content_type_score = self._classify_content_relevance(section, job_task)
            
            # Structural importance (10%)
            structural_score = self._calculate_structural_importance(section)
            
            total_score = (semantic_score * 0.45 + 
                          keyword_score * 0.25 + 
                          content_type_score * 0.20 + 
                          structural_score * 0.10)
            
            return float(np.clip(total_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating relevance score: {str(e)}")
            return 0.0
    
    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """Enhanced keyword overlap with TF-IDF weighting"""
        try:
            doc1 = self.nlp(text1.lower()[:1000])  # Limit for performance
            doc2 = self.nlp(text2.lower()[:1000])
            
            # Extract important tokens with POS filtering
            def extract_keywords(doc):
                return set([
                    token.lemma_ for token in doc 
                    if (not token.is_stop and not token.is_punct and 
                        token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                        len(token.text) > 2)
                ])
            
            tokens1 = extract_keywords(doc1)
            tokens2 = extract_keywords(doc2)
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error in keyword overlap calculation: {str(e)}")
            return 0.0
    
    def _classify_content_relevance(self, section: Dict, job_task: str) -> float:
        """Enhanced content classification with expanded indicators"""
        content = section["content"].lower()
        title = section["title"].lower()
        job_lower = job_task.lower()
        
        # Expanded content indicators
        content_indicators = {
            'methodology': ["method", "approach", "algorithm", "technique", "procedure", "framework", "model"],
            'results': ["result", "finding", "outcome", "performance", "evaluation", "experiment", "data"],
            'analysis': ["analysis", "discussion", "interpretation", "implication", "conclusion", "insight"],
            'background': ["background", "literature", "review", "related", "previous", "study"],
            'implementation': ["implementation", "system", "design", "architecture", "development"]
        }
        
        score = 0.0
        text_combined = title + " " + content
        
        # Match job requirements to content types
        for category, indicators in content_indicators.items():
            if any(keyword in job_lower for keyword in indicators):
                if any(indicator in text_combined for indicator in indicators):
                    score += 0.6
        
        # Bonus for exact matches
        job_words = set(job_lower.split())
        content_words = set(text_combined.split())
        exact_matches = len(job_words.intersection(content_words))
        score += min(exact_matches * 0.1, 0.4)
        
        return min(score, 1.0)
    
    def _calculate_structural_importance(self, section: Dict) -> float:
        """Enhanced structural importance calculation"""
        title = section["title"].lower()
        content_length = len(section["content"])
        
        # Critical sections (higher weight)
        critical_sections = ["abstract", "introduction", "conclusion", "summary"]
        important_sections = ["results", "methodology", "methods", "discussion", "findings"]
        
        score = 0.2  # Base score
        
        # Section type bonus
        if any(crit in title for crit in critical_sections):
            score += 0.6
        elif any(imp in title for imp in important_sections):
            score += 0.4
        
        # Content length consideration
        if 200 <= content_length <= 1500:  # Optimal length range
            score += 0.3
        elif content_length > 1500:
            score += 0.1
        
        # Position bonus (earlier sections often more important)
        if section.get("page", 1) <= 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def rank_sections_across_documents(self, processed_docs: List[Dict], persona: str, job_task: str) -> List[Dict]:
        """Enhanced section ranking with diversity consideration"""
        all_sections = []
        
        logger.info(f"Ranking sections across {len(processed_docs)} documents")
        
        for doc in processed_docs:
            for section in doc["sections"]:
                if len(section["content"].strip()) < 50:
                    continue
                    
                relevance_score = self.calculate_relevance_score(
                    section, persona, job_task, doc["title"]
                )
                
                all_sections.append({
                    "document": doc["filename"],
                    "section_title": section["title"],
                    "page_number": section["page"],
                    "relevance_score": relevance_score,
                    "content": section["content"],
                    "doc_title": doc["title"]
                })
        
        # Sort by relevance score
        all_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Ensure diversity across documents
        all_sections = self._ensure_document_diversity(all_sections)
        
        # Assign importance ranks
        for i, section in enumerate(all_sections):
            section["importance_rank"] = i + 1
        
        logger.info(f"Ranked {len(all_sections)} sections")
        return all_sections
    
    def _ensure_document_diversity(self, sections: List[Dict], max_per_doc: int = 3) -> List[Dict]:
        """Ensure diverse representation across documents"""
        doc_counts = {}
        diverse_sections = []
        
        for section in sections:
            doc_name = section["document"]
            current_count = doc_counts.get(doc_name, 0)
            
            if current_count < max_per_doc:
                diverse_sections.append(section)
                doc_counts[doc_name] = current_count + 1
        
        # Fill remaining slots with best remaining sections
        remaining_slots = min(10 - len(diverse_sections), len(sections) - len(diverse_sections))
        for section in sections:
            if section not in diverse_sections and remaining_slots > 0:
                diverse_sections.append(section)
                remaining_slots -= 1
        
        return diverse_sections
    
    def analyze_subsections(self, top_sections: List[Dict], persona: str, job_task: str, max_sections: int = 5) -> List[Dict]:
        """Enhanced subsection analysis with better content refinement"""
        subsection_analysis = []
        
        for section in top_sections[:max_sections]:
            refined_content = self._advanced_content_refinement(
                section["content"], persona, job_task
            )
            
            if refined_content:
                subsection_analysis.append({
                    "document": section["document"],
                    "refined_text": refined_content,
                    "page_number": section["page_number"]
                })
        
        return subsection_analysis[:12]  # Increased limit
    
    def _advanced_content_refinement(self, content: str, persona: str, job_task: str) -> str:
        """Advanced content refinement using NLP"""
        try:
            # Split into sentences
            doc = self.nlp(content[:1000])  # Limit for performance
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
            
            if not sentences:
                return content[:200] + '...' if len(content) > 200 else content
            
            # Score sentences based on relevance
            sentence_scores = []
            persona_job_embedding = self._get_embedding(f"{persona} {job_task}")
            
            for sentence in sentences:
                try:
                    sent_embedding = self._get_embedding(sentence)
                    similarity = cosine_similarity(
                        sent_embedding.reshape(1, -1),
                        persona_job_embedding.reshape(1, -1)
                    )[0][0]
                    sentence_scores.append((sentence, similarity))
                except:
                    sentence_scores.append((sentence, 0.0))
            
            # Sort by relevance and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent for sent, score in sentence_scores[:3]]
            
            refined_text = ' '.join(top_sentences)
            return refined_text if len(refined_text) > 50 else content[:300]
            
        except Exception as e:
            logger.warning(f"Error in content refinement: {str(e)}")
            return content[:200] + '...' if len(content) > 200 else content
    
    def process_challenge(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced main processing with better error handling and logging"""
        start_time = datetime.now()
        
        try:
            # Parse input with validation
            challenge_info = input_json.get("challenge_info", {})
            documents = input_json.get("documents", [])
            persona = input_json.get("persona", {}).get("role", "")
            job_to_be_done = input_json.get("job_to_be_done", {}).get("task", "")
            
            if not documents or not persona or not job_to_be_done:
                raise ValueError("Missing required input fields")
            
            logger.info(f"Processing {len(documents)} documents for persona: {persona}")
            logger.info(f"Job to be done: {job_to_be_done}")
            
            # Process documents with progress tracking
            processed_docs = []
            for i, doc_info in enumerate(documents):
                filename = doc_info.get("filename", "")
                title = doc_info.get("title", "")
                
                logger.info(f"Processing document {i+1}/{len(documents)}: {filename}")
                doc_content = self.extract_pdf_content(filename)
                
                processed_docs.append({
                    "filename": filename,
                    "title": title,
                    "sections": doc_content["sections"]
                })
            
            # Rank sections
            ranked_sections = self.rank_sections_across_documents(
                processed_docs, persona, job_to_be_done
            )
            
            # Analyze subsections
            subsection_analysis = self.analyze_subsections(
                ranked_sections, persona, job_to_be_done
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate enhanced output with metadata
            output = {
                "metadata": {
                    "input_documents": [doc["filename"] for doc in documents],
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat(),
                    "processing_time_seconds": round(processing_time, 2),
                    "total_sections_found": len(ranked_sections),
                    "documents_processed": len(processed_docs)
                },
                "extracted_sections": [
                    {
                        "document": section["document"],
                        "section_title": section["section_title"],
                        "importance_rank": section["importance_rank"],
                        "page_number": section["page_number"]
                    }
                    for section in ranked_sections[:15]  # Increased from 10
                ],
                "subsection_analysis": subsection_analysis
            }
            
            logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
            return output
            
        except Exception as e:
            error_msg = f"Error processing challenge: {str(e)}"
            logger.error(error_msg)
            return {
                "metadata": {
                    "error": error_msg,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": [],
                "subsection_analysis": []
            }

def main():
    """Enhanced main function with better argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Persona-Driven Document Intelligence System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python app.py -i input.json -o output.json
  python app.py --input /path/to/input.json --output /path/to/output.json
        """
    )
    parser.add_argument('--input', '-i', required=True, 
                       help='Input JSON file path')
    parser.add_argument('--output', '-o', required=True, 
                       help='Output JSON file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize analyzer
    logger.info("Initializing document analyzer...")
    analyzer = PersonaDrivenDocumentAnalyzer()
    
    # Load input
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        logger.info(f"Loaded input from {args.input}")
    except Exception as e:
        logger.error(f"Error loading input file: {str(e)}")
        return 1
    
    # Process challenge
    start_time = datetime.now()
    result = analyzer.process_challenge(input_data)
    end_time = datetime.now()
    
    processing_time = (end_time - start_time).total_seconds()
    
    # Save output
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        return 0
    except Exception as e:
        logger.error(f"Error saving output file: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
