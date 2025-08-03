import pandas as pd
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import yaml
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline"""
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 1000

    # Quality settings
    min_text_length: int = 10
    max_text_length: int = 10000
    duplicate_threshold: float = 0.95

    # Embedding settings
    embedding_model: str = 'all-MiniLM-L6-v2'
    batch_size: int = 32

    # File processing
    max_file_size_mb: int = 100
    supported_encodings: List[str] = None

    def __post_init__(self):
        if self.supported_encodings is None:
            self.supported_encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

@dataclass
class ChunkMetadata:
    """Enhanced metadata for each chunk"""
    source: str
    chunk_id: str
    location: str
    content_type: str  # 'ticket', 'manual', 'structured'
    domain: Optional[str] = None
    section: Optional[str] = None
    language: Optional[str] = None
    quality_score: float = 0.0
    word_count: int = 0
    sentence_count: int = 0
    created_at: str = None
    tags: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []

@dataclass
class ProcessedChunk:
    """A processed text chunk with enhanced metadata"""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[np.ndarray] = None
    hash: Optional[str] = None

    def __post_init__(self):
        if self.hash is None:
            self.hash = hashlib.md5(self.text.encode()).hexdigest()

class DataProcessor:
    """Enhanced data processing pipeline for KB conversion"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_hashes = set()
        self.stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'duplicates_removed': 0,
            'errors': 0
        }

    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            if len(text.strip()) < 10:
                return 'unknown'
            return detect(text)
        except LangDetectException:
            return 'unknown'

    def calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for text chunk"""
        if not text or len(text.strip()) < self.config.min_text_length:
            return 0.0

        score = 1.0

        # Length penalty for very short or very long texts
        length = len(text)
        if length < 50:
            score *= 0.5
        elif length > 2000:
            score *= 0.8

        # Check for meaningful content (not just punctuation/numbers)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.3:
            score *= 0.3

        # Check for sentence structure
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            score *= 0.2

        return min(score, 1.0)

    def extract_domain_from_content(self, text: str, source: str) -> str:
        """Extract domain information from content"""
        text_lower = text.lower()
        source_lower = source.lower()

        # Domain keywords
        hr_keywords = ['leave', 'salary', 'payroll', 'employee', 'hr', 'human resources', 'vacation', 'benefits']
        it_keywords = ['vpn', 'network', 'computer', 'software', 'hardware', 'it', 'technical', 'system', 'server']
        payroll_keywords = ['payroll', 'salary', 'payment', 'wage', 'compensation', 'tax', 'deduction']

        # Check source filename first
        if any(keyword in source_lower for keyword in ['hr', 'human']):
            return 'HR'
        elif any(keyword in source_lower for keyword in ['it', 'technical', 'support']):
            return 'IT'
        elif any(keyword in source_lower for keyword in ['payroll', 'salary']):
            return 'Payroll'

        # Check content
        hr_score = sum(1 for keyword in hr_keywords if keyword in text_lower)
        it_score = sum(1 for keyword in it_keywords if keyword in text_lower)
        payroll_score = sum(1 for keyword in payroll_keywords if keyword in text_lower)

        if hr_score > it_score and hr_score > payroll_score:
            return 'HR'
        elif it_score > payroll_score:
            return 'IT'
        elif payroll_score > 0:
            return 'Payroll'

        return 'General'

    def smart_chunk_text(self, text: str, content_type: str) -> List[str]:
        """Intelligent text chunking based on content type"""
        if not text or len(text.strip()) < self.config.min_chunk_size:
            return []

        chunks = []

        if content_type == 'manual':
            # For manuals, try to preserve paragraph structure
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.config.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"

            if current_chunk:
                chunks.append(current_chunk.strip())

        elif content_type == 'ticket':
            # For tickets, try to preserve sentence structure
            sentences = sent_tokenize(text)
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.config.chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "

            if current_chunk:
                chunks.append(current_chunk.strip())

        else:
            # Default chunking with overlap
            words = text.split()
            chunk_words = self.config.chunk_size // 5  # Approximate words per chunk
            overlap_words = self.config.chunk_overlap // 5

            for i in range(0, len(words), chunk_words - overlap_words):
                chunk = " ".join(words[i:i + chunk_words])
                if len(chunk) >= self.config.min_chunk_size:
                    chunks.append(chunk)

        return [chunk for chunk in chunks if len(chunk) >= self.config.min_chunk_size]

    def process_docx_file(self, file_path: str) -> List[ProcessedChunk]:
        """Process DOCX files with enhanced metadata extraction"""
        chunks = []
        try:
            doc = docx.Document(file_path)
            file_name = os.path.basename(file_path)
            current_heading = "General"
            full_text = ""

            # Extract all text first
            for para in doc.paragraphs:
                if para.runs and para.runs[0].bold and para.text.strip():
                    current_heading = para.text.strip()

                if para.text.strip():
                    full_text += para.text + "\n\n"

            # Smart chunking
            text_chunks = self.smart_chunk_text(full_text, 'manual')

            for i, chunk_text in enumerate(text_chunks):
                if self.is_duplicate(chunk_text):
                    continue

                chunk_id = f"{file_name}_{i}"
                language = self.detect_language(chunk_text)
                quality_score = self.calculate_quality_score(chunk_text)
                domain = self.extract_domain_from_content(chunk_text, file_name)

                metadata = ChunkMetadata(
                    source=file_name,
                    chunk_id=chunk_id,
                    location=f"chunk_{i}",
                    content_type='manual',
                    domain=domain,
                    section=current_heading,
                    language=language,
                    quality_score=quality_score,
                    word_count=len(chunk_text.split()),
                    sentence_count=len(sent_tokenize(chunk_text))
                )

                chunk = ProcessedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=metadata
                )

                chunks.append(chunk)
                self.processed_hashes.add(chunk.hash)

            logger.info(f"Processed {file_name}: {len(chunks)} chunks created")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['errors'] += 1

        return chunks

    def process_csv_file(self, file_path: str) -> List[ProcessedChunk]:
        """Process CSV files with enhanced data handling"""
        chunks = []
        try:
            file_name = os.path.basename(file_path)
            df = None

            # Try different encodings
            for encoding in self.config.supported_encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                    logger.info(f"Successfully read {file_name} with {encoding} encoding")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue

            if df is None:
                logger.error(f"Could not read {file_name} with any supported encoding")
                return chunks

            # Identify text columns
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            df = df.dropna(subset=text_columns, how='all')

            for i, row in df.iterrows():
                # Create comprehensive text from row
                text_parts = []
                for col in text_columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        text_parts.append(f"{col.replace('_', ' ').title()}: {row[col]}")

                if not text_parts:
                    continue

                chunk_text = ". ".join(text_parts)

                if self.is_duplicate(chunk_text):
                    continue

                chunk_id = f"{file_name}_row_{i}"
                language = self.detect_language(chunk_text)
                quality_score = self.calculate_quality_score(chunk_text)
                domain = self.extract_domain_from_content(chunk_text, file_name)

                # Extract additional metadata from row
                tags = []
                if 'type' in row and pd.notna(row['type']):
                    tags.append(str(row['type']).lower())
                if 'priority' in row and pd.notna(row['priority']):
                    tags.append(f"priority_{str(row['priority']).lower()}")

                metadata = ChunkMetadata(
                    source=file_name,
                    chunk_id=chunk_id,
                    location=f"row_{i}",
                    content_type='ticket',
                    domain=domain,
                    language=language,
                    quality_score=quality_score,
                    word_count=len(chunk_text.split()),
                    sentence_count=len(sent_tokenize(chunk_text)),
                    tags=tags
                )

                chunk = ProcessedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=metadata
                )

                chunks.append(chunk)
                self.processed_hashes.add(chunk.hash)

            logger.info(f"Processed {file_name}: {len(chunks)} chunks created from {len(df)} rows")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['errors'] += 1

        return chunks

    def process_xlsx_file(self, file_path: str) -> List[ProcessedChunk]:
        """Process XLSX files similar to CSV but with Excel-specific handling"""
        chunks = []
        try:
            file_name = os.path.basename(file_path)
            df = pd.read_excel(file_path)

            # Similar processing to CSV
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            df = df.dropna(subset=text_columns, how='all')

            for i, row in df.iterrows():
                text_parts = []
                for col in text_columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        text_parts.append(f"{col.replace('_', ' ').title()}: {row[col]}")

                if not text_parts:
                    continue

                chunk_text = ". ".join(text_parts)

                if self.is_duplicate(chunk_text):
                    continue

                chunk_id = f"{file_name}_row_{i}"
                language = self.detect_language(chunk_text)
                quality_score = self.calculate_quality_score(chunk_text)
                domain = self.extract_domain_from_content(chunk_text, file_name)

                metadata = ChunkMetadata(
                    source=file_name,
                    chunk_id=chunk_id,
                    location=f"row_{i}",
                    content_type='structured',
                    domain=domain,
                    language=language,
                    quality_score=quality_score,
                    word_count=len(chunk_text.split()),
                    sentence_count=len(sent_tokenize(chunk_text))
                )

                chunk = ProcessedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=metadata
                )

                chunks.append(chunk)
                self.processed_hashes.add(chunk.hash)

            logger.info(f"Processed {file_name}: {len(chunks)} chunks created")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['errors'] += 1

        return chunks

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate based on hash"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.processed_hashes:
            self.stats['duplicates_removed'] += 1
            return True
        return False

    def process_file(self, file_path: str) -> List[ProcessedChunk]:
        """Process a single file based on its extension"""
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return []

        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            logger.warning(f"File too large ({file_size_mb:.1f}MB): {file_path}")
            return []

        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.docx':
            return self.process_docx_file(file_path)
        elif file_ext == '.csv':
            return self.process_csv_file(file_path)
        elif file_ext == '.xlsx':
            return self.process_xlsx_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []

    def process_files(self, file_paths: List[str]) -> List[ProcessedChunk]:
        """Process multiple files and return all chunks"""
        all_chunks = []

        for file_path in file_paths:
            logger.info(f"Processing file: {file_path}")
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)
            self.stats['files_processed'] += 1

        self.stats['chunks_created'] = len(all_chunks)

        # Filter by quality score
        quality_filtered = [chunk for chunk in all_chunks if chunk.metadata.quality_score > 0.3]

        logger.info(f"Processing complete: {self.stats}")
        logger.info(f"Quality filtered: {len(quality_filtered)}/{len(all_chunks)} chunks retained")

        return quality_filtered

class EnhancedKnowledgeBaseBuilder:
    """Enhanced knowledge base builder with improved processing and indexing"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processor = DataProcessor(config)
        self.model = None

    def load_embedding_model(self):
        """Load the embedding model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
        return self.model

    def generate_embeddings(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """Generate embeddings for chunks with batch processing"""
        if not chunks:
            return chunks

        model = self.load_embedding_model()
        texts = [chunk.text for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.config.batch_size,
            convert_to_numpy=True
        )

        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.astype('float32')

        return chunks

    def create_faiss_index(self, chunks: List[ProcessedChunk]) -> Tuple[faiss.Index, Dict[str, Dict]]:
        """Create FAISS index with enhanced mapping"""
        if not chunks or not chunks[0].embedding is not None:
            raise ValueError("Chunks must have embeddings before creating index")

        # Create index
        dimension = chunks[0].embedding.shape[0]
        index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(index)

        # Prepare data
        embeddings = np.array([chunk.embedding for chunk in chunks])
        chunk_ids = np.array([hash(chunk.chunk_id) % (2**31) for chunk in chunks])  # Convert to int32

        # Add to index
        index.add_with_ids(embeddings, chunk_ids)

        # Create enhanced mapping
        id_to_chunk = {}
        for chunk, chunk_id in zip(chunks, chunk_ids):
            id_to_chunk[str(chunk_id)] = {
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                'metadata': asdict(chunk.metadata),
                'hash': chunk.hash
            }

        logger.info(f"Created FAISS index with {index.ntotal} vectors")
        return index, id_to_chunk

    def save_knowledge_base(self, index: faiss.Index, mapping: Dict, index_path: str, mapping_path: str):
        """Save the knowledge base to disk"""
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)

        logger.info(f"Saving mapping to {mapping_path}")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Knowledge base saved with {index.ntotal} vectors")

    def build_knowledge_base(self, file_paths: List[str], index_path: str, mapping_path: str):
        """Complete pipeline to build enhanced knowledge base"""
        logger.info("Starting enhanced knowledge base creation...")

        # Process files
        chunks = self.processor.process_files(file_paths)
        if not chunks:
            logger.error("No chunks were created. Aborting knowledge base creation.")
            return

        # Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)

        # Create index
        index, mapping = self.create_faiss_index(chunks_with_embeddings)

        # Save to disk
        self.save_knowledge_base(index, mapping, index_path, mapping_path)

        # Print statistics
        stats = self.processor.stats
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   Files processed: {stats['files_processed']}")
        logger.info(f"   Chunks created: {stats['chunks_created']}")
        logger.info(f"   Duplicates removed: {stats['duplicates_removed']}")
        logger.info(f"   Errors encountered: {stats['errors']}")

        return index, mapping

def create_default_config() -> ProcessingConfig:
    """Create default configuration for processing"""
    return ProcessingConfig(
        chunk_size=512,
        chunk_overlap=50,
        min_chunk_size=50,
        embedding_model='all-MiniLM-L6-v2',
        batch_size=32
    )

def main():
    """Main function to build the enhanced knowledge base"""
    # Configuration
    config = create_default_config()

    # File paths (relative to dataset directory)
    all_files = [
        "NexaCorp HR Manual.docx",
        "NexaCorp IT Support Manual.docx",
        "NexaCorp Payroll Support Manual.docx",
        "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
        "dataset-tickets-german_normalized.csv",
        "dataset-tickets-german_normalized_50_5_2.csv",
        "dataset-tickets-multi-lang3-4k.csv",
        "dataset-tickets-multi-lang-4-20k.csv",
        "nexacorp_tickets.xlsx"
    ]

    # Create builder
    builder = EnhancedKnowledgeBaseBuilder(config)

    # Build knowledge base
    try:
        builder.build_knowledge_base(
            file_paths=all_files,
            index_path="nexa_corp_enhanced.index",
            mapping_path="nexa_corp_enhanced_mapping.json"
        )
        logger.info("🎉 Enhanced knowledge base creation completed successfully!")

    except Exception as e:
        logger.error(f"Failed to create knowledge base: {e}")
        raise

if __name__ == '__main__':
    main()