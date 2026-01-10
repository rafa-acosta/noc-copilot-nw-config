import logging
import sys
import hashlib
import re

def setup_logging():
    """Configures the logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("RAG_Chatbot")

logger = setup_logging()

def compute_file_hash(file_bytes):
    """Computes SHA256 hash of file content for deduplication."""
    return hashlib.sha256(file_bytes).hexdigest()

def clean_filename(filename):
    """Sanitizes filenames."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
