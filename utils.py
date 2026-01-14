import logging
import sys
import hashlib
import re
import os

def setup_logging():
    """Configures the logging for the application."""
    # Ensure logs directory exists
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "app.log"))
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
