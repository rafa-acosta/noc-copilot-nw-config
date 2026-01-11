import re
import uuid
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Import logger
from utils import logger, compute_file_hash

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

# Supported Vendors/OS (for detection logic)
VENDORS = {
    "cisco": ["ios", "ios-xe", "nx-os"],
    "aruba": ["aos-cx", "aos-s"]
}

# Regex for detecting potential OS/Vendor
PATTERNS = {
    "cisco_ios": r"Current configuration :",
    "aruba_aoscx": r"Current configuration:"
}

# Redaction Patterns
SECRETS_PATTERNS = [
    # Cisco Type 7 Passwords
    (r'(password|secret) 7 [a-zA-Z0-9]+', r'\1 7 [REDACTED]'),
    # Cisco Type 5 Passwords
    (r'(password|secret) 5 [a-zA-Z0-9$]+', r'\1 5 [REDACTED]'),
    # SNMP Communities
    (r'(snmp-server community) [a-zA-Z0-9]+', r'\1 [REDACTED]'),
    # TACACS/RADIUS Keys
    (r'(key) [a-zA-Z0-9]+', r'\1 [REDACTED]'),
    (r'(key 7) [a-zA-Z0-9]+', r'\1 [REDACTED]'),
]

# -----------------------------------------------------------------------------
# AST Parser
# -----------------------------------------------------------------------------

@dataclass
class ConfigBlock:
    full_text: str
    parent_line: str
    header_type: str
    children: List[str]
    line_start: int
    line_end: int
    has_secret: bool

class NetworkConfigParser:
    """
    Parses network configuration files into logical blocks (AST-like).
    Handles indentation-based structures common in Cisco/Aruba.
    """

    def __init__(self, content: str, filename: str):
        self.raw_content = content
        self.filename = filename
        self.lines = content.splitlines()
        self.blocks: List[ConfigBlock] = []
        self.metadata: Dict[str, Any] = self._detect_metadata()

    def _detect_metadata(self) -> Dict[str, Any]:
        """Auto-detects vendor, OS, and hostname."""
        meta = {
            "vendor": "unknown",
            "os_family": "unknown",
            "hostname": "unknown",
            "filename": self.filename
        }
        
        # Simple heuristics
        for line in self.lines[:50]: # Check first 50 lines
            if "Current configuration :" in line:
                meta["vendor"] = "cisco"
                meta["os_family"] = "sugg_ios_xe"
            if "version " in line.lower():
                # Extract version info if needed
                pass
            if line.startswith("hostname "):
                meta["hostname"] = line.split("hostname ")[1].strip()

        return meta

    def _is_child(self, line: str) -> bool:
        """Determines if a line is a child (indented or specialized)."""
        return line.startswith(" ") or line.startswith("\t")

    def _redact_line(self, line: str) -> (str, bool):
        """Redacts secrets from a single line."""
        found_secret = False
        redacted_line = line
        for pattern, replacement in SECRETS_PATTERNS:
            new_line = re.sub(pattern, replacement, redacted_line)
            if new_line != redacted_line:
                found_secret = True
                redacted_line = new_line
        return redacted_line, found_secret

    def parse(self) -> List[ConfigBlock]:
        """Main parsing logic."""
        current_block_lines = []
        current_parent = None
        block_start = 0
        has_secret_in_block = False

        for i, line in enumerate(self.lines):
            stripped = line.strip()
            
            # Skip empty lines or banners (simple filter)
            if not stripped or stripped.startswith("!"):
                continue

            # Redaction
            safe_line, detected_secret = self._redact_line(line)
            if detected_secret:
                has_secret_in_block = True

            if not self._is_child(line):
                # New Parent Block Found
                if current_parent:
                    # Save previous block
                    self._commit_block(current_parent, current_block_lines, block_start, i-1, has_secret_in_block)
                
                # Reset for new block
                current_parent = safe_line
                current_block_lines = [safe_line]
                block_start = i
                has_secret_in_block = detected_secret
            else:
                # Child line
                current_block_lines.append(safe_line)

        # Commit last block
        if current_parent:
             self._commit_block(current_parent, current_block_lines, block_start, len(self.lines), has_secret_in_block)

        return self.blocks

    def _commit_block(self, parent: str, lines: List[str], start: int, end: int, has_secret: bool):
        """Creates a ConfigBlock object."""
        # Determine basic type
        header_type = parent.split()[0] if parent else "global"
        
        full_text = "\n".join(lines)
        
        block = ConfigBlock(
            full_text=full_text,
            parent_line=parent,
            header_type=header_type,
            children=lines[1:],
            line_start=start,
            line_end=end,
            has_secret=has_secret
        )
        self.blocks.append(block)

# -----------------------------------------------------------------------------
# Vector Store Ingestion
# -----------------------------------------------------------------------------

class IngestionEngine:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="network_configs"
        )

    def process_file(self, file_path: str, extra_metadata: Dict[str, Any] = None):
        """Reads, parses, and indexes a file."""
        logger.info(f"Processing file: {file_path}")
        
        # 1. Read File
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return False

        # 2. Parse (AST)
        parser = NetworkConfigParser(content, file_path.split("/")[-1])
        blocks = parser.parse()
        file_meta = parser.metadata

        # 3. Convert to Documents
        documents = []
        for block in blocks:
            # Construct rich metadata
            meta = {
                "source": file_meta["filename"],
                "vendor": file_meta["vendor"],
                "hostname": file_meta["hostname"],
                "section_type": block.header_type,
                "line_start": block.line_start,
                "line_end": block.line_end,
                "has_secret": block.has_secret,
                "parent_line": block.parent_line
            }
            
            # Merge extra_metadata if provided
            if extra_metadata:
                meta.update(extra_metadata)
            
            # Create Document
            doc = Document(
                page_content=block.full_text,
                metadata=meta
            )
            documents.append(doc)

        logger.info(f"Generated {len(documents)} chunks for {file_path}")

        # 4. Filter existing (Simple deduplication logic could go here)
        # For now, we assume overwrite or append.
        
        # 5. Index
        if documents:
            self.vector_store.add_documents(documents)
            logger.info("Indexed successfully.")
            return True
        return False

    def get_retriever(self):
        """Returns the retriever interface."""
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}
        )

if __name__ == "__main__":
    # Test run
    pass
