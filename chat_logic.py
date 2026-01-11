import time
from typing import List, Dict, Any, Tuple
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

from ingestion import IngestionEngine
from utils import logger

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.2:3b"

PROMPT_TEMPLATE = """
You are a Senior Network Engineer assistant. You answer questions STRICTLY based on the provided network configuration chunks.
You are deterministic and precise.

CONTEXT:
{context}

USER QUERY:
{question}

INSTRUCTIONS:
1. Answer the query using ONLY the information in the CONTEXT.
2. If the answer is not in the context, state: "Not found in the provided configuration."
3. Cite the exact configuration lines, section names, or file names for every fact.
4. Do not speculate or use outside knowledge.
5. Format the output as clean Markdown.

ANSWER:
"""

# -----------------------------------------------------------------------------
# Chat Module
# -----------------------------------------------------------------------------

class RAGChatbot:
    """
    Main RAG Class governing the interaction between UI, VectorDB, and LLM.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.ingestion = IngestionEngine()
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1, # Low temp for deterministic outputs
            keep_alive="5m"
        )
        self.retriever = self.ingestion.get_retriever()
        self.chain = self._build_chain()

    def _build_chain(self):
        """Constructs the LangChain QA pipeline."""
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        return chain

    def update_model(self, model_name: str):
        """Switches the backend LLM."""
        logger.info(f"Switching model to {model_name}")
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            keep_alive="5m"
        )
        # Rebuild chain with new LLM
        self.chain = self._build_chain()

    def process_file(self, file_path: str, extra_metadata: Dict[str, Any] = None) -> bool:
        """Wrapper to pass file ingestion to the engine."""
        return self.ingestion.process_file(file_path, extra_metadata)

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Main query method.
        Returns: {
            "result": str, # The LLM response
            "source_documents": List[Document], # Citations
            "model": str,
            "latency": float
        }
        """
        start_time = time.time()
        
        try:
            logger.info(f"Querying: {query}")
            response = self.chain.invoke({"query": query})
            
            end_time = time.time()
            latency = end_time - start_time
            
            return {
                "result": response["result"],
                "source_documents": response["source_documents"],
                "model": self.model_name,
                "latency": round(latency, 2)
            }
        except Exception as e:
            logger.error(f"Chat Error: {e}")
            return {
                "result": f"Error generating response: {str(e)}",
                "source_documents": [],
                "model": self.model_name,
                "latency": 0.0
            }

    def compare_configs(self, query: str, golden_filename: str = None, candidate_filename: str = None) -> Dict[str, Any]:
        """
        Specialized method for comparing two configurations.
        Manually retrieves context and segregates it by 'config_role' AND filename
        to prevent hallucination/pollution from previous files.
        """
        start_time = time.time()
        try:
            logger.info(f"Comparing: {query} (Golden: {golden_filename}, Candidate: {candidate_filename})")
            
            # 1. Start with Role Filters (Using $and syntax for multiple conditions if needed)
            
            # Construct Golden Filter
            if golden_filename:
                filter_golden = {
                    "$and": [
                        {"config_role": "golden"},
                        {"source": golden_filename}
                    ]
                }
            else:
                filter_golden = {"config_role": "golden"}

            # Construct Candidate Filter
            if candidate_filename:
                filter_candidate = {
                    "$and": [
                        {"config_role": "candidate"},
                        {"source": candidate_filename}
                    ]
                }
            else:
                filter_candidate = {"config_role": "candidate"}
                
            # 3. Dual Retrieval Strategy with Specific Filters
            
            # Retrieve Golden Context
            docs_golden = self.ingestion.vector_store.similarity_search(
                query, 
                k=10, 
                filter=filter_golden
            )
            
            # Retrieve Candidate Context
            docs_candidate = self.ingestion.vector_store.similarity_search(
                query, 
                k=10, 
                filter=filter_candidate
            )
            
            # Combine all docs for citation purposes
            docs = docs_golden + docs_candidate
            
            # 2. Segregate Context (Already separated by query, but format for prompt)
            golden_context = []
            for doc in docs_golden:
                content = f"[Line {doc.metadata.get('line_start')}] {doc.page_content}"
                golden_context.append(content)

            candidate_context = []
            for doc in docs_candidate:
                content = f"[Line {doc.metadata.get('line_start')}] {doc.page_content}"
                candidate_context.append(content)
            
            # 3. Build Structured Prompt
            golden_text = "\n".join(golden_context) if golden_context else "No Golden Config context found."
            candidate_text = "\n".join(candidate_context) if candidate_context else "No Candidate Config context found."
            
            prompt = f"""
You are a Senior Network Engineer assistant specializing in configuration auditing.
You are comparing a **Golden Configuration** (Reference) against a **Candidate Configuration** (Target).

CONTEXT FROM GOLDEN CONFIG:
{golden_text}

CONTEXT FROM CANDIDATE CONFIG:
{candidate_text}

USER INSTRUCTION:
{query}

STRICT RULES:
1. Compare ONLY the provided context.
2. If the Golden and Candidate contexts for a specific feature are IDENTICAL, you MUST report it as a MATCH.
3. Only report "MISSING" or "EXTRA" if there is an actual difference in the provided text.
4. Be precise.
            """
            
            # 4. Invoke LLM directly (bypass RetrievalQA chain to use our custom prompt)
            # using the same LLM instance
            response_msg = self.llm.invoke(prompt)
            result_text = response_msg.content
            
            end_time = time.time()
            latency = end_time - start_time
            
            return {
                "result": result_text,
                "source_documents": docs,
                "model": self.model_name,
                "latency": round(latency, 2)
            }
            
        except Exception as e:
            logger.error(f"Comparison Error: {e}")
            return {
                "result": f"Error generating comparison: {str(e)}",
                "source_documents": [],
                "model": self.model_name,
                "latency": 0.0
            }

