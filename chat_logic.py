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

    def compare_configs(self, query: str, golden_filename: str = None, candidate_filename: str = None, mode: str = "quick") -> Dict[str, Any]:
        """
        Specialized method for comparing two configurations.
        
        Args:
            query: The comparison query/prompt
            golden_filename: Name of the golden config file
            candidate_filename: Name of the candidate config file
            mode: "quick" for deterministic table, "deep" for LLM analysis
        
        Returns:
            Dict with result, source_documents, model, and latency
        """
        start_time = time.time()
        try:
            logger.info(f"Comparing: {query} (Golden: {golden_filename}, Candidate: {candidate_filename}, Mode: {mode})")
            
            # 1. Construct Filters
            if golden_filename:
                filter_golden = {
                    "$and": [
                        {"config_role": "golden"},
                        {"source": golden_filename}
                    ]
                }
            else:
                filter_golden = {"config_role": "golden"}

            if candidate_filename:
                filter_candidate = {
                    "$and": [
                        {"config_role": "candidate"},
                        {"source": candidate_filename}
                    ]
                }
            else:
                filter_candidate = {"config_role": "candidate"}
                
            # 2. Retrieve Chunks
            docs_golden = self.ingestion.vector_store.similarity_search(
                query, 
                k=50, 
                filter=filter_golden
            )
            
            docs_candidate = self.ingestion.vector_store.similarity_search(
                query, 
                k=50, 
                filter=filter_candidate
            )
            
            # 3. Group by parent_line for comparison
            golden_by_parent = {}
            for doc in docs_golden:
                parent = doc.metadata.get('parent_line', 'unknown')
                section_type = doc.metadata.get('section_type', 'unknown')
                golden_by_parent[parent] = {
                    'content': doc.page_content.strip(),
                    'section_type': section_type
                }
            
            candidate_by_parent = {}
            for doc in docs_candidate:
                parent = doc.metadata.get('parent_line', 'unknown')
                section_type = doc.metadata.get('section_type', 'unknown')
                candidate_by_parent[parent] = {
                    'content': doc.page_content.strip(),
                    'section_type': section_type
                }
            
            all_parents = set(golden_by_parent.keys()) | set(candidate_by_parent.keys())
            
            # Parse query to understand focus areas
            focus_vlans = 'vlan' in query.lower()
            focus_interfaces = 'interface' in query.lower()
            focus_routes = 'route' in query.lower() or 'ospf' in query.lower()
            focus_acls = 'acl' in query.lower() or 'security' in query.lower()
            focus_qos = 'qos' in query.lower()
            
            # MODE: QUICK - Deterministic Comparison
            if mode == "quick":
                result_rows = []
                
                for parent in sorted(all_parents):
                    golden_info = golden_by_parent.get(parent)
                    candidate_info = candidate_by_parent.get(parent)
                    
                    # Skip hostname unless explicitly requested
                    if 'hostname' in parent.lower() and 'hostname' not in query.lower():
                        continue
                    
                    # Filter by section type based on query
                    if golden_info:
                        section_type = golden_info['section_type']
                    elif candidate_info:
                        section_type = candidate_info['section_type']
                    else:
                        section_type = 'unknown'
                    
                    # Skip if not in focus areas
                    skip = True
                    if focus_vlans and section_type == 'vlan':
                        skip = False
                    if focus_interfaces and section_type == 'interface':
                        skip = False
                    if focus_routes and section_type == 'router':
                        skip = False
                    if focus_acls and 'access-list' in parent.lower():
                        skip = False
                    if focus_qos and 'qos' in parent.lower():
                        skip = False
                    
                    if skip and (focus_vlans or focus_interfaces or focus_routes or focus_acls or focus_qos):
                        continue
                    
                    # Determine status
                    if golden_info and candidate_info:
                        if golden_info['content'] == candidate_info['content']:
                            status = "✅ MATCH"
                        else:
                            status = f"⚠️ DIFF"
                    elif golden_info and not candidate_info:
                        status = "❌ MISSING"
                    elif not golden_info and candidate_info:
                        status = "➕ EXTRA"
                    else:
                        continue
                    
                    # Format the parent line for display
                    display_parent = parent.replace('\n', ' / ')
                    result_rows.append(f"| {display_parent} | {status} |")
                
                # Build result table
                if result_rows:
                    result_lines = [
                        "| Feature/Line | Golden Config | Candidate Status |",
                        "| --- | --- | --- |"
                    ]
                    result_lines.extend(result_rows)
                    result_text = "\n".join(result_lines)
                else:
                    result_text = "No relevant features found to compare based on the query."
                
                model_label = f"{self.model_name} (deterministic)"
            
            # MODE: DEEP - LLM Analysis
            else:
                # Build comprehensive context for LLM
                differences = []
                matches = []
                missing = []
                extra = []
                
                for parent in sorted(all_parents):
                    golden_info = golden_by_parent.get(parent)
                    candidate_info = candidate_by_parent.get(parent)
                    
                    if golden_info and candidate_info:
                        if golden_info['content'] == candidate_info['content']:
                            matches.append(f"**{parent}**: Identical")
                        else:
                            differences.append({
                                'feature': parent,
                                'golden': golden_info['content'],
                                'candidate': candidate_info['content']
                            })
                    elif golden_info and not candidate_info:
                        missing.append(f"**{parent}**:\n```\n{golden_info['content']}\n```")
                    elif not golden_info and candidate_info:
                        extra.append(f"**{parent}**:\n```\n{candidate_info['content']}\n```")
                
                # Build detailed prompt for LLM
                context_parts = []
                
                if differences:
                    context_parts.append("### DIFFERENCES DETECTED:\n")
                    for diff in differences:
                        context_parts.append(f"\n**Feature: {diff['feature']}**")
                        context_parts.append(f"\nGolden Config:\n```\n{diff['golden']}\n```")
                        context_parts.append(f"\nCandidate Config:\n```\n{diff['candidate']}\n```")
                
                if missing:
                    context_parts.append("\n### MISSING IN CANDIDATE:\n")
                    context_parts.extend(missing)
                
                if extra:
                    context_parts.append("\n### EXTRA IN CANDIDATE:\n")
                    context_parts.extend(extra)
                
                if matches:
                    context_parts.append(f"\n### MATCHING FEATURES: {len(matches)} features are identical")
                
                context_text = "\n".join(context_parts)
                
                prompt = f"""
You are a Senior Network Engineer performing a detailed configuration audit.

CONFIGURATION COMPARISON RESULTS:
{context_text}

USER REQUEST:
{query}

INSTRUCTIONS:
Provide a comprehensive analysis including:

1. **Executive Summary**: Brief overview of the comparison results
2. **Critical Differences**: Highlight any differences that could impact:
   - Network functionality
   - Security posture
   - Performance
   - Compliance
3. **Missing Configurations**: Analyze what's missing in the candidate and why it matters
4. **Extra Configurations**: Evaluate additional configs in candidate (good or bad?)
5. **Security Implications**: Any security concerns from the differences?
6. **Best Practice Recommendations**: Suggestions for improvement
7. **Risk Assessment**: Rate the risk of deploying the candidate config (Low/Medium/High)

Format your response in clear, well-structured Markdown with headers and bullet points.
"""
                
                # Invoke LLM for deep analysis
                response_msg = self.llm.invoke(prompt)
                result_text = response_msg.content
                model_label = self.model_name
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Combine docs for citation
            docs = docs_golden + docs_candidate
            
            return {
                "result": result_text,
                "source_documents": docs,
                "model": model_label,
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

