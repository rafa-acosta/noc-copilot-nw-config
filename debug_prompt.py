import os
import shutil
from chat_logic import RAGChatbot

# Clean up previous runs
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Create IDENTICAL dummy content
cfg_content = """
hostname Switch-Core
!
vlan 10
 name Sales
vlan 20
 name Engineering
!
interface GigabitEthernet1/0/1
 description Uplink to Router
 switchport mode trunk
!
router ospf 1
 network 10.0.0.0 0.0.0.255 area 0
"""

# Save dummy files
with open("golden.cfg", "w") as f:
    f.write(cfg_content)

with open("candidate.cfg", "w") as f:
    f.write(cfg_content)

# Initialize Chatbot
bot = RAGChatbot(model_name="llama3.2:3b")

# Process Files
print("Processing Golden Config...")
bot.process_file("golden.cfg", extra_metadata={"config_role": "golden"})

print("Processing Candidate Config...")
bot.process_file("candidate.cfg", extra_metadata={"config_role": "candidate"})

# Debug: Manually build the comparison table to see what LLM receives
query = "Compare 'candidate.cfg' against 'golden.cfg'. Focus on VLANs, Interfaces, and Routes."

filter_golden = {"config_role": "golden"}
filter_candidate = {"config_role": "candidate"}

docs_golden = bot.ingestion.vector_store.similarity_search(
    query, 
    k=50, 
    filter=filter_golden
)

docs_candidate = bot.ingestion.vector_store.similarity_search(
    query, 
    k=50, 
    filter=filter_candidate
)

# Group by parent_line
golden_by_parent = {}
for doc in docs_golden:
    parent = doc.metadata.get('parent_line', 'unknown')
    golden_by_parent[parent] = doc.page_content.strip()

candidate_by_parent = {}
for doc in docs_candidate:
    parent = doc.metadata.get('parent_line', 'unknown')
    candidate_by_parent[parent] = doc.page_content.strip()

# Create comparison table
all_parents = set(golden_by_parent.keys()) | set(candidate_by_parent.keys())

print("\n" + "="*80)
print("COMPARISON TABLE THAT WILL BE SENT TO LLM:")
print("="*80)

comparison_table = []
comparison_table.append("| Feature (Parent Line) | Golden Config | Candidate Config | Pre-Analysis |")
comparison_table.append("|---|---|---|---|")

for parent in sorted(all_parents):
    golden_content = golden_by_parent.get(parent, "NOT FOUND")
    candidate_content = candidate_by_parent.get(parent, "NOT FOUND")
    
    # Pre-analyze for exact matches
    if golden_content == candidate_content and golden_content != "NOT FOUND":
        pre_analysis = "EXACT MATCH"
    elif golden_content == "NOT FOUND":
        pre_analysis = "EXTRA in Candidate"
    elif candidate_content == "NOT FOUND":
        pre_analysis = "MISSING in Candidate"
    else:
        pre_analysis = "NEEDS COMPARISON"
    
    print(f"\nParent: {parent}")
    print(f"  Golden: {golden_content[:50]}...")
    print(f"  Candidate: {candidate_content[:50]}...")
    print(f"  Pre-Analysis: {pre_analysis}")
    print(f"  Are they equal? {golden_content == candidate_content}")
    
    # Escape pipe characters in content for table formatting
    golden_display = golden_content.replace("|", "\\|").replace("\n", " / ")
    candidate_display = candidate_content.replace("|", "\\|").replace("\n", " / ")
    
    comparison_table.append(f"| {parent} | {golden_display} | {candidate_display} | {pre_analysis} |")

comparison_text = "\n".join(comparison_table)
print("\n" + "="*80)
print("FULL TABLE:")
print("="*80)
print(comparison_text)

# Cleanup
os.remove("golden.cfg")
os.remove("candidate.cfg")
