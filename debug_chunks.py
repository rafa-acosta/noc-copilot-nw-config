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

# Debug: Check what chunks were created
print("\n" + "="*80)
print("DEBUGGING: Examining stored chunks")
print("="*80)

# Query for all golden chunks
filter_golden = {"config_role": "golden"}
docs_golden = bot.ingestion.vector_store.similarity_search(
    "vlan interface router", 
    k=50, 
    filter=filter_golden
)

print(f"\n📦 GOLDEN CONFIG CHUNKS (Total: {len(docs_golden)}):")
print("-"*80)
for i, doc in enumerate(docs_golden, 1):
    print(f"\nChunk {i}:")
    print(f"  Parent Line: {doc.metadata.get('parent_line', 'N/A')}")
    print(f"  Section Type: {doc.metadata.get('section_type', 'N/A')}")
    print(f"  Lines: {doc.metadata.get('line_start', 'N/A')}-{doc.metadata.get('line_end', 'N/A')}")
    print(f"  Source: {doc.metadata.get('source', 'N/A')}")
    print(f"  Content:\n{doc.page_content}")

# Query for all candidate chunks
filter_candidate = {"config_role": "candidate"}
docs_candidate = bot.ingestion.vector_store.similarity_search(
    "vlan interface router", 
    k=50, 
    filter=filter_candidate
)

print(f"\n\n📦 CANDIDATE CONFIG CHUNKS (Total: {len(docs_candidate)}):")
print("-"*80)
for i, doc in enumerate(docs_candidate, 1):
    print(f"\nChunk {i}:")
    print(f"  Parent Line: {doc.metadata.get('parent_line', 'N/A')}")
    print(f"  Section Type: {doc.metadata.get('section_type', 'N/A')}")
    print(f"  Lines: {doc.metadata.get('line_start', 'N/A')}-{doc.metadata.get('line_end', 'N/A')}")
    print(f"  Source: {doc.metadata.get('source', 'N/A')}")
    print(f"  Content:\n{doc.page_content}")

# Now test the actual comparison query
print("\n\n" + "="*80)
print("TESTING COMPARISON QUERY")
print("="*80)

query = "Compare 'candidate.cfg' against 'golden.cfg'. Focus on VLANs, Interfaces, and Routes."

# Simulate what compare_configs does
docs_golden_compare = bot.ingestion.vector_store.similarity_search(
    query, 
    k=50, 
    filter=filter_golden
)

docs_candidate_compare = bot.ingestion.vector_store.similarity_search(
    query, 
    k=50, 
    filter=filter_candidate
)

print(f"\n🔍 GOLDEN CHUNKS RETRIEVED FOR COMPARISON (Total: {len(docs_golden_compare)}):")
for i, doc in enumerate(docs_golden_compare, 1):
    print(f"  {i}. {doc.metadata.get('parent_line', 'N/A')}")

print(f"\n🔍 CANDIDATE CHUNKS RETRIEVED FOR COMPARISON (Total: {len(docs_candidate_compare)}):")
for i, doc in enumerate(docs_candidate_compare, 1):
    print(f"  {i}. {doc.metadata.get('parent_line', 'N/A')}")

# Cleanup
os.remove("golden.cfg")
os.remove("candidate.cfg")
