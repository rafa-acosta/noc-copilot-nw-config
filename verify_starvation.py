
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

# Run Quick Comparison (Standard Retrieval)
query = "Compare 'candidate.cfg' against 'golden.cfg'."

print(f"\nQuery: {query}\n")

# Manually trigger similarity search to inspect results
print("Inspecting Retrieval Balance...")
# Note: Using the underlying vectorstore directly to see raw results
docs = bot.ingestion.vector_store.similarity_search(query, k=10)

golden_count = sum(1 for d in docs if d.metadata.get("config_role") == "golden")
candidate_count = sum(1 for d in docs if d.metadata.get("config_role") == "candidate")

print(f"Golden Docs Retrieved: {golden_count}")
print(f"Candidate Docs Retrieved: {candidate_count}")

if golden_count == 0 or candidate_count == 0:
    print("🚨 STARVATION DETECTED! One context is empty.")
else:
    print("✅ Balance looks okay.")

# Cleanup
os.remove("golden.cfg")
os.remove("candidate.cfg")
