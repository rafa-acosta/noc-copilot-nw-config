
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

# Run Quick Comparison
query = (
    "Compare 'candidate.cfg' against 'golden.cfg'. "
    "Produce a **Markdown Table** only. No summary text.\n"
    "Columns: | Feature/Line | Golden Config | Candidate Status |\n"
    "Rules for 'Candidate Status':\n"
    "- ✅ MATCH\n"
    "- ❌ MISSING\n"
    "- ➕ EXTRA\n"
    "- ⚠️ DIFF: <Show value>\n"
    "Focus on VLANs, Interfaces, and Routes."
)

print(f"\nQuery: {query}\n")
response = bot.compare_configs(query)

print("-" * 50)
print("RESPONSE (Should be ALL MATCHES):")
print(response["result"])
print("-" * 50)

# Cleanup
os.remove("golden.cfg")
os.remove("candidate.cfg")
