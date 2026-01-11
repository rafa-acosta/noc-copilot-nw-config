
import os
import shutil
from chat_logic import RAGChatbot

# Clean up previous runs
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# 1. Create content
DIRTY_CONTENT = "hostname OLD_ROUTER_SHOULD_NOT_SEE_THIS"
CLEAN_CONTENT = "hostname CORRECT_SWITCH"

with open("dirty.cfg", "w") as f: f.write(DIRTY_CONTENT)
with open("clean_g.cfg", "w") as f: f.write(CLEAN_CONTENT)
with open("clean_c.cfg", "w") as f: f.write(CLEAN_CONTENT)

# Initialize
bot = RAGChatbot(model_name="llama3.2:3b")

# 2. Pollute DB with 'dirty.cfg' as Candidate
print("Ingesting Pollutant...")
bot.process_file("dirty.cfg", extra_metadata={"config_role": "candidate"})

# 3. Ingest Actual Files
print("Ingesting Clean Files...")
bot.process_file("clean_g.cfg", extra_metadata={"config_role": "golden"})
bot.process_file("clean_c.cfg", extra_metadata={"config_role": "candidate"})

# 4. Compare with Strict Filenames
query = "Compare 'clean_c.cfg' against 'clean_g.cfg'."
print(f"Running Filtered Comparison: {query}")

# We pass explicit filenames to filtering
response = bot.compare_configs(
    query, 
    golden_filename="clean_g.cfg", 
    candidate_filename="clean_c.cfg" # This should exclude dirty.cfg
)

# 5. Inspect Results
print("\n--- RETRIEVED SOURCES ---")
found_dirty = False
for doc in response["source_documents"]:
    print(f"[{doc.metadata.get('config_role')}] {doc.metadata.get('source')}")
    if "dirty" in doc.metadata.get("source", ""):
        found_dirty = True

print("-" * 30)
if found_dirty:
    print("❌ FAILED: Found dirty file in context despite filtering!")
else:
    print("✅ SUCCESS: Only cleaned files retrieved.")

# Cleanup
os.remove("dirty.cfg")
os.remove("clean_g.cfg")
os.remove("clean_c.cfg")
