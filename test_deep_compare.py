import os
import shutil
from chat_logic import RAGChatbot

# Clean up previous runs
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

print("="*80)
print("TESTING DEEP COMPARE vs QUICK DIFF")
print("="*80)

# Create test configs with differences
golden_content = """
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
!
access-list 100 permit ip any any
"""

candidate_content = """
hostname Switch-Core
!
vlan 10
 name Sales-Department
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

with open("golden.cfg", "w") as f:
    f.write(golden_content)
with open("candidate.cfg", "w") as f:
    f.write(candidate_content)

bot = RAGChatbot(model_name="llama3.2:3b")
bot.process_file("golden.cfg", extra_metadata={"config_role": "golden"})
bot.process_file("candidate.cfg", extra_metadata={"config_role": "candidate"})

# Test 1: Quick Diff (Deterministic)
print("\n" + "="*80)
print("TEST 1: QUICK DIFF (Deterministic)")
print("="*80)

query_quick = "Compare 'candidate.cfg' against 'golden.cfg'. Focus on VLANs, Interfaces, Routes, ACLs."
response_quick = bot.compare_configs(query_quick, mode="quick")

print(f"\nMode: {response_quick['model']}")
print(f"Latency: {response_quick['latency']}s")
print("\nResult:")
print(response_quick["result"])

# Test 2: Deep Compare (LLM Analysis)
print("\n\n" + "="*80)
print("TEST 2: DEEP COMPARE (LLM Analysis)")
print("="*80)

query_deep = (
    "Compare the candidate configuration 'candidate.cfg' "
    "against the golden configuration 'golden.cfg'. "
    "Provide a detailed analysis of differences in VLANs, Interfaces, Routing, and ACLs. "
    "Highlight missing or extra configurations in the candidate file."
)
response_deep = bot.compare_configs(query_deep, mode="deep")

print(f"\nMode: {response_deep['model']}")
print(f"Latency: {response_deep['latency']}s")
print("\nResult:")
print(response_deep["result"])

# Cleanup
os.remove("golden.cfg")
os.remove("candidate.cfg")

print("\n" + "="*80)
print("TESTS COMPLETED")
print("="*80)
print("\nExpected Results:")
print("- Quick Diff: Simple table with ✅ MATCH, ⚠️ DIFF, ❌ MISSING")
print("- Deep Compare: Detailed LLM analysis with security implications and recommendations")
