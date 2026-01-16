import os
import shutil
from chat_logic import RAGChatbot

# Clean up previous runs
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

print("="*80)
print("COMPREHENSIVE COMPARISON TESTS")
print("="*80)

# Test 1: Identical Configs (Should be ALL MATCH)
print("\n" + "="*80)
print("TEST 1: IDENTICAL CONFIGS")
print("="*80)

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
"""

with open("golden.cfg", "w") as f:
    f.write(golden_content)
with open("candidate.cfg", "w") as f:
    f.write(golden_content)

bot = RAGChatbot(model_name="llama3.2:3b")
bot.process_file("golden.cfg", extra_metadata={"config_role": "golden"})
bot.process_file("candidate.cfg", extra_metadata={"config_role": "candidate"})

query = "Compare 'candidate.cfg' against 'golden.cfg'. Focus on VLANs, Interfaces, and Routes."
response = bot.compare_configs(query)

print("\nRESULT:")
print(response["result"])
print("\nExpected: All features should show ✅ MATCH")

os.remove("golden.cfg")
os.remove("candidate.cfg")
shutil.rmtree("./chroma_db")

# Test 2: Missing VLAN in Candidate
print("\n" + "="*80)
print("TEST 2: MISSING VLAN IN CANDIDATE")
print("="*80)

candidate_content = """
hostname Switch-Core
!
vlan 10
 name Sales
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

# Reinitialize bot with fresh vector store
bot = RAGChatbot(model_name="llama3.2:3b")
bot.process_file("golden.cfg", extra_metadata={"config_role": "golden"})
bot.process_file("candidate.cfg", extra_metadata={"config_role": "candidate"})

response = bot.compare_configs(query)

print("\nRESULT:")
print(response["result"])
print("\nExpected: vlan 10 = MATCH, vlan 20 = MISSING, interface = MATCH, router = MATCH")

os.remove("golden.cfg")
os.remove("candidate.cfg")
shutil.rmtree("./chroma_db")

# Test 3: Extra VLAN in Candidate
print("\n" + "="*80)
print("TEST 3: EXTRA VLAN IN CANDIDATE")
print("="*80)

candidate_content = """
hostname Switch-Core
!
vlan 10
 name Sales
vlan 20
 name Engineering
vlan 30
 name Marketing
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

response = bot.compare_configs(query)

print("\nRESULT:")
print(response["result"])
print("\nExpected: vlan 10 = MATCH, vlan 20 = MATCH, vlan 30 = EXTRA, interface = MATCH, router = MATCH")

os.remove("golden.cfg")
os.remove("candidate.cfg")
shutil.rmtree("./chroma_db")

# Test 4: Modified VLAN Name
print("\n" + "="*80)
print("TEST 4: MODIFIED VLAN NAME")
print("="*80)

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

response = bot.compare_configs(query)

print("\nRESULT:")
print(response["result"])
print("\nExpected: vlan 10 = DIFF, vlan 20 = MATCH, interface = MATCH, router = MATCH")

os.remove("golden.cfg")
os.remove("candidate.cfg")

print("\n" + "="*80)
print("ALL TESTS COMPLETED")
print("="*80)
