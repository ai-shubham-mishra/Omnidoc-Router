"""
Register PO workflow in MongoDB with corrected schema
"""
import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import json

# Load from parent directory
sys.path.insert(0, '..')
load_dotenv('../.env')

MONGO_DB_URL = os.getenv('MONGO_DB_URL')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')

# Load workflow registration data
with open('../Resources/workflow_registration_payloads.json', 'r') as f:
    workflows_data = json.load(f)

po_workflow = workflows_data['po_registration']

# Connect to MongoDB
client = MongoClient(MONGO_DB_URL)
db = client[MONGO_DB_NAME]

# Update or insert the workflow
result = db['workflows'].update_one(
    {'workflowEndpoint': '/private-po-registration'},
    {'$set': po_workflow},
    upsert=True
)

if result.matched_count > 0:
    print(f"✅ Updated existing PO Registration workflow")
else:
    print(f"✅ Inserted new PO Registration workflow")

# Verify the update
workflow = db['workflows'].find_one({'workflowEndpoint': '/private-po-registration'})
if workflow:
    print(f"\n📋 Workflow: {workflow.get('workflowName')}")
    print(f"   Endpoint: {workflow.get('workflowEndpoint')}")
    schema = workflow.get('workflowSchema', {})
    call0 = schema.get('call0', {})
    input0 = call0.get('Input0', {})
    print(f"\n   Input0 Schema:")
    print(f"     - inputLabel: {input0.get('inputLabel')}")
    print(f"     - inputType: {input0.get('inputType')}")
    print(f"     - inputSize: {input0.get('inputSize')}")
    print(f"\n✅ The router will now map 'Input0' → '{input0.get('inputLabel')}' when forwarding files to the endpoint!")
