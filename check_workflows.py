import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import json

sys.path.insert(0, '..')
load_dotenv('../.env')

MONGO_DB_URL = os.getenv('MONGO_DB_URL')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')

client = MongoClient(MONGO_DB_URL)
db = client[MONGO_DB_NAME]

workflows = list(db['workflows'].find())
print(f'Found {len(workflows)} workflows:')
for w in workflows:
    print(f"  - {w.get('workflowName')} - Endpoint: {w.get('workflowEndpoint')}")

# Find PO registration workflow
po_workflow = None
for w in workflows:
    if 'PO' in w.get('workflowName', '') or 'po' in w.get('workflowEndpoint', ''):
        po_workflow = w
        break

if po_workflow:
    print(f"\nFound PO workflow: {po_workflow.get('workflowName')}")
    print(f"Endpoint: {po_workflow.get('workflowEndpoint')}")
    schema = po_workflow.get('workflowSchema', {})
    print('\nFull schema:')
    print(json.dumps(schema, indent=2))
