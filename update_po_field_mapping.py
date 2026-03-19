"""
Update PO Registration Agent workflow to add endpoint_field_name for Input0.
This allows the router to map Input0 -> 'files' when forwarding to the endpoint.
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "omnidoc")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
workflows_collection = db["workflows"]


def update_po_workflow():
    """Add endpoint_field_name to Input0 in PO Registration Agent workflow."""
    
    workflow_id = "93b14d16-fad5-4ecd-a106-bcc2054c5391"
    
    # Find the workflow
    workflow = workflows_collection.find_one({"workflowId": workflow_id})
    
    if not workflow:
        print(f"❌ Workflow with ID {workflow_id} not found!")
        return
    
    print(f"✅ Found workflow: {workflow['workflowName']}")
    
    # Update the schema to add endpoint_field_name to Input0
    update_result = workflows_collection.update_one(
        {"workflowId": workflow_id},
        {
            "$set": {
                "workflowSchema.call0.body_data.Input0.endpoint_field_name": "files",
                "UpdatedAt": "2026-03-19 00:00:00"
            }
        }
    )
    
    if update_result.modified_count > 0:
        print(f"✅ Successfully updated workflow schema!")
        print(f"   - Added endpoint_field_name='files' to Input0")
        
        # Verify the update
        updated_workflow = workflows_collection.find_one({"workflowId": workflow_id})
        input0 = updated_workflow["workflowSchema"]["call0"]["body_data"]["Input0"]
        print(f"\n📋 Updated Input0 configuration:")
        print(f"   - inputLabel: {input0.get('inputLabel')}")
        print(f"   - inputType: {input0.get('inputType')}")
        print(f"   - endpoint_field_name: {input0.get('endpoint_field_name')}")
    else:
        print(f"⚠️  No changes made (field might already exist)")


if __name__ == "__main__":
    print("=" * 60)
    print("Updating PO Registration Agent Workflow")
    print("=" * 60)
    update_po_workflow()
    print("=" * 60)
    client.close()
