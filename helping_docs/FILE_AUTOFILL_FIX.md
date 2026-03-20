# File Auto-Fill Fix - Test Documentation

## 🐛 Issue Fixed

**Problem:** Files uploaded to the session were not automatically being associated with file-type workflow inputs, causing the workflow to stay in "collecting" status even when all inputs were provided.

**Root Cause:** Missing logic to check session's `uploaded_files` and auto-fill file-type inputs in the `_collect_inputs_v2` method.

---

## ✅ What Was Fixed

### Changes Made:

1. **`core/orchestrator.py` - `_collect_inputs_v2` method**
   - Added logic to auto-fill file inputs from `session.uploaded_files`
   - Checks for file-type inputs that aren't collected yet
   - Retrieves file paths from session context using `FileHandler.get_files_for_workflow()`
   - Marks file inputs as collected with the file paths
   - Logs when files are auto-filled

2. **Added `total_session_files` to responses**
   - Shows how many files are stored in session context
   - Added to all `RouterResponse` returns in:
     - `_identify_workflow_v2`
     - `_collect_inputs_v2`

---

## 🧪 How to Test

### Test Case: Upload File with Message

**Request:**
```bash
POST http://localhost:8500/api/router/message
Content-Type: multipart/form-data
Authorization: Bearer <jwt>

Fields:
  session_id: "3b00a454-51d5-4e9f-99e2-76d436bf7a0a"
  message: "I need to register a purchase order using my PO document"
  files: [SAMPLE_PO.pdf]
```

**Expected Response (BEFORE FIX):**
```json
{
  "status": "collecting",  ❌ Still collecting
  "response": "Please upload the PO Document File",
  "inputs_required": [
    {
      "field": "Input0",
      "type": "file",
      "collected": false  ❌ Not marked as collected
    }
  ],
  "files_uploaded": [{...}],  ✅ File saved to context
  "total_session_files": 1
}
```

**Expected Response (AFTER FIX):**
```json
{
  "status": "ready_to_execute",  ✅ Ready!
  "response": "All required inputs collected for PO Registration Agent!\n\n• PO Document File: ✓\n• Run ID: ✓\n\nReady to proceed? Reply 'yes' to execute, or provide additional information.",
  "workflow_identified": {...},
  "inputs_required": [
    {
      "field": "Input0",
      "type": "file",
      "collected": true  ✅ Auto-filled from session context!
    },
    {
      "field": "Input1",
      "collected": true
    }
  ],
  "files_uploaded": [{...}],
  "total_session_files": 1  ✅ Shows total files in session
}
```

---

## 🔍 How It Works

### Flow After Fix:

```
1. User uploads file via /api/router/message
   ↓
2. File saved to tmp/{session_id}/uuid_filename.pdf
   ↓
3. File metadata added to session.uploaded_files array
   ↓
4. Router checks if workflow has file-type inputs
   ↓
5. Finds Input0 (type: "file", collected: false)
   ↓
6. Auto-fills Input0 with file paths from session.uploaded_files
   ↓
7. Marks Input0 as collected: true
   ↓
8. All inputs collected → status: "ready_to_execute"
   ↓
9. User says "yes" → Workflow executes
```

### Code Logic:

```python
# In _collect_inputs_v2 (line ~907)
for inp in missing:
    if inp.get("type") == "file" and not inp.get("collected"):
        session_files = session.get("uploaded_files", [])
        if session_files:
            # Get file paths from session context
            file_paths = self.files.get_files_for_workflow(session_files)
            if file_paths:
                # Mark input as collected with the file paths
                await self.sessions.mark_input_collected(
                    session_id, 
                    inp["field"], 
                    file_paths
                )
                logger.info(f"✅ Auto-filled {inp['field']} from session files ({len(file_paths)} files)")
```

---

## 📊 Benefits

✅ **Seamless UX** - Users don't need to upload files separately  
✅ **Single Request** - Upload file + message + specify intent in one call  
✅ **File Context** - Files persist across messages, available for any workflow  
✅ **Smart Detection** - Auto-matches files to file-type inputs  
✅ **Multi-File Support** - Handles multiple files for single input  
✅ **Transparent** - Logs show when files are auto-filled  

---

## 🚀 Next Request Should Work!

Retry your request with the same parameters:

```bash
POST http://localhost:8500/api/router/message
session_id: "3b00a454-51d5-4e9f-99e2-76d436bf7a0a"
message: "I need to register a purchase order using my PO document"
files: [SAMPLE_PO.pdf]
```

**You should now see:**
- ✅ `status: "ready_to_execute"`
- ✅ `Input0.collected: true`
- ✅ Confirmation prompt asking for "yes"

Then send:
```bash
POST /api/router/message
session_id: "3b00a454-51d5-4e9f-99e2-76d436bf7a0a"
message: "yes, proceed"
```

**Workflow will execute!** 🎉

---

## 🔧 Files Modified

- `Omnidoc-Router/core/orchestrator.py` (3 changes)
  - Added file auto-fill logic in `_collect_inputs_v2`
  - Added `total_session_files` to responses (3 locations)

---

**Status:** ✅ **Fixed and Ready to Test!**
