# 📂 File Intelligence & Multi-Workflow Sessions Guide

## Overview

The Router now features intelligent file management across multi-workflow sessions. Files are automatically classified, matched to the right workflows, and tracked throughout their lifecycle.

---

## 🎯 Key Features

### 1. **Automatic File Classification**
Every uploaded file is analyzed and classified using AI:
- Document type detection (purchase_order, invoice, business_card, resume, etc.)
- Content summary generation
- Keyword extraction
- Confidence scoring

### 2. **Smart File-to-Workflow Matching**
Files are intelligently matched to workflow inputs based on:
- ✅ Document type alignment with workflow purpose
- ✅ Keyword overlap between file and workflow
- ✅ Upload timing (files uploaded during a workflow are preferred for that workflow)
- ❌ Previous usage (files already used by completed workflows are deprioritized)

### 3. **Conversational File Queries**
Ask questions about uploaded files and workflows:
- "Which document are you referring to?"
- "What files have I uploaded?"
- "What happened with the PO workflow?"

### 4. **File Lifecycle Tracking**
Files persist across workflows with status tracking:
- `available` — Fresh file, ready to use
- `used` — File was used by a completed workflow
- Reuse prevention for new workflows

---

## 🚀 Usage Examples

### Example 1: Single Workflow with File Upload

```
POST /api/router/message
{
  "session_id": "abc123",
  "message": "I want to register a purchase order",
  "files": [SAMPLE_PO.pdf]
}

Response:
{
  "status": "ready_to_execute",
  "response": "✅ All required inputs collected for PO Registration Agent!\n• PO Document: ✓\n• Run ID: ✓\n\nReady to proceed? Reply 'yes' to execute.",
  "workflow_identified": {
    "name": "PO Registration Agent"
  },
  "total_session_files": 1
}
```

**What Happened:**
1. File `SAMPLE_PO.pdf` uploaded
2. Classified as `"purchase_order"` (confidence: 95%)
3. Smart-matched to PO workflow's file input
4. Auto-filled because score ≥ 4.0 threshold
5. Workflow ready to execute

---

### Example 2: Multi-Workflow Session (Problem Solved!)

**Step 1: Complete PO Workflow**
```
POST /api/router/message
{
  "session_id": "abc123",
  "message": "yes, proceed"
}

→ PO workflow executes and completes
→ SAMPLE_PO.pdf marked as "used" by PO Registration Agent
```

**Step 2: Start HubSpot Workflow (Same Session)**
```
POST /api/router/message
{
  "session_id": "abc123",  // Same session!
  "message": "I want to add a lead from my business card to HubSpot"
}

Response:
{
  "status": "collecting",
  "response": "Please upload the Business Card file.",
  "workflow_identified": {
    "name": "HubSpot: Business Card to Lead"
  },
  "inputs_required": [
    {
      "field": "Input0",
      "label": "Business Card",
      "collected": false  // ✅ NOT auto-filled with PO file!
    }
  ]
}
```

**Why SAMPLE_PO.pdf wasn't used:**
- Document type: `purchase_order` ❌ (doesn't match "business card")
- Already used: Yes ❌ (by PO Registration workflow)
- Status: `used` ❌
- **Score: -7.0** (below threshold of 4.0)
- Result: Not auto-filled, asks for correct file ✅

**Step 3: Upload Business Card**
```
POST /api/router/message
{
  "session_id": "abc123",
  "files": [business_card.jpg]
}

→ Classified as "business_card" (confidence: 92%)
→ Smart-matched to HubSpot input (score: 8.5)
→ Auto-filled ✅
→ Ready to execute!
```

---

### Example 3: Asking Questions About Files

**Scenario:** Router says "All inputs collected" but you're confused about which file it's using.

```
POST /api/router/message
{
  "session_id": "abc123",
  "message": "Which document are you referring to?"
}

Response:
{
  "status": "ready_to_execute",
  "response": "I'm referring to 'SAMPLE_PO.pdf' which you uploaded earlier. This is a purchase order document that was classified with 95% confidence. It's been matched to the 'PO Document File' input for the PO Registration Agent workflow. The file contains 3 pages and is 2.3 MB in size."
}
```

**Other Questions You Can Ask:**
- "What files have I uploaded?"
- "What workflows have I completed?"
- "What happened with the last workflow?"
- "What inputs are still missing?"
- "Show me the file status"

---

## 📊 File Classification Examples

| File Name | Detected Type | Confidence | Keywords |
|-----------|---------------|------------|----------|
| `SAMPLE_PO.pdf` | purchase_order | 95% | purchase, order, po, procurement |
| `John_Doe_Card.jpg` | business_card | 92% | business, card, contact, lead |
| `Invoice_12345.pdf` | invoice | 98% | invoice, billing, payment |
| `Resume_Jane.pdf` | resume | 88% | resume, cv, candidate, experience |
| `Contract_v2.docx` | contract | 85% | contract, agreement, legal, terms |

---

## 🔢 Smart Matching Score Calculation

When determining if a file should auto-fill an input, the system scores it:

| Factor | Points | Example |
|--------|--------|---------|
| **Type matches workflow** | +5.0 | `business_card` file → HubSpot lead workflow |
| **Keyword overlap** | +1.5 each | File keywords match workflow/input labels |
| **Uploaded during this workflow** | +3.0 | File uploaded after workflow was identified |
| **File NOT previously used** | +1.0 | Fresh file, never used by another workflow |
| **Type DOESN'T match** | -2.0 | `purchase_order` file → business card input |
| **Already used by workflow** | -3.0 | File used by a completed workflow |
| **Status is "used"** | -2.0 | File marked as used |

**Auto-fill threshold:** Score ≥ 4.0

**Example Scores:**
- Perfect match (new business card for HubSpot): **9.5** ✅ Auto-fill
- Wrong type (PO for business card input): **-7.0** ❌ Ask user
- Reusing old file: **1.0** ❌ Ask user (below threshold)

---

## 💬 Conversational Context (RAG)

The router maintains awareness of:
- ✅ All uploaded files with classifications
- ✅ Completed workflows and their results
- ✅ Current workflow status
- ✅ Which files were used by which workflows
- ✅ Recent conversation history

**Session Context Example:**
```json
{
  "completed_workflows": [
    {
      "name": "PO Registration Agent",
      "status": "completed",
      "completed_at": "2026-03-20T10:30:00Z"
    }
  ],
  "current_workflow": {
    "name": "HubSpot: Business Card to Lead",
    "status": "collecting",
    "missing_inputs": ["Business Card File"]
  },
  "session_files": [
    {
      "name": "SAMPLE_PO.pdf",
      "type": "purchase_order",
      "status": "used",
      "used_by": ["PO Registration Agent"]
    },
    {
      "name": "business_card.jpg",
      "type": "business_card",
      "status": "available",
      "used_by": []
    }
  ]
}
```

---

## 🎨 User Experience Flow

### **Before (Blind Auto-Fill)**
```
User: Complete PO workflow with SAMPLE_PO.pdf ✅
User: "I want to add a business card to HubSpot"
Router: ✅ All inputs collected! [WRONG - used PO file]
User: "Wait, which file?"
Router: [No answer, stuck]
```

### **After (Smart Matching + RAG)**
```
User: Complete PO workflow with SAMPLE_PO.pdf ✅
User: "I want to add a business card to HubSpot"
Router: Please upload the Business Card file. ✅
User: Uploads business_card.jpg
Router: ✅ All inputs collected! [CORRECT]

OR if user asks:
User: "Which document are you referring to?"
Router: "The HubSpot workflow needs a business card. You've uploaded SAMPLE_PO.pdf (purchase order, already used). Please upload a business card image." ✅
```

---

## 🛠️ Advanced: File Status Transitions

```
Upload → [available]
           ↓
      Matched to workflow
           ↓
      Workflow executes
           ↓
      [used] ← Marked after completion
           ↓
      New workflow starts
           ↓
      Not auto-filled (score penalty)
           ↓
      User must explicitly re-upload or confirm reuse
```

---

## 📝 Best Practices

### **For Users:**
1. ✅ Upload files close to when you need them (better matching)
2. ✅ Use descriptive filenames (`business_card.jpg` better than `IMG_1234.jpg`)
3. ✅ Ask questions if unsure which file is being used
4. ✅ Complete one workflow before starting another in the same session

### **For Developers:**
1. ✅ Workflow input labels should be descriptive ("Business Card" not "Input0")
2. ✅ Add relevant keywords to workflow tags
3. ✅ Test multi-workflow sessions with different file types
4. ✅ Monitor classification confidence scores in logs

---

## 🔍 Troubleshooting

### **Problem:** File not auto-filling when it should
**Check:**
- Is the file type correct for the workflow?
- Was the file recently uploaded or is it old/used?
- Check logs for smart-match score

**Solution:**
- Re-upload the file fresh
- Or ask: "Can you use the [filename] I uploaded earlier?"

---

### **Problem:** Wrong file auto-filled
**Check:**
- Was there a file with similar classification?
- Multiple files of the same type in session?

**Solution:**
- System will ask for confirmation before executing
- Say "no" and specify the correct file

---

### **Problem:** Router not answering questions
**Check:**
- Include question mark or question keywords
- Examples: "What files?", "Which document?", "Tell me about..."

**Solution:**
- Make questions explicit
- Quote file names if referring to specific ones

---

## 🎯 Key Takeaways

1. **Files are smart** — They know what type they are and where they should go
2. **No blind reuse** — Old files won't auto-fill new workflows
3. **Ask anything** — The router understands context and can answer questions
4. **Multi-workflow ready** — Sessions persist across unlimited workflows
5. **Threshold-based** — Only high-confidence matches auto-fill (≥ 4.0 score)

---

## 📚 Related Documentation

- [FILE_AUTOFILL_FIX.md](FILE_AUTOFILL_FIX.md) — Initial file auto-fill fix
- [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) — Router v2.0 architecture
- [DEPLOYMENT_V2.md](../DEPLOYMENT_V2.md) — Deployment guide

---

**Last Updated:** March 20, 2026  
**Version:** Router v2.1 (File Intelligence)
