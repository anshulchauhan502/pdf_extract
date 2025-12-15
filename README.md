# Technical Specification Extraction using Local LLMs

> Assignment submission for LLM / AI Engineer role  
> Fully local, GPU-accelerated, no paid APIs

---

# 📄 LLM-Based Specification Extraction from Technical Manuals

## 📌 Project Overview

This project implements an **end-to-end local LLM pipeline** to extract **structured technical specifications** (e.g., torque values) from a **single large service manual PDF**.

The system is designed to:

* Work **fully offline** (no paid APIs)
* Run on **GPU servers (H100)**
* Produce output in an **exact predefined JSON schema**, as required by the assignment

### 🎯 Target Output Format (Assignment Requirement)

```json
[
  {
    "component": "Brake Caliper Bolt",
    "spec_type": "Torque",
    "value": "35",
    "unit": "Nm"
  }
]
```

Only the **most relevant** result is returned per query.

---

## 🧠 High-Level Architecture

```
PDF → Text Extraction → Chunking
        ↓
Embedding Model (BGE)
        ↓
FAISS Similarity Search
        ↓
Top-Relevant Chunk
        ↓
Instruction-Tuned LLM
        ↓
Strict JSON Output
```

---

## 📁 Project Structure

```
.
├── extract_pipeline_final.py   # Main pipeline
├── pdfs/
│   └── sample.pdf              # Input service manual
├── models/
│   └── Hermes-2-Pro/            # Local LLM (downloaded)
├── output/
│   └── results.json             # Final extracted output
└── README.md
```

---

## ⚙️ Environment & Dependencies

### Python

* Python **3.10**
* CUDA **12.8**
* PyTorch **2.9+cu128**

### Key Libraries

```bash
pip install torch transformers faiss-cpu pymupdf sentencepiece
pip install huggingface_hub
```

> `sentencepiece` is required for LLaMA/Mistral-based tokenizers.

---

## 🚀 How to Run

```bash
python3 extract_pipeline_final.py \
  --pdf pdfs/sample.pdf \
  --query "Torque for brake caliper bolts" \
  --out output/results.json
```

---

## 🔍 Embedding Model Selection

### ✅ Final Choice

**`BAAI/bge-large-en-v1.5`**

**Why:**

* Excellent for **technical & engineering text**
* Better semantic retrieval than MPNet
* Strong performance on long manuals

### ❌ Previous Model

`sentence-transformers/all-mpnet-base-v2`

* Good general embeddings
* Less accurate for mechanical/engineering specs

---

## 🤖 LLM Models Tested — Full Evaluation Log

This section documents **all models tried**, issues faced, and final decisions.

---

### 1️⃣ `microsoft/phi-2`

**Status:** ❌ Failed
**Issues:**

* Generated Python code instead of JSON
* Weak instruction following
* Hallucinated values

**Verdict:** Too small for structured extraction

---

### 2️⃣ `tiiuae/falcon-7b-instruct`

**Status:** ❌ Failed
**Issues:**

* Verbose outputs
* Ignored strict JSON requirement
* Returned templates instead of real values

**Verdict:** Poor JSON compliance

---

### 3️⃣ `mistralai/Mistral-7B-Instruct`

**Status:** ❌ Failed
**Issues:**

* Gated / authentication issues
* Inconsistent extraction
* Weak component disambiguation

**Verdict:** Not reliable for assignment format

---

### 4️⃣ `meta-llama/Llama-3-8B-Instruct`

**Status:** ❌ Blocked
**Issues:**

* Gated model
* Requires license + manual approval
* Access not granted during assignment timeline

**Verdict:** Ideal model, but unavailable

---

### 5️⃣ `meta-llama/Llama-3.1-8B-Instruct`

**Status:** ❌ Pending approval
**Issues:**

* Requires Meta approval (manual review)
* 401 Unauthorized until approved

**Verdict:** Best possible model, but blocked by access delay

---

### 6️⃣ ✅ **`NousResearch/Hermes-2-Pro-Mistral-7B` (FINAL)**

**Status:** ✅ Success
**Why this worked:**

* Open access (no approval needed)
* Strong instruction tuning
* Excellent JSON compliance
* Correctly extracts multiple specs from same chunk
* Runs efficiently on H100

**Minor Issue (Solved):**

* Required `sentencepiece` for tokenizer

**Verdict:**
✅ **Best open-source model for this task under constraints**

---

## 🧪 Known Challenges & Fixes

| Issue                         | Root Cause                | Fix                      |
| ----------------------------- | ------------------------- | ------------------------ |
| NumPy crashes                 | Version mismatch          | Pinned NumPy 1.26.4      |
| SciPy / sklearn import errors | Transformers auto-import  | Uninstalled unused deps  |
| accelerate circular import    | Broken accelerate install | Removed accelerate       |
| Missing tokenizer             | No sentencepiece          | Installed sentencepiece  |
| Wrong torque selected         | Multiple specs in chunk   | Prompt + filtering logic |

---

## 🧠 Design Decisions

* **Top-1 retrieval** (not top-K) → matches assignment example
* **Local models only** → no paid APIs
* **Strict JSON enforcement** → post-processing + validation
* **Modular pipeline** → easy model replacement

---

## 📈 Possible Improvements (Future Work)

* Cross-encoder reranking for chunks
* Table-aware PDF parsing
* Multi-query batch extraction
* Automatic unit normalization
* Support for multiple documents

---

## ✅ Final Status

* ✔ Fully local pipeline
* ✔ GPU-accelerated
* ✔ Deterministic output format
* ✔ Assignment-compliant JSON
* ✔ Reproducible & documented

---

## 👤 Author

**Anshul Chauhan**
LLM / AI Engineer Candidate


