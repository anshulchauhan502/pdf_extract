#!/usr/bin/env python3
"""
extract_pipeline_final.py

End-to-end local extractor tuned to the assignment example.
- Uses PyMuPDF (fitz) to extract text from PDF(s)
- Uses transformers for embeddings (mean pooling) on GPU
- Uses FAISS (CPU) for fast similarity search
- Uses a local HF LLM (default: Mistral-7B-Instruct) for extraction
- Returns exactly one most-relevant record in the assignment JSON schema:
  { "component": "...", "spec_type": "...", "value": "...", "unit": "..." }

Notes:
- Works with your reported environment: Python 3.10, torch+cu128, faiss (CPU), transformers.
- If LLM model is too large to fit on one GPU, the script will attempt a device_map auto fallback.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import fitz  # pymupdf
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# -----------------------
# CONFIG - edit if desired
# -----------------------
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

LLM_MODEL = "models/Hermes-2-Pro"


  # change if you prefer
EMBED_BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 1  # retrieve only the top chunk -> single most relevant
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hard-coded specification types we expect in the document. You can extend this list.
SPEC_TYPES = ["Torque", "Pressure", "Temperature", "Tightening torque", "Torque (Nm)"]

# Regex to detect numeric + unit patterns (simple but covers common cases)
NUM_UNIT_RE = re.compile(r'(\d+(?:\.\d+)?)(?:\s*(?:–|-|to)\s*(\d+(?:\.\d+)?))?\s*(Nm|N·m|lb[-\s]?ft|lb-ft|in-lb|in·lb|psi|bar|kPa|°C|C|°F|F)?', re.I)


# -----------------------
# Utilities: PDF -> pages
# -----------------------
def extract_text_from_pdf(path: str):
    doc = fitz.open(path)
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        pages.append({"page_no": i + 1, "text": text})
    return pages


# -----------------------
# Chunking (simple paragraph-based)
# -----------------------
def clean_linewraps(text: str) -> str:
    text = re.sub(r"-\n\s*", "", text)  # fix hyphenated breaks
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"(?<=[^\n])\n(?=[^\n])", " ", text)
    return text.strip()


def chunk_page_text(page_text: str, page_no: int, approx_words=200, overlap_words=30):
    text = clean_linewraps(page_text)
    paragraphs = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    cur = ""
    cur_words = 0
    idx = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        w = len(p.split())
        if cur_words + w > approx_words and cur:
            chunks.append({"page": page_no, "chunk_id": f"p{page_no}_c{idx}", "text": cur.strip()})
            idx += 1
            keep = " ".join(cur.split()[-overlap_words:]) if overlap_words > 0 else ""
            cur = keep + " " + p
            cur_words = len(cur.split())
        else:
            cur += " " + p
            cur_words += w
    if cur.strip():
        chunks.append({"page": page_no, "chunk_id": f"p{page_no}_c{idx}", "text": cur.strip()})
    return chunks


# -----------------------
# Embeddings: transformers mean pooling
# -----------------------
def make_embedder(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()
    model.to(DEVICE)

    def embed_texts(texts: List[str]) -> np.ndarray:
        all_emb = []
        with torch.no_grad():
            for i in range(0, len(texts), EMBED_BATCH):
                batch = texts[i: i + EMBED_BATCH]
                enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
                input_ids = enc["input_ids"].to(DEVICE)
                attention_mask = enc["attention_mask"].to(DEVICE)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                last = out.last_hidden_state  # bs x seq x dim
                mask = attention_mask.unsqueeze(-1)
                summed = (last * mask).sum(1)
                denom = mask.sum(1).clamp(min=1e-9)
                emb = (summed / denom).cpu().numpy()
                # l2 normalize
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / np.clip(norms, 1e-12, None)
                all_emb.append(emb)
        return np.vstack(all_emb)

    return embed_texts, tokenizer


# -----------------------
# FAISS index (CPU)
# -----------------------
def build_faiss_index(emb_matrix: np.ndarray):
    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)
    return index


# -----------------------
# Query parsing helper
# -----------------------
def parse_query_for_filters(raw_query: str):
    """
    Allow queries like:
    - "Torque for brake caliper bolts"
    - "spec_type=Torque; component=Brake caliper bolt"
    Return: (spec_type or None, component or None, cleaned free-text query)
    """
    spec = None
    comp = None
    q = raw_query.strip()

    # direct kv pairs
    if "=" in q:
        parts = [p.strip() for p in re.split(r"[;|,]", q) if p.strip()]
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k in ("spec_type", "spec", "type"):
                    spec = v
                elif k in ("component", "comp", "item"):
                    comp = v
        # remaining free text
        q = " ".join([p for p in parts if "=" not in p]) or q

    # find known spec types by keyword
    for s in SPEC_TYPES:
        if re.search(r"\b" + re.escape(s.lower()) + r"\b", raw_query.lower()):
            spec = s
            break

    # if comp not provided, attempt to extract last noun phrase after 'for' or 'of'
    if comp is None:
        m = re.search(r"(?:for|of)\s+(.+)$", raw_query, re.I)
        if m:
            comp = m.group(1).strip()
            # remove trailing punctuation
            comp = comp.rstrip(".;")
    return spec, comp, q


# -----------------------
# LLM prompt generation
# -----------------------
def make_prompt_for_assignment(chunk_text: str, spec_hint: str = None, comp_hint: str = None):
    """
    Force the model to output EXACTLY the array of objects with exactly these keys:
    component, spec_type, value, unit
    Return single JSON array (even if empty).
    """
    spec_list_text = ", ".join(SPEC_TYPES)
    comp_hint_text = f' Component hint: "{comp_hint}".' if comp_hint else ""
    spec_hint_text = f' Spec_type hint: "{spec_hint}".' if spec_hint else ""

    prompt = f"""
You are an information extraction assistant. Extract at most one MOST-RELEVANT specification from the CHUNK below, returning a JSON array with exactly one object or an empty array if there is nothing relevant.

REQUIRED output format (exactly, no extra fields, keys in this order):
[
  {{
    "component": "<component name>",
    "spec_type": "<one of: {spec_list_text}>",
    "value": "<numeric value as string>",
    "unit": "<unit as string>"
  }}
]

Rules:
- Only return the JSON array. No explanation, no metadata, no extra keys.
- If there are multiple values in the chunk relating to the spec, pick the most relevant one and return a single object.
- Normalize the spec_type to one of the known types if possible (e.g., 'Torque').
- If the chunk mentions units in parentheses or multiple units, choose the primary unit (prefer Nm for torque when present).
- If no matching spec exists in the chunk, return [].

{spec_hint_text}{comp_hint_text}

CHUNK:
\"\"\"{chunk_text}\"\"\"
"""
    return prompt.strip()


# -----------------------
# LLM invoke (safe, single prompt)
# -----------------------
def run_llm_single_prompt(llm_tokenizer, llm_model, prompt: str, max_new_tokens=200) -> str:
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        out = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=llm_tokenizer.eos_token_id,
            pad_token_id=llm_tokenizer.pad_token_id
        )
    # decode only generation portion
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = llm_tokenizer.decode(gen, skip_special_tokens=True)
    return text.strip()


# -----------------------
# Postprocess LLM JSON -> final assignment fields (strings)
# -----------------------
def sanitize_and_pick(record_json_text: str):
    """
    Extract first JSON array from text and ensure it matches required schema.
    Return None or dict with 4 string fields.
    """
    try:
        start = record_json_text.index("[")
        end = record_json_text.rindex("]") + 1
        block = record_json_text[start:end]
        data = json.loads(block)
    except Exception:
        # fallback: try to find a single number/unit in text
        m = NUM_UNIT_RE.search(record_json_text)
        if m:
            val = m.group(1)
            unit = m.group(3) or ""
            return {"component": "", "spec_type": "", "value": str(val), "unit": unit}
        return None

    if not isinstance(data, list) or len(data) == 0:
        return None
    obj = data[0]  # take first (the model must return only one)
    # enforce schema keys and convert to string
    final = {}
    for k in ("component", "spec_type", "value", "unit"):
        v = obj.get(k) if isinstance(obj, dict) else None
        if v is None:
            # empty string is allowed but keep as ""
            final[k] = ""
        else:
            final[k] = str(v).strip()
    return final


# -----------------------
# Top-level pipeline
# -----------------------
def run_pipeline(pdf_paths: List[str], query: str, out_json_path: str):
    # 1) parse each PDF
    all_pages = []
    for p in pdf_paths:
        print("Parsing:", p)
        all_pages.extend(extract_text_from_pdf(p))

    print("Total pages:", len(all_pages))

    # 2) chunk all pages
    all_chunks = []
    for pg in all_pages:
        cs = chunk_page_text(pg["text"], page_no=pg["page_no"])
        for c in cs:
            c["source"] = ""  # reserved
        all_chunks.extend(cs)
    print("Total chunks:", len(all_chunks))

    # 3) embeddings
    print("Loading embedding model:", EMBED_MODEL)
    embed_fn, emb_tokenizer = make_embedder(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    print("Computing embeddings (this may take a bit)...")
    embeddings = embed_fn(texts).astype("float32")
    print("Embeddings shape:", embeddings.shape)

    # 4) FAISS index (CPU)
    index = build_faiss_index(embeddings)

    # 5) embed query
    spec_hint, comp_hint, cleaned_q = parse_query_for_filters(query)
    print(f"Query parsed -> spec_hint: {spec_hint}, comp_hint: {comp_hint}, cleaned_q: {cleaned_q}")
    q_emb = embed_fn([cleaned_q]).astype("float32")
    # ensure normalized (we normalized embeddings earlier)
    # do search
    D, I = index.search(q_emb, TOP_K)
    idxs = I[0].tolist()
    scores = D[0].tolist()
    # choose top chunk
    chosen = None
    for idx, sc in zip(idxs, scores):
        if idx < 0: continue
        chosen = all_chunks[idx]
        chosen_score = sc
        break
    if chosen is None:
        print("No chunk retrieved. Saving empty output.")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return

    print(f"Using top chunk (score={chosen_score:.4f}) from page {chosen['page']} ...")

    # 6) Load LLM (tokenizer + model)
    print("Loading LLM:", LLM_MODEL)
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=False)
        # try default load
        llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float16)
        # move to device if possible
        try:
            llm_model.to(DEVICE)
        except Exception as e:
            print("Could not .to(DEVICE) directly:", e)
    except Exception as e:
        print("LLM load error:", e)
        print("Attempting device_map='auto' fallback (requires transformers >=4.11)...")
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=False)
            llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto", torch_dtype=torch.float16)
        except Exception as e2:
            print("Fallback failed:", e2)
            raise

    # 7) Create prompt and call LLM
    prompt = make_prompt_for_assignment(chosen["text"], spec_hint=spec_hint, comp_hint=comp_hint)
    print("Prompt length:", len(prompt))
    raw_out = run_llm_single_prompt(llm_tokenizer, llm_model, prompt, max_new_tokens=180)
    print("Raw LLM output (truncated):", raw_out[:300].replace("\n", " "))

    # 8) sanitize and produce final single record
    final_rec = sanitize_and_pick(raw_out)
    if final_rec is None:
        print("No structured output found by LLM; returning empty array.")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return

    # If spec_hint present and model returned empty spec_type, fill it
    if final_rec.get("spec_type", "") == "" and spec_hint:
        final_rec["spec_type"] = spec_hint

    # If component empty, and comp_hint present, fill it
    if final_rec.get("component", "") == "" and comp_hint:
        final_rec["component"] = comp_hint

    # Ensure value is a simple numeric string (try to salvage from chunk if missing)
    if final_rec.get("value", "") == "":
        m = NUM_UNIT_RE.search(chosen["text"])
        if m:
            final_rec["value"] = str(m.group(1))
            final_rec["unit"] = m.group(3) or final_rec.get("unit", "")

    # Final output must be an array with a single object or empty array per assignment
    out_array = [final_rec] if final_rec else []
    # Save
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_array, f, indent=2)
    print("Saved output to:", out_json_path)
    print("Final record:", final_rec)


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="+", required=True, help="Paths to PDF(s) (put sample PDF(s) in pdfs/).")
    ap.add_argument("--query", required=True, help='Query string (e.g., "Torque for brake caliper bolts" or "spec_type=Torque; component=Brake caliper bolt").')
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "results.json"), help="Output JSON file path.")
    args = ap.parse_args()

    run_pipeline(args.pdf, args.query, args.out)
