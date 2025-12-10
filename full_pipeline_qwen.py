#!/usr/bin/env python3
"""
full_pipeline_qwen.py
Final single-file RAG + local Qwen pipeline.
- Local model folder: ./qwen_local (change MODEL_PATH below if different)
- Default behavior: SPEC extraction (unless query starts with 'S:' or 'D:')
- Ultra-strict JSON enforcement with automatic repair/retry
- Fallback: safe regex spec extractor (spec-only)
"""
import os
import re
import json
import argparse
from collections import Counter
from typing import List, Dict, Any
from tqdm import tqdm

# Core libs
try:
    import fitz  # PyMuPDF
except Exception as e:
    raise ImportError("PyMuPDF not installed. pip install pymupdf") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("sentence-transformers not installed. pip install sentence-transformers") from e

try:
    import faiss
except Exception as e:
    raise ImportError("faiss not installed. pip install faiss-cpu (or use conda)") from e

# transformers (required for LLM)
HAS_TRANSFORMERS = True
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    HAS_TRANSFORMERS = False

# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = "./qwen7_local"            # << confirmed by you
HF_CACHE_DIR = "./models"              # HF cache for tokenizers etc (kept local)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS_PER_CHUNK = 1500
TOP_K = 6

# Regexes & helpers for cleaning & chunking
PAGE_NUM_PAT = re.compile(r'(^|\s)(\d+\s*[-–]\s*\d+|\d+/\d+|\bPage\s*\d+\b|\bpg\.\s*\d+\b|\b\d+\b)(\s|$)', flags=re.IGNORECASE)
ALL_CAPS_LINE = re.compile(r'^[A-Z0-9 .:/\-\(\)\'\"%]{3,}$')
MULTI_SPACES = re.compile(r'\s{2,}')
HYPHEN_LINE_END = re.compile(r'(-)\s*$')
TRAILING_DOT = re.compile(r'[\.!?]\s*$')

UNIT_NORMALIZATIONS = {
    r'\bN·m\b': 'Nm',
    r'\bN ?m\b': 'Nm',
    r'\bnewton-?metre?s?\b': 'Nm',
    r'\bkgf·m\b': 'kgf·m',
    r'\bkgf ?m\b': 'kgf·m',
    r'\bmm\b': 'mm',
    r'\bcm\b': 'cm',
    r'\bin-?l?b?s?\b': 'in-lb',
    r'\bin ?lb\b': 'in-lb',
    r'\bl\b': 'L',
}

SUSPECT_HEADER_WORDS = ['SERVICE SPECIFICATIONS', 'SPECIFICATIONS', 'BRAKES', 'TORQUE', 'SERVICE MANUAL']

# Ultra-strict master prompt: instruct model to print JSON ONLY and wrap result with <JSON>... </JSON>
MASTER_PROMPT = """
You are an automotive service-manual SPEC extractor.
Use ONLY the context provided below. Never invent values.

OUTPUT RULES (MANDATORY):
- You MUST output ONLY valid JSON.
- Wrap JSON ONLY inside:
  <JSON>
  [...]
  </JSON>
- Absolutely no explanation, no notes, no commentary.
- JSON must be a SINGLE array of SPEC objects.
- Each object MUST contain EXACTLY these four fields:

{
 "component": "string",
 "spec_type": "string",
 "value": "string",
 "unit": "string"
}

If context does not provide any usable SPEC values, output:

<JSON>
{"error": "Insufficient data"}
</JSON>

CONTEXT:
\"\"\" 
{context}
\"\"\"

USER QUERY:
{query}

SPEC FORMAT (REQUIRED):

<JSON>
[
 {
  "component": "string",
  "spec_type": "string",
  "value": "string",
  "unit": "string"
 }
]
</JSON>

Return ONLY the JSON. No text outside <JSON>…</JSON>.
"""



# ---------------------------
# PDF extraction & cleaning
# ---------------------------
def extract_pages_pymupdf(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for p in range(doc.page_count):
        page = doc.load_page(p)
        txt = page.get_text("text")
        txt = txt.replace('\r\n', '\n').replace('\r', '\n')
        pages.append(txt)
    doc.close()
    return pages

def candidate_headers_from_pages(pages: List[str], top_k=1) -> List[str]:
    counter = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not lines:
            continue
        for ln in lines[:3] + lines[-3:]:
            normalized = MULTI_SPACES.sub(' ', ln)
            if len(normalized) < 3 or len(normalized) > 120:
                continue
            counter[normalized] += 1
    candidates = [k for k, v in counter.most_common(top_k)]
    filtered = []
    for c in candidates:
        if any(w.upper() in c.upper() for w in SUSPECT_HEADER_WORDS) or counter[c] > 1:
            filtered.append(c)
    for c in candidates:
        if c not in filtered:
            filtered.append(c)
    return filtered

def remove_repeated_headers(text: str, header_candidates: List[str]) -> str:
    lines = text.splitlines()
    out = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            out.append('')
            continue
        skip = False
        for h in header_candidates:
            if not h:
                continue
            if stripped == h:
                skip = True
                break
            if (len(h) > 6 and h in stripped) or (len(stripped) > 6 and stripped in h):
                skip = True
                break
        if skip:
            continue
        if PAGE_NUM_PAT.fullmatch(stripped):
            continue
        out.append(ln)
    return '\n'.join(out)

def fix_hyphenation_and_linejoins(page_text: str) -> str:
    lines = page_text.splitlines()
    cleaned_lines = []
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            cleaned_lines.append('')
            i += 1
            continue
        if MULTI_SPACES.search(ln):
            cleaned_lines.append(MULTI_SPACES.sub('  ', ln.strip()))
            i += 1
            continue
        if ALL_CAPS_LINE.match(ln.strip()):
            cleaned_lines.append(ln.strip())
            i += 1
            continue
        if HYPHEN_LINE_END.search(ln):
            next_ln = lines[i+1].lstrip() if i+1 < len(lines) else ''
            merged = re.sub(r'(-)\s*$', '', ln) + next_ln
            cleaned_lines.append(merged.strip())
            i += 2
            continue
        if i+1 < len(lines):
            next_ln = lines[i+1].lstrip()
            if TRAILING_DOT.search(ln.strip()):
                cleaned_lines.append(ln.strip())
                i += 1
                continue
            if next_ln and next_ln[0].islower():
                joined = (ln + ' ' + next_ln).strip()
                cleaned_lines.append(joined)
                i += 2
                continue
            if len(next_ln) <= 40 and not ALL_CAPS_LINE.match(next_ln):
                joined = (ln + ' ' + next_ln).strip()
                cleaned_lines.append(joined)
                i += 2
                continue
        cleaned_lines.append(ln.strip())
        i += 1
    return '\n'.join([MULTI_SPACES.sub(' ', l) for l in cleaned_lines])

def normalize_units(text: str) -> str:
    out = text
    for pat, repl in UNIT_NORMALIZATIONS.items():
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    out = re.sub(r'[–—]', '-', out)
    out = MULTI_SPACES.sub(' ', out)
    return out

def clean_page_text(raw_page: str, header_candidates: List[str]) -> str:
    t = remove_repeated_headers(raw_page, header_candidates)
    t = fix_hyphenation_and_linejoins(t)
    t = normalize_units(t)
    t = '\n'.join([ln.rstrip() for ln in t.splitlines()])
    return t

# ---------------------------
# Chunking
# ---------------------------
def chunk_text_by_headings(pages_clean: List[Dict[str, Any]], max_chars=MAX_CHARS_PER_CHUNK) -> List[Dict[str, Any]]:
    chunks = []
    for p in pages_clean:
        page_no = p['page_no']
        text = p['cleaned']
        lines = [ln for ln in text.splitlines() if ln.strip()]
        current_section = "UNKNOWN"
        buffer = []
        for ln in lines:
            if ALL_CAPS_LINE.match(ln) and len(ln) <= 120:
                if buffer:
                    chunk_text = '\n'.join(buffer).strip()
                    if len(chunk_text) > max_chars:
                        for i in range(0, len(chunk_text), max_chars):
                            sub = chunk_text[i:i+max_chars]
                            chunks.append({"text": sub.strip(),"page": page_no,"section": current_section})
                    else:
                        chunks.append({"text": chunk_text,"page": page_no,"section": current_section})
                    buffer = []
                current_section = ln.strip()
                continue
            buffer.append(ln)
        if buffer:
            chunk_text = '\n'.join(buffer).strip()
            if len(chunk_text) > max_chars:
                for i in range(0, len(chunk_text), max_chars):
                    sub = chunk_text[i:i+max_chars]
                    chunks.append({"text": sub.strip(),"page": page_no,"section": current_section})
            else:
                chunks.append({"text": chunk_text,"page": page_no,"section": current_section})
    merged = []
    for c in chunks:
        if merged and len(c['text']) < 200:
            merged[-1]['text'] += "\n" + c['text']
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c['id'] = i
    return merged

# ---------------------------
# Embedding + FAISS
# ---------------------------
def build_embeddings_and_faiss(chunks: List[Dict[str, Any]], outdir: str, embed_model_name=EMBED_MODEL):
    print("🔁 Building embeddings with", embed_model_name)
    model = SentenceTransformer(embed_model_name)
    texts = [c['text'] for c in chunks]
    embs = model.encode(texts, convert_to_numpy=True)
    embs = embs.astype('float32')
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    faiss_path = os.path.join(outdir, "faiss_index.bin")
    faiss.write_index(index, faiss_path)
    chunks_path = os.path.join(outdir, "chunks.json")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("✅ FAISS index saved to", faiss_path)
    print("✅ Chunks metadata saved to", chunks_path)
    return faiss_path, chunks_path

# ---------------------------
# Retrieval
# ---------------------------
def load_index_and_chunks(outdir: str):
    faiss_path = os.path.join(outdir, "faiss_index.bin")
    chunks_path = os.path.join(outdir, "chunks.json")
    if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("Index or chunks not found. Build first with --mode build")
    index = faiss.read_index(faiss_path)
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return index, chunks

def retrieve(query: str, outdir: str, top_k: int = TOP_K, embed_model_name=EMBED_MODEL):
    index, chunks = load_index_and_chunks(outdir)
    model = SentenceTransformer(embed_model_name)
    qvec = model.encode([query], convert_to_numpy=True).astype('float32')
    dist, idx = index.search(qvec, top_k)
    results = []
    for rank, i in enumerate(idx[0]):
        if i < 0 or i >= len(chunks):
            continue
        c = chunks[i]
        results.append({
            "rank": rank+1,
            "score": float(dist[0][rank]),
            "id": c.get("id"),
            "page": c.get("page"),
            "section": c.get("section"),
            "text": c.get("text")
        })
    return results

# ---------------------------
# JSON extraction & repair helpers
# ---------------------------
def extract_json_substring(s: str):
    """Find the first JSON substring between <JSON>...</JSON> or first {...} / [...] block."""
    # prefer explicit markers
    m = re.search(r"<JSON>(.*)</JSON>", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: find first { ... } or [ ... ] balanced
    start_idx = None
    for i, ch in enumerate(s):
        if ch in '[{':
            start_idx = i
            break
    if start_idx is None:
        return None
    stack = []
    for j in range(start_idx, len(s)):
        ch = s[j]
        if ch in '[{':
            stack.append(ch)
        elif ch in ']}':
            if not stack:
                return None
            opening = stack.pop()
            if (opening == '{' and ch != '}') or (opening == '[' and ch != ']'):
                return None
            if not stack:
                return s[start_idx:j+1]
    return None

def safe_json_load(s: str):
    """Try multiple strategies to parse and repair JSON returned by model."""
    if s is None:
        return None
    # quick direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # try to extract JSON substring between markers or first balanced block
    sub = extract_json_substring(s)
    if not sub:
        return None
    # try direct
    try:
        return json.loads(sub)
    except Exception:
        # attempt simple repairs: single quotes, trailing commas
        rep = sub.replace("'", '"')
        rep = re.sub(r",\s*}", "}", rep)
        rep = re.sub(r",\s*]", "]", rep)
        try:
            return json.loads(rep)
        except Exception:
            return None

# ---------------------------
# LLM call with strict-retry semantics
# ---------------------------
def call_local_qwen_strict(context_text: str, query: str, model_path: str = MODEL_PATH, cache_dir: str = HF_CACHE_DIR, max_retries: int = 2):
    """Call local Qwen, enforce JSON-only output. Will attempt a repair prompt if first output is invalid."""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers/torch not available.")
    # load tokenizer & model once (cached)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 cache_dir=cache_dir,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True)
    model.eval()
    device = next(model.parameters()).device

    system_prompt = MASTER_PROMPT.format(context=context_text, query=query)

    # try initial generation
    def gen(prompt_text):
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        return decoded

    # 1st attempt
    try:
        raw = gen(system_prompt)
        parsed = safe_json_load(raw)
        if parsed is not None:
            return parsed, raw
    except Exception as e:
        # allow retry below
        raw = None

    # 2nd attempt: ask model to OUTPUT ONLY JSON between tags (explicit repair instruction)
    repair_prompt = system_prompt + "\n\nIMPORTANT: You must now output ONLY the JSON wrapped inside <JSON> ... </JSON> and nothing else."
    try:
        raw2 = gen(repair_prompt)
        parsed2 = safe_json_load(raw2)
        if parsed2 is not None:
            return parsed2, raw2
    except Exception:
        raw2 = None

    # optionally additional retries (one more)
    if max_retries > 2:
        try:
            raw3 = gen(repair_prompt)
            parsed3 = safe_json_load(raw3)
            if parsed3 is not None:
                return parsed3, raw3
        except Exception:
            raw3 = None

    # nothing worked
    return None, raw or raw2

# ---------------------------
# Regex fallback for SPEC mode (clean)
# ---------------------------
SPEC_REGEX_PATTERNS = [
    re.compile(r'(?P<comp>[A-Za-z0-9 \-\/\(\)\.\:]{3,120}?)[:\s\-–=]{1,6}\s*(?P<val>\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?)\s*(?P<unit>Nm|N ?m|mm|cm|kgf·m|kgf ?m|in-?lb|in ?lb|L|litre?s?|psi|bar)?\b', flags=re.IGNORECASE),
    re.compile(r'(?P<comp>[A-Za-z0-9 \-\/\(\)\.]{3,80})\s{2,}(?P<val>\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?)\s*(?P<unit>Nm|mm|kgf·m|in-?lb|L|psi|bar)?', flags=re.IGNORECASE),
]

def regex_spec_extract(text: str):
    results = []
    seen = set()
    for pat in SPEC_REGEX_PATTERNS:
        for m in pat.finditer(text):
            comp = (m.groupdict().get('comp') or "").strip(" .:-\n\t")
            val = (m.groupdict().get('val') or "").replace(',', '.').strip()
            unit = (m.groupdict().get('unit') or "").strip()
            if not comp or not val:
                continue
            key = (comp.lower(), val, unit.lower())
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "component": comp,
                "spec_type": guess_spec_type_from_component(comp),
                "value": val,
                "unit": normalize_unit_str(unit)
            })
    return results

def normalize_unit_str(u: str) -> str:
    if not u:
        return ""
    u = u.strip().replace('N m','Nm').replace('N·m','Nm').replace('inlb','in-lb')
    return u

def guess_spec_type_from_component(comp: str) -> str:
    lc = comp.lower()
    if 'torque' in lc or 'tighten' in lc or 'tightening' in lc:
        return "Torque"
    if 'capacity' in lc or 'liter' in lc or re.search(r'\bL\b', comp):
        return "Capacity"
    if 'thickness' in lc or 'pad' in lc:
        return "Pad Thickness"
    if 'clearance' in lc or 'gap' in lc:
        return "Clearance"
    return "Specification"

# ---------------------------
# Orchestration: build / query
# ---------------------------
def build_pipeline(pdf_folder: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    pages_clean = []
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError("No pdf files found in folder: " + pdf_folder)
    for pdf in tqdm(pdf_files, desc="Extracting PDFs"):
        pages = extract_pages_pymupdf(pdf)
        header_cands = candidate_headers_from_pages(pages, top_k=20)
        for i, raw in enumerate(pages):
            cleaned = clean_page_text(raw, header_cands)
            pages_clean.append({"pdf_file": os.path.basename(pdf),"page_no": i+1,"cleaned": cleaned})
    # save cleaned
    with open(os.path.join(outdir, "pages_cleaned.json"), 'w', encoding='utf-8') as f:
        json.dump(pages_clean, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "cleaned_full.txt"), 'w', encoding='utf-8') as f:
        f.write("\n\n".join(p['cleaned'] for p in pages_clean))
    print("✅ Cleaned pages saved.")
    chunks = chunk_text_by_headings(pages_clean, max_chars=MAX_CHARS_PER_CHUNK)
    print(f"✅ Created {len(chunks)} chunks.")
    with open(os.path.join(outdir, "chunks_debug.json"), 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    build_embeddings_and_faiss(chunks, outdir, embed_model_name=EMBED_MODEL)
    print("Build finished.")

def query_and_extract(user_query: str, outdir: str, top_k: int = TOP_K, out_file: str = None):
    # Mode detection by prefix
    mode = "spec"
    q_clean = user_query.strip()
    

    print(f"Mode = {mode}. Query => {q_clean}")

    results = retrieve(q_clean, outdir, top_k=top_k, embed_model_name=EMBED_MODEL)
    if not results:
        print("No chunks retrieved. Exiting.")
        return

    combined_context = "\n\n".join([f"--- Page: {r['page']} Section: {r['section']} ---\n{r['text']}" for r in results])

    # prefer strict LLM extraction (only for environments where transformers are available)
    final_output = None
    if HAS_TRANSFORMERS:
        try:
            parsed, raw_text = call_local_qwen_strict(combined_context, q_clean, model_path=MODEL_PATH, cache_dir=HF_CACHE_DIR, max_retries=3)
            if parsed is None:
                raise ValueError("LLM did not return parseable JSON")
            # If mode == spec, coerce output to list of spec objects
            # STRICT: final_output must be a single array
            if isinstance(parsed, list):
                final_output = parsed
            elif isinstance(parsed, dict):
                if "error" in parsed:
                    final_output = parsed
                else:
                    final_output = [parsed]
            else:
                final_output = {"error": "Invalid JSON structure"}

                # parsed can be list or object
                if isinstance(parsed, dict):
                    # maybe dict contains array under some key
                    # find first list value
                    arr = None
                    for v in parsed.values():
                        if isinstance(v, list):
                            arr = v
                            break
                    if arr is None:
                        # maybe single object representing one spec
                        arr = [parsed]
                    parsed = arr
                # sanitize items
                sanitized = []
                for it in parsed:
                    if not isinstance(it, dict):
                        continue
                    sanitized.append({
                        "component": str(it.get("component","")).strip(),
                        "spec_type": str(it.get("spec_type","")).strip(),
                        "value": str(it.get("value","")).strip(),
                        "unit": str(it.get("unit","")).strip()
                    })
                final_output = sanitized
     else:
                final_output = parsed
            print("✅ LLM provided structured JSON.")
        except Exception as e:
            print("⚠️ LLM failed or JSON invalid:", str(e))

    if final_output is None:
        # fallback: SPEC-only regex extraction
        if mode != "spec":
            print("No reliable LLM output and mode is not SPEC. Returning insufficient data.")
            final_output = {"error": "Insufficient LLM output and fallback only supports SPEC mode."}
        else:
            agg_text = "\n\n".join([r['text'] for r in results])
            specs = regex_spec_extract(agg_text)
            # optionally prioritize matches that mention key terms from query
            query_terms = re.findall(r'\w+', q_clean.lower())
            def score_spec(s):
                comp = s['component'].lower()
                return sum(1 for t in query_terms if t in comp)
            specs_sorted = sorted(specs, key=score_spec, reverse=True)
            final_output = specs_sorted
            print(f"✅ Regex fallback returned {len(final_output)} items.")

    out_json = json.dumps(final_output, ensure_ascii=False, indent=2)
    if out_file:
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(out_json)
        print(f"✅ Output saved to {out_file}")
    else:
        print("\n--- OUTPUT ---\n")
        print(out_json)
    return final_output

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "query"], required=True, help="build or query")
    parser.add_argument("--pdf_folder", default="./pdfs", help="folder with PDF(s)")
    parser.add_argument("--outdir", default="./data", help="where to save artifacts")
    parser.add_argument("--query", type=str, help="query text (use quotes)")
    parser.add_argument("--query_file", type=str, help="path to .txt file containing query")
    parser.add_argument("--topk", type=int, default=TOP_K, help="top-k chunks to retrieve")
    parser.add_argument("--out", type=str, help="file to save final JSON output")
    args = parser.parse_args()

    if args.mode == "build":
        build_pipeline(args.pdf_folder, args.outdir)
    elif args.mode == "query":
        if args.query_file:
            if not os.path.exists(args.query_file):
                print("Query file not found:", args.query_file)
                return
            with open(args.query_file, 'r', encoding='utf-8') as f:
                q = f.read().strip()
        else:
            if not args.query:
                print("Please provide --query or --query_file")
                return
            q = args.query.strip()
        query_and_extract(q, args.outdir, top_k=args.topk, out_file=args.out)

if __name__ == "__main__":
    main()
