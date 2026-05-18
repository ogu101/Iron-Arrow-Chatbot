# Iron Arrow RAG Pipeline

A two-script RAG system built on:
- **PyPDF + LangChain** — text extraction & chunking
- **Voyage AI** (`voyage-4`) — embeddings
- **PostgreSQL + pgvector** — vector storage & retrieval
- **Claude** (`claude-sonnet-4`) — answer generation

```
┌─────────────────────────────────────────────────────────┐
│  INGESTION  (ingest.py)                                 │
│                                                         │
│  PDF  →  PyPDFLoader  →  RecursiveCharacterTextSplitter │
│       →  Voyage AI embed (document)  →  pgvector store  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  QUERY  (query.py)                                      │
│                                                         │
│  Question  →  Voyage AI embed (query)                   │
│           →  pgvector cosine search (top-k chunks)      │
│           →  Prompt with context  →  Claude  →  Answer  │
└─────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Spin up PostgreSQL with pgvector
The easiest path is Docker:
```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 3. Set environment variables
```bash
export VOYAGE_API_KEY="your-voyage-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DATABASE_URL="postgresql://postgres:secret@localhost:5432/ragdb"
```

---

## Usage

### Ingest the PDF (run once)
```bash
python ingest.py --pdf Iron_Arrow_-_A_History_3rd_ed.indb
```
> The file is a PDF despite the `.indb` extension — the script handles it automatically.

Expected output:
```
[1/4] Loading and chunking: Iron_Arrow_-_A_History_3rd_ed.indb
      555 chunks created
[2/4] Embedding 555 chunks with Voyage AI (voyage-4)
      555 embeddings generated (dim=1024)
[3/4] Setting up PostgreSQL / pgvector schema
[4/4] Storing chunks in table 'iron_arrow_chunks'
      Done. Ingestion complete.
```

### Query (interactive REPL)
```bash
python query.py
```

### Query (single question)
```bash
python query.py --question "Who founded Iron Arrow and when?"
python query.py --question "What are the requirements to become a member?" --top-k 8
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| `chunk_size=1000, overlap=100` | Balances context richness vs. retrieval precision for a ~240-page history book |
| `input_type="document"` on ingest | Voyage AI's recommended mode for passage-level content |
| `input_type="query"` on retrieval | Asymmetric embedding — queries and docs live in the same space but are encoded differently |
| `voyage-4` | Current Voyage AI flagship model; 1024-dim vectors |
| Cosine similarity (`<=>` operator) | Standard for normalized text embeddings |
| `ivfflat` index with `lists=100` | Appropriate for ~555 vectors; switch to `hnsw` for 100k+ rows |
| `TOP_K=5` | 5 × 1000-char chunks ≈ 5000 tokens of context — well within Claude's window |
| System prompt constrains to context | Prevents hallucination on out-of-scope questions |

---

## Notes

- **Re-ingestion**: the script appends rows. To start fresh, `TRUNCATE TABLE iron_arrow_chunks` before re-running.
- **Scaling the index**: if you add significantly more documents, increase `lists` in the `ivfflat` index proportionally (rule of thumb: `sqrt(n_rows)`), or switch to `hnsw` for better recall at scale.
- **Image content**: the PDF contains images. PyPDF extracts text only; images are skipped. If image captions or charts matter, add a vision extraction step with `pdf2image` + Claude's vision API before chunking.
