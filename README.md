# Simple RAG pipeline (TypeScript)

Everything is in `server.ts`; the UI is `public/index.html`.

```
upload  -> EXTRACT (pdf-parse) -> CHUNK (800 chars, 150 overlap) -> EMBED -> STORE (data/store.json)
ask     -> EMBED question -> RETRIEVE top-K by cosine similarity (the relevance score) -> GENERATE answer with [n] citations
```

## Run

```bash
npm install
export OPENAI_API_KEY=sk-...   # optional, see below
npm start                      # http://localhost:4827
npm run typecheck
```

## Modes

| | Embeddings | Answer |
|---|---|---|
| `OPENAI_API_KEY` set | `text-embedding-3-small` (1536-dim) via `POST /v1/embeddings` | `gpt-4o-mini` via `POST /v1/chat/completions`, answers only from retrieved chunks |
| no key | local hashed bag-of-words (512-dim, keyword match only) | extractive: most relevant sentences |

## Storage

Only text chunks + vectors are kept (not the PDF file), in `data/store.json` (git-ignored), loaded into memory by `VectorStore`.
Each chunk: `{ id, docId, docName, index, text, embedder, vector }`. Chunks from a different embedder than the current one are skipped at search time.
The hand-written `VectorStore` is a stand-in; Chroma / pgvector / Pinecone would replace that class only.

## Environment variables

`OPENAI_API_KEY`, `OPENAI_BASE_URL` (default `https://api.openai.com/v1`), `EMBED_MODEL`, `CHAT_MODEL`, `PORT` (default 4827).

## API

- `POST /api/upload?name=file.pdf` (raw bytes body)
- `GET /api/docs`, `DELETE /api/docs/:id`, `GET /api/status`
- `POST /api/ask` with `{ "question": "...", "topK": 4 }`
