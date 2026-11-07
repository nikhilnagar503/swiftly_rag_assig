/**
 * Simple RAG pipeline in one file.
 *
 *   upload -> EXTRACT text -> CHUNK -> EMBED -> STORE (vector store)
 *   ask    -> EMBED question -> RETRIEVE top-K by cosine similarity (= relevance score) -> GENERATE answer
 *
 * With OPENAI_API_KEY set: OpenAI embeddings + chat model.
 * Without a key: local hashed bag-of-words embeddings + extractive answer (demo fallback).
 */
import express from "express";
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const require = createRequire(import.meta.url);
// Import the inner file: the package index runs a debug routine when it has no parent module.
const pdfParse = require("pdf-parse/lib/pdf-parse.js") as (
  data: Buffer
) => Promise<{ text: string; numpages: number }>;

// ---------------------------------------------------------------- config
const PORT = Number(process.env.PORT ?? 4827);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1").replace(/\/$/, "");
const EMBED_MODEL = process.env.EMBED_MODEL ?? "text-embedding-3-small";
const CHAT_MODEL = process.env.CHAT_MODEL ?? "gpt-4o-mini";
const CHUNK_SIZE = 800; // characters per chunk
const CHUNK_OVERLAP = 150; // characters shared between neighbouring chunks
const LOCAL_DIMS = 512; // size of the local fallback vector
const STORE_FILE = path.join(process.cwd(), "data", "store.json");

const USE_OPENAI = OPENAI_API_KEY.length > 0;
const EMBEDDER_ID = USE_OPENAI ? EMBED_MODEL : "local-hash-512";

// ---------------------------------------------------------------- types
interface Chunk {
  id: string;
  docId: string;
  docName: string;
  index: number; // position of the chunk inside its document
  text: string;
  embedder: string; // which model produced `vector` (vectors from different models can't be compared)
  vector: number[];
}
interface DocInfo {
  id: string;
  name: string;
  chunks: number;
  chars: number;
  addedAt: string;
}
interface Hit {
  chunk: Chunk;
  score: number; // cosine similarity, higher = more relevant
}

// ---------------------------------------------------------------- 1. EXTRACT
async function extractText(buf: Buffer, filename: string): Promise<string> {
  const ext = path.extname(filename).toLowerCase();
  if (ext === ".pdf") {
    const parsed = await pdfParse(buf);
    return parsed.text;
  }
  if (ext === ".txt" || ext === ".md") return buf.toString("utf8");
  throw new Error(`Unsupported file type "${ext}". Use PDF, TXT or MD.`);
}

// ---------------------------------------------------------------- 2. CHUNK
/** Split text into ~CHUNK_SIZE pieces, preferring to cut at paragraph/sentence/word boundaries. */
function chunkText(raw: string): string[] {
  const text = raw.replace(/\r/g, "").replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + CHUNK_SIZE, text.length);
    if (end < text.length) {
      // look back (at most 30% of the chunk) for a natural break
      const window = text.slice(start, end);
      const floor = Math.floor(CHUNK_SIZE * 0.7);
      const cut = Math.max(
        window.lastIndexOf("\n\n"),
        window.lastIndexOf(". "),
        window.lastIndexOf("\n"),
        window.lastIndexOf(" ")
      );
      if (cut >= floor) end = start + cut + 1;
    }
    const piece = text.slice(start, end).trim();
    if (piece) chunks.push(piece);
    if (end >= text.length) break;
    start = Math.max(end - CHUNK_OVERLAP, start + 1); // step back to create the overlap
  }
  return chunks;
}

// ---------------------------------------------------------------- 3. EMBED
const STOPWORDS = new Set(
  "a an and are as at be by for from has have in is it its of on or that the this to was were will with you your i we they he she not but if then so do does did can could should would".split(" ")
);

function tokenize(s: string): string[] {
  return (s.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => !STOPWORDS.has(t));
}

function hashSlot(token: string): number {
  return crypto.createHash("md5").update(token).digest().readUInt32BE(0) % LOCAL_DIMS;
}

/** Offline fallback: hash words and word pairs into a fixed-size count vector, then L2-normalise. */
function localEmbed(text: string): number[] {
  const v = new Array<number>(LOCAL_DIMS).fill(0);
  const toks = tokenize(text);
  for (let i = 0; i < toks.length; i++) {
    v[hashSlot(toks[i])] += 1;
    if (i + 1 < toks.length) v[hashSlot(toks[i] + " " + toks[i + 1])] += 1;
  }
  for (let i = 0; i < v.length; i++) v[i] = Math.log1p(v[i]); // dampen very frequent words
  return normalize(v);
}

function normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

/** POST https://api.openai.com/v1/embeddings, in batches of 100. */
async function openaiEmbed(texts: string[]): Promise<number[][]> {
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += 100) {
    const batch = texts.slice(i, i + 100);
    const res = await fetch(`${OPENAI_BASE_URL}/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
      body: JSON.stringify({ model: EMBED_MODEL, input: batch }),
    });
    if (!res.ok) throw new Error(`OpenAI embeddings failed (${res.status}): ${await res.text()}`);
    const json = (await res.json()) as { data: { index: number; embedding: number[] }[] };
    json.data.sort((a, b) => a.index - b.index).forEach((d) => out.push(d.embedding));
  }
  return out;
}

async function embed(texts: string[]): Promise<number[][]> {
  return USE_OPENAI ? openaiEmbed(texts) : texts.map(localEmbed);
}

// ---------------------------------------------------------------- 4. STORE (tiny vector DB)
class VectorStore {
  entries: Chunk[] = [];
  docs: DocInfo[] = [];

  constructor(private file: string) {
    if (fs.existsSync(file)) {
      const saved = JSON.parse(fs.readFileSync(file, "utf8")) as { entries: Chunk[]; docs: DocInfo[] };
      this.entries = saved.entries ?? [];
      this.docs = saved.docs ?? [];
    }
  }

  private save() {
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    const tmp = this.file + ".tmp";
    fs.writeFileSync(tmp, JSON.stringify({ entries: this.entries, docs: this.docs }));
    fs.renameSync(tmp, this.file);
  }

  add(doc: DocInfo, chunks: Chunk[]) {
    this.docs.push(doc);
    this.entries.push(...chunks);
    this.save();
  }

  remove(docId: string): boolean {
    const before = this.docs.length;
    this.docs = this.docs.filter((d) => d.id !== docId);
    this.entries = this.entries.filter((c) => c.docId !== docId);
    this.save();
    return this.docs.length < before;
  }

  /** Brute-force nearest neighbours. Vectors are unit length, so cosine similarity = dot product. */
  search(queryVec: number[], embedder: string, topK: number): Hit[] {
    return this.entries
      .filter((c) => c.embedder === embedder) // never compare vectors from different models
      .map((chunk) => ({ chunk, score: dot(chunk.vector, queryVec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

const store = new VectorStore(STORE_FILE);

// ---------------------------------------------------------------- 5. RETRIEVE
async function retrieve(question: string, topK: number): Promise<Hit[]> {
  const [qVec] = await embed([question]);
  return store.search(normalize(qVec), EMBEDDER_ID, topK);
}

// ---------------------------------------------------------------- 6. GENERATE
const SYSTEM_PROMPT =
  "You answer questions using ONLY the numbered context passages provided. " +
  "Cite the passages you used like [1] or [2]. " +
  "If the answer is not in the context, say you could not find it in the uploaded documents. Be concise.";

async function generate(question: string, hits: Hit[]): Promise<string> {
  if (hits.length === 0) return "No relevant content found. Upload a document first.";

  if (!USE_OPENAI) return extractiveAnswer(question, hits);

  const context = hits.map((h, i) => `[${i + 1}] (${h.chunk.docName})\n${h.chunk.text}`).join("\n\n");
  const res = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: CHAT_MODEL,
      temperature: 0.2,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` },
      ],
    }),
  });
  if (!res.ok) throw new Error(`OpenAI chat failed (${res.status}): ${await res.text()}`);
  const json = (await res.json()) as { choices: { message: { content: string } }[] };
  return json.choices[0]?.message?.content?.trim() ?? "";
}

/** No-LLM fallback: return the sentences from the top chunks that share the most words with the question. */
function extractiveAnswer(question: string, hits: Hit[]): string {
  const qTerms = new Set(tokenize(question));
  const scored: { text: string; n: number; score: number }[] = [];
  hits.forEach((h, i) => {
    const sentences = h.chunk.text.split(/(?<=[.!?])\s+|\n+/).filter((s) => s.trim().length > 20);
    for (const s of sentences) {
      const overlap = tokenize(s).filter((t) => qTerms.has(t)).length;
      if (overlap > 0) scored.push({ text: s.trim(), n: i + 1, score: overlap });
    }
  });
  scored.sort((a, b) => b.score - a.score);
  const seen = new Set<string>();
  const best = scored.filter((s) => !seen.has(s.text) && seen.add(s.text)).slice(0, 3);
  if (best.length === 0) return "I could not find a clear answer in the uploaded documents.";
  return (
    "(local mode - no LLM key set, showing the most relevant sentences)\n" +
    best.map((s) => `${s.text} [${s.n}]`).join("\n")
  );
}

// ---------------------------------------------------------------- HTTP API
const app = express();
app.use(express.static(path.join(process.cwd(), "public")));
app.use(express.json());

app.get("/api/status", (_req, res) => {
  res.json({
    mode: USE_OPENAI ? "openai" : "local",
    embedder: EMBEDDER_ID,
    chatModel: USE_OPENAI ? CHAT_MODEL : "extractive (no LLM)",
    docs: store.docs.length,
    chunks: store.entries.length,
  });
});

app.get("/api/docs", (_req, res) => res.json(store.docs));

// Upload: raw file bytes in the body, filename in ?name=
app.post("/api/upload", express.raw({ type: () => true, limit: "50mb" }), async (req, res) => {
  try {
    const name = String(req.query.name ?? "").trim();
    if (!name) return res.status(400).json({ error: "Missing ?name=<filename>" });
    if (!Buffer.isBuffer(req.body) || req.body.length === 0) return res.status(400).json({ error: "Empty file" });

    const text = await extractText(req.body, name);
    const pieces = chunkText(text);
    if (pieces.length === 0) return res.status(422).json({ error: "No extractable text (is the PDF a scan?)" });

    const vectors = await embed(pieces);
    const docId = crypto.randomUUID();
    const chunks: Chunk[] = pieces.map((t, i) => ({
      id: `${docId}:${i}`,
      docId,
      docName: name,
      index: i,
      text: t,
      embedder: EMBEDDER_ID,
      vector: normalize(vectors[i]),
    }));
    const doc: DocInfo = { id: docId, name, chunks: chunks.length, chars: text.length, addedAt: new Date().toISOString() };
    store.add(doc, chunks);
    res.json(doc);
  } catch (e) {
    res.status(500).json({ error: (e as Error).message });
  }
});

app.delete("/api/docs/:id", (req, res) => {
  res.json({ removed: store.remove(req.params.id) });
});

app.post("/api/ask", async (req, res) => {
  try {
    const question = String(req.body?.question ?? "").trim();
    const topK = Math.min(Math.max(Number(req.body?.topK ?? 4), 1), 10);
    if (!question) return res.status(400).json({ error: "Missing question" });

    const hits = await retrieve(question, topK);
    const answer = await generate(question, hits);
    res.json({
      answer,
      sources: hits.map((h, i) => ({
        n: i + 1,
        docName: h.chunk.docName,
        chunkIndex: h.chunk.index,
        score: Number(h.score.toFixed(4)),
        text: h.chunk.text,
      })),
    });
  } catch (e) {
    res.status(500).json({ error: (e as Error).message });
  }
});

app.listen(PORT, () => {
  console.log(`RAG server: http://localhost:${PORT}`);
  console.log(`Mode: ${USE_OPENAI ? `OpenAI (${EMBED_MODEL} + ${CHAT_MODEL})` : "local fallback (set OPENAI_API_KEY for real embeddings)"}`);
});
