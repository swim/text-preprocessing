/**
 * A BERT WordPiece tokenizer and a static embedder ('liquidau-static/1'), for edge scorers that need
 * meaning without a transformer. The tokenizer reproduces Hugging Face tokenizers' BertNormalizer
 * (clean text, CJK padding, accent stripping, lowercasing), BertPreTokenizer (whitespace and
 * punctuation) and WordPiece (greedy longest match, '##' continuations, max 100 characters per word)
 * from a tokenizer.json. The static embedder averages precomputed token vectors (Model2Vec-style):
 * no special tokens, unknown tokens dropped, at most maxTokens tokens, unit length.
 *
 * The host supplies the bytes (tokenizer.json and the weight table): no I/O here, no dependencies.
 */
import { sha256Hex } from './digest.ts';

export const STATIC_SCHEMA = 'liquidau-static/1';

export interface WordPieceTokenizer {
  /** Token ids for a text, without special tokens. */
  encode(text: string): number[];
  readonly unkId: number;
  readonly vocabSize: number;
}

export interface StaticIdentity {
  schema: typeof STATIC_SCHEMA;
  /** e.g. 'minishlab/potion-base-8M'. */
  modelId: string;
  dimensions: number;
  maxTokens: number;
  /** SHA-256 of the exact tokenizer.json bytes and of the float32 weight table's bytes. */
  tokenizerSha256: string;
  weightsSha256: string;
}

export interface StaticEmbedder {
  readonly identity: Readonly<StaticIdentity>;
  embed(text: string): Float32Array;
}

interface TokenizerJson {
  normalizer?: { type?: string; clean_text?: boolean; handle_chinese_chars?: boolean; strip_accents?: boolean | null; lowercase?: boolean };
  pre_tokenizer?: { type?: string };
  model?: { type?: string; unk_token?: string; continuing_subword_prefix?: string; max_input_chars_per_word?: number; vocab?: Record<string, number> };
}

const isControl = (ch: string) => {
  if (ch === '\t' || ch === '\n' || ch === '\r') return false;
  return /\p{C}/u.test(ch);
};
const isWhitespace = (ch: string) => ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r' || /\p{Zs}/u.test(ch);
const isChinese = (cp: number) =>
  (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b73f) ||
  (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) || (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);
const isPunctuation = (ch: string) => {
  const cp = ch.codePointAt(0)!;
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return /\p{P}/u.test(ch);
};

/** A WordPiece tokenizer from a parsed BERT tokenizer.json; throws on anything else. */
export function createWordPiece(tokenizer: unknown): WordPieceTokenizer {
  const t = tokenizer as TokenizerJson;
  const n = t?.normalizer, m = t?.model;
  if (n?.type !== 'BertNormalizer') throw new Error(`createWordPiece needs a BertNormalizer, got ${n?.type}`);
  if (t.pre_tokenizer?.type !== 'BertPreTokenizer') throw new Error(`createWordPiece needs a BertPreTokenizer, got ${t.pre_tokenizer?.type}`);
  if (m?.type !== 'WordPiece' || !m.vocab) throw new Error(`createWordPiece needs a WordPiece model with a vocabulary, got ${m?.type}`);
  const vocab = new Map(Object.entries(m.vocab));
  const unk = vocab.get(m.unk_token ?? '[UNK]');
  if (unk === undefined) throw new Error('the vocabulary has no unknown token');
  const prefix = m.continuing_subword_prefix ?? '##';
  const maxChars = m.max_input_chars_per_word ?? 100;
  const lowercase = n.lowercase !== false;
  const stripAccents = n.strip_accents ?? lowercase;

  const normalize = (text: string): string => {
    let out = '';
    for (const ch of text) {
      const cp = ch.codePointAt(0)!;
      if (n.clean_text !== false) {
        if (cp === 0 || cp === 0xfffd || isControl(ch)) continue;
        if (isWhitespace(ch)) { out += ' '; continue; }
      }
      out += n.handle_chinese_chars !== false && isChinese(cp) ? ` ${ch} ` : ch;
    }
    if (stripAccents) out = out.normalize('NFD').replace(/\p{Mn}/gu, '');
    if (lowercase) out = out.toLowerCase();
    return out;
  };
  const preTokenize = (text: string): string[] => {
    const words: string[] = [];
    let cur = '';
    for (const ch of text) {
      if (isWhitespace(ch)) { if (cur) words.push(cur); cur = ''; }
      else if (isPunctuation(ch)) { if (cur) words.push(cur); words.push(ch); cur = ''; }
      else cur += ch;
    }
    if (cur) words.push(cur);
    return words;
  };
  const wordPiece = (word: string, out: number[]) => {
    const chars = [...word];
    if (chars.length > maxChars) { out.push(unk); return; }
    const pieces: number[] = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length, found: number | undefined;
      while (start < end) {
        const sub = (start > 0 ? prefix : '') + chars.slice(start, end).join('');
        found = vocab.get(sub);
        if (found !== undefined) break;
        end--;
      }
      if (found === undefined) { out.push(unk); return; }
      pieces.push(found);
      start = end;
    }
    out.push(...pieces);
  };
  return {
    unkId: unk,
    vocabSize: vocab.size,
    encode(text: string) {
      const ids: number[] = [];
      for (const w of preTokenize(normalize(text))) wordPiece(w, ids);
      return ids;
    },
  };
}

/**
 * A static embedder: the mean of the token vectors in `weights` (row-major, vocabSize × dimensions),
 * unit length; unknown tokens dropped; at most maxTokens tokens (default 512). The identity records
 * digests of the exact bytes, so a release can require the same table.
 */
export async function createStaticEmbedder(options: { modelId: string; tokenizerBytes: Uint8Array; weights: Float32Array; dimensions: number; maxTokens?: number }): Promise<StaticEmbedder> {
  const { modelId, tokenizerBytes, weights, dimensions, maxTokens = 512 } = options;
  if (!Number.isInteger(dimensions) || dimensions < 1) throw new Error('dimensions must be a positive integer');
  if (!Number.isInteger(maxTokens) || maxTokens < 1) throw new Error('maxTokens must be a positive integer');
  if (weights.length % dimensions !== 0) throw new Error(`weights (${weights.length}) is not a multiple of dimensions (${dimensions})`);
  const json = JSON.parse(new TextDecoder().decode(tokenizerBytes));
  const tokenizer = createWordPiece(json);
  const rows = weights.length / dimensions;
  if (rows < tokenizer.vocabSize) throw new Error(`weights have ${rows} rows for a vocabulary of ${tokenizer.vocabSize}`);
  // Ids need not be contiguous: every id must index a row, or embed() would read past the table (NaN).
  let maxId = -1;
  for (const id of Object.values((json?.model?.vocab ?? {}) as Record<string, number>)) if (id > maxId) maxId = id;
  if (maxId >= rows) throw new Error(`the vocabulary's largest id is ${maxId}, but the weights have only ${rows} rows`);
  const identity: StaticIdentity = Object.freeze({
    schema: STATIC_SCHEMA, modelId, dimensions, maxTokens,
    tokenizerSha256: await sha256Hex(tokenizerBytes),
    weightsSha256: await sha256Hex(new Uint8Array(weights.buffer, weights.byteOffset, weights.byteLength)),
  });
  return {
    identity,
    embed(text: string) {
      const out = new Float32Array(dimensions);
      let count = 0;
      for (const id of tokenizer.encode(text)) {
        if (id === tokenizer.unkId) continue;
        if (count === maxTokens) break;
        const base = id * dimensions;
        for (let j = 0; j < dimensions; j++) out[j] += weights[base + j];
        count++;
      }
      let norm = 0;
      for (let j = 0; j < dimensions; j++) norm += out[j] * out[j];
      norm = Math.sqrt(norm);
      if (norm > 0) for (let j = 0; j < dimensions; j++) out[j] /= norm;
      return out;
    },
  };
}

/** Problems with a static identity (empty when valid). */
export function staticIdentityProblems(raw: unknown): string[] {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return ['static identity must be an object'];
  const s = raw as Record<string, unknown>;
  const p: string[] = [];
  const keys = ['schema', 'modelId', 'dimensions', 'maxTokens', 'tokenizerSha256', 'weightsSha256'];
  for (const k of Object.keys(s)) if (!keys.includes(k)) p.push(`static identity has unknown field ${k}`);
  if (s.schema !== STATIC_SCHEMA) p.push(`static identity schema must be '${STATIC_SCHEMA}'`);
  if (typeof s.modelId !== 'string' || !s.modelId) p.push('static identity modelId must be a non-empty string');
  for (const k of ['dimensions', 'maxTokens']) if (!Number.isInteger(s[k]) || (s[k] as number) < 1) p.push(`static identity ${k} must be a positive integer`);
  for (const k of ['tokenizerSha256', 'weightsSha256']) if (typeof s[k] !== 'string' || !/^[0-9a-f]{64}$/.test(s[k] as string)) p.push(`static identity ${k} must be lowercase hex SHA-256`);
  return p;
}
