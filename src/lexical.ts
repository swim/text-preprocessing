/**
 * Hashed lexical features for edge scorers ('liquidau-lexical/1'): what a lexical model sees, defined
 * once so training and the edge compute identical features. Text is NFKC-normalised and lowercased;
 * words are runs of letters, digits and apostrophes. Features: word unigrams and bigrams, and character
 * 3-5-grams of each word wrapped as ^word$, hashed with 32-bit FNV-1a into 2^bits buckets (collisions
 * add). Values are √count, scaled to unit length. Deterministic, no dependencies, no I/O.
 */
export const LEXICAL_SCHEMA = 'liquidau-lexical/1';

export interface LexicalSpec {
  schema: typeof LEXICAL_SCHEMA;
  /** Hash buckets: 2^bits (default 18). */
  bits: number;
}

export const DEFAULT_LEXICAL: Readonly<LexicalSpec> = Object.freeze({ schema: LEXICAL_SCHEMA, bits: 18 });

/** Problems with a lexical spec (empty when valid). */
export function lexicalSpecProblems(raw: unknown): string[] {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return ['lexical spec must be an object'];
  const s = raw as Record<string, unknown>;
  const p: string[] = [];
  for (const k of Object.keys(s)) if (k !== 'schema' && k !== 'bits') p.push(`lexical spec has unknown field ${k}`);
  if (s.schema !== LEXICAL_SCHEMA) p.push(`lexical spec schema must be '${LEXICAL_SCHEMA}'`);
  if (!Number.isInteger(s.bits) || (s.bits as number) < 10 || (s.bits as number) > 24) p.push('lexical spec bits must be an integer in [10, 24]');
  return p;
}

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** The words lexical features are built from. */
export function lexicalWords(text: string): string[] {
  return text.normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? [];
}

/**
 * Sparse features of a text: parallel arrays of bucket indices (ascending, unique) and values, unit
 * length. Empty text gives empty arrays.
 */
export function lexicalFeatures(text: string, spec: LexicalSpec = DEFAULT_LEXICAL): { index: Uint32Array; value: Float32Array } {
  if (!(Number.isInteger(spec.bits) && spec.bits >= 10 && spec.bits <= 24)) throw new Error(`lexical spec bits must be an integer in [10, 24], got ${spec.bits}`);
  const mask = (1 << spec.bits) - 1;
  const counts = new Map<number, number>();
  const add = (key: string) => {
    const h = fnv1a(key) & mask;
    counts.set(h, (counts.get(h) ?? 0) + 1);
  };
  const ws = lexicalWords(text);
  ws.forEach((w, i) => {
    add(`w:${w}`);
    if (i + 1 < ws.length) add(`b:${w} ${ws[i + 1]}`);
    const p = `^${w}$`;
    for (let n = 3; n <= 5; n++) for (let j = 0; j + n <= p.length; j++) add(`c:${p.slice(j, j + n)}`);
  });
  const keys = [...counts.keys()].sort((a, b) => a - b);
  const index = Uint32Array.from(keys);
  const value = new Float32Array(keys.length);
  let norm = 0;
  keys.forEach((k, i) => {
    const v = Math.sqrt(counts.get(k)!);
    value[i] = v;
    norm += v * v;
  });
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < value.length; i++) value[i] /= norm;
  return { index, value };
}
