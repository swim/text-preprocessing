/**
 * The tokenizer contract. An adapter counts the EXACT input its encoder will consume (prefixes, input
 * templates and special tokens included) and proposes legal cut offsets in the source text. Real model
 * tokenization lives in injected adapters; this is not permission to substitute an approximate counter.
 * A provider whose tokenizer is opaque cannot claim a hard budget by reporting 'provider-managed'.
 */
import { isCodePointBoundary } from './spans.ts';

export const TOKENIZER_SCHEMA = 'liquidau-tokenizer/1';

export interface TokenizerIdentity {
  schema: typeof TOKENIZER_SCHEMA;
  /** Tokenizer digest or immutable version. */
  revision: string;
  /** Versioned prefix/template/special-token behaviour, shared with the encoder's preprocessing. */
  inputPreparation: string;
}

export interface TokenizerAdapter {
  readonly identity: Readonly<TokenizerIdentity>;
  /** Tokens the encoder consumes for exactly this text, after its input preparation. Deterministic, no I/O. */
  countInput(text: string): number;
  /** Legal cut offsets into `text` (0 < offset < text.length, code-point boundaries), e.g. token starts. */
  cutOffsets(text: string): readonly number[];
}

/** Problems with a tokenizer identity (empty when valid). Unknown fields are refused. */
export function tokenizerIdentityProblems(raw: unknown): string[] {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return ['tokenizer identity must be an object'];
  const t = raw as Record<string, unknown>;
  const p: string[] = [];
  for (const k of Object.keys(t)) if (!['schema', 'revision', 'inputPreparation'].includes(k)) p.push(`tokenizer identity has unknown field ${k}`);
  if (t.schema !== TOKENIZER_SCHEMA) p.push(`tokenizer identity schema must be '${TOKENIZER_SCHEMA}'`);
  for (const k of ['revision', 'inputPreparation']) if (typeof t[k] !== 'string' || !t[k]) p.push(`tokenizer identity ${k} must be a non-empty string`);
  if (t.revision === 'provider-managed') p.push("tokenizer revision 'provider-managed' cannot count tokens exactly: document chunking needs an exact compatible tokenizer");
  return p;
}

/**
 * Contract checks for a tokenizer adapter on fixed probes: a valid identity, finite non-negative integer
 * counts, determinism and legal cut offsets. With `minEmptyCount`, the empty string must count at least
 * that many tokens (the encoder's special tokens; a tokenizer without them may legitimately count 0).
 */
export function tokenizerContractProblems(tokenizer: TokenizerAdapter, probes: readonly string[] = ['', 'a', '  leading and trailing  ', 'Unicode: café, 🙂, é, ＡＢＣ。', 'line\r\nbreaks\n\n\tand tabs', 'x'.repeat(500)], options: { minEmptyCount?: number } = {}): string[] {
  const problems = tokenizerIdentityProblems(tokenizer.identity);
  if (problems.length) return problems;
  if (options.minEmptyCount !== undefined) {
    let n: number | null = null;
    try { n = tokenizer.countInput(''); } catch { /* reported as null */ }
    if (n === null || n < options.minEmptyCount) problems.push(`the empty string counts ${n} tokens, expected at least ${options.minEmptyCount} (special tokens)`);
  }
  for (const text of probes) {
    const label = JSON.stringify(text.length > 30 ? `${text.slice(0, 30)}…` : text);
    let a: number, b: number;
    try {
      a = tokenizer.countInput(text);
      b = tokenizer.countInput(text);
    } catch (e) {
      problems.push(`countInput(${label}) threw: ${(e as Error).message}`);
      continue;
    }
    if (!Number.isInteger(a) || a < 0) problems.push(`countInput(${label}) returned ${a}, not a non-negative integer`);
    if (a !== b) problems.push(`countInput(${label}) is not deterministic (${a}, ${b})`);
    try {
      const cuts = tokenizer.cutOffsets(text);
      const again = tokenizer.cutOffsets(text);
      if (JSON.stringify(cuts) !== JSON.stringify(again)) problems.push(`cutOffsets(${label}) is not deterministic`);
      for (const o of cuts) if (!(Number.isInteger(o) && o > 0 && o < text.length && isCodePointBoundary(text, o))) problems.push(`cutOffsets(${label}) returned illegal offset ${o}`);
    } catch (e) {
      problems.push(`cutOffsets(${label}) threw: ${(e as Error).message}`);
    }
  }
  return problems;
}

/**
 * A deterministic FIXTURE tokenizer for tests and examples - not any real model's tokenizer. Tokens are
 * runs of letters/marks/digits, single other non-space characters, and whitespace runs (so counts are
 * not additive across joins); a configured prefix is prepared exactly once and `special` tokens are
 * added per input. Cut offsets are token starts.
 */
export function fixtureTokenizer(options: { revision?: string; prefix?: string; special?: number } = {}): TokenizerAdapter {
  const prefix = options.prefix ?? '';
  const special = options.special ?? 2;
  const TOKEN = /[\p{L}\p{M}\p{N}]+|\s+|[^\s\p{L}\p{M}\p{N}]/gu;
  const tokens = (s: string) => [...s.matchAll(TOKEN)];
  return {
    identity: Object.freeze({ schema: TOKENIZER_SCHEMA, revision: options.revision ?? 'fixture-tokenizer/1', inputPreparation: `fixture-prefix/1:${JSON.stringify(prefix)}+special:${special}` }),
    countInput: (text) => special + tokens(prefix + text).length,
    cutOffsets: (text) => tokens(text).map((m) => m.index).filter((o) => o > 0),
  };
}
