/**
 * The encoder identity and vector checks, as plain data and structural callbacks (no router types).
 * The same field definitions as @liquidau/router's EncoderIdentity; conformance fixtures in the router
 * keep the two in step. Used for the semantic-grouping boundary encoder here, and by
 * embedding-classifier for the chunk encoder.
 */

export const ENCODER_IDENTITY_SCHEMA = 'liquidau-encoder/1';

export interface FullEncoderIdentity {
  schema: typeof ENCODER_IDENTITY_SCHEMA;
  modelId: string;
  /** Immutable provider version or local model-content digest. */
  revision: string;
  dimensions: number;
  precision: string;
  inputType: string | null;
  /** Ordered 1-based layers, or null for the model's standard embedding. */
  layers: number[] | null;
  pooling: string;
  normalization: 'none' | 'unit' | 'per-layer-unit';
  tokenizerRevision: string;
  maxChars: number | null;
  maxTokens: number | null;
  truncation: string;
}

/** Structurally compatible with @liquidau/router's Encoder. */
export interface EncoderAdapter {
  readonly identity: Readonly<FullEncoderIdentity>;
  embed(texts: readonly string[], options: { signal: AbortSignal }): Promise<readonly (readonly number[])[]>;
}

export const ENCODER_IDENTITY_FIELDS = ['schema', 'modelId', 'revision', 'dimensions', 'precision', 'inputType', 'layers', 'pooling', 'normalization', 'tokenizerRevision', 'maxChars', 'maxTokens', 'truncation'] as const;

const nonEmpty = (v: unknown) => typeof v === 'string' && v.length > 0;
const positiveInt = (v: unknown) => Number.isInteger(v) && (v as number) > 0;

/** Problems with a full encoder identity (empty when valid). Unknown fields are refused. */
export function fullEncoderIdentityProblems(raw: unknown): string[] {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return ['encoder identity must be an object'];
  const id = raw as Record<string, unknown>;
  const p: string[] = [];
  if (id.schema !== ENCODER_IDENTITY_SCHEMA) p.push(`schema must be '${ENCODER_IDENTITY_SCHEMA}'`);
  for (const k of Object.keys(id)) if (!(ENCODER_IDENTITY_FIELDS as readonly string[]).includes(k)) p.push(`unknown field ${k}`);
  for (const k of ['modelId', 'revision', 'precision', 'pooling', 'tokenizerRevision', 'truncation'] as const) if (!nonEmpty(id[k])) p.push(`${k} must be a non-empty string`);
  if (!positiveInt(id.dimensions)) p.push('dimensions must be a positive integer');
  if (id.inputType !== null && !nonEmpty(id.inputType)) p.push('inputType must be a non-empty string or null');
  for (const k of ['maxChars', 'maxTokens'] as const) if (id[k] !== null && !positiveInt(id[k])) p.push(`${k} must be a positive integer or null`);
  if (!['none', 'unit', 'per-layer-unit'].includes(id.normalization as string)) p.push('normalization must be one of none, unit, per-layer-unit');
  const layers = id.layers;
  if (layers !== null) {
    if (!Array.isArray(layers) || !layers.length || !layers.every(positiveInt) || new Set(layers).size !== layers.length) p.push('layers must be null or distinct positive integers');
    else if (positiveInt(id.dimensions) && (id.dimensions as number) % layers.length !== 0) p.push(`dimensions ${id.dimensions} don't split evenly over ${layers.length} layers`);
  }
  if (id.normalization === 'per-layer-unit' && !Array.isArray(layers)) p.push("normalization 'per-layer-unit' needs layers");
  return p;
}

/** The identity's fields in a fixed order (drops anything else), for canonical comparison and hashing. */
export const identityFields = (id: FullEncoderIdentity): Record<string, unknown> => Object.fromEntries(ENCODER_IDENTITY_FIELDS.map((f) => [f, id[f]]));

/**
 * Why this encoder can't consume token-budgeted inputs exactly (empty when it can): no silent truncation,
 * a known token limit at or above `maxInputTokens`, and an exact tokenizer whose revision is `tokenizerRevision`.
 */
export function budgetedEncoderProblems(identity: FullEncoderIdentity, budget: { maxInputTokens: number; tokenizerRevision: string }, what = 'the encoder'): string[] {
  const p: string[] = [];
  if (identity.truncation !== 'none') p.push(`${what} truncates (${identity.truncation}); budgeted inputs need truncation 'none' so every planned token is consumed`);
  if (identity.maxTokens === null) p.push(`${what}'s maxTokens is unknown, so the token budget can't be checked against it`);
  else if (budget.maxInputTokens > identity.maxTokens) p.push(`the budget of ${budget.maxInputTokens} input tokens exceeds ${what}'s ${identity.maxTokens}`);
  if (identity.tokenizerRevision === 'provider-managed') p.push(`a 'provider-managed' tokenizer cannot count tokens exactly: chunking is unsupported for ${what}`);
  if (identity.tokenizerRevision !== budget.tokenizerRevision) p.push(`the tokenizer revision ${budget.tokenizerRevision} is not ${what}'s ${identity.tokenizerRevision}`);
  return p;
}

/** Problems with one output vector for an identity: width, finiteness and the declared normalisation (null when fine). */
export function vectorProblems(identity: FullEncoderIdentity, vector: unknown, tolerance = 1e-3): string | null {
  if (!Array.isArray(vector) && !(ArrayBuffer.isView(vector) && !(vector instanceof DataView))) return 'not an array';
  const v = vector as ArrayLike<number>;
  if (v.length !== identity.dimensions) return `width ${v.length}, expected ${identity.dimensions}`;
  for (let i = 0; i < v.length; i++) if (typeof v[i] !== 'number' || !Number.isFinite(v[i])) return `non-finite value at index ${i}`;
  const norm = (a: number, b: number) => { let s = 0; for (let i = a; i < b; i++) s += v[i] * v[i]; return Math.sqrt(s); };
  if (identity.normalization === 'unit' && Math.abs(norm(0, v.length) - 1) > tolerance) return `norm ${norm(0, v.length)}, expected unit length`;
  if (identity.normalization === 'per-layer-unit' && identity.layers) {
    const w = v.length / identity.layers.length;
    for (let l = 0; l < identity.layers.length; l++) if (Math.abs(norm(l * w, (l + 1) * w) - 1) > tolerance) return `layer block ${l} is not unit length`;
  }
  return null;
}
