/**
 * The document pipeline: a validated, fully explicit configuration whose canonical digest is its
 * identity. Every output-affecting setting is a field; there are no defaults inside the digest.
 */
import { canonicalDigest } from './digest.ts';
import { budgetedEncoderProblems, fullEncoderIdentityProblems, identityFields, type FullEncoderIdentity } from './encoder.ts';
import { SPLITTER_ALGORITHMS, type SplitterAlgorithm } from './splitters.ts';
import { tokenizerIdentityProblems, type TokenizerIdentity } from './tokenizer.ts';

export const PIPELINE_SCHEMA = 'liquidau-document-pipeline/2';

export interface DocumentPipeline {
  schema: typeof PIPELINE_SCHEMA;
  splitter: { algorithm: SplitterAlgorithm };
  /**
   * 'token-pack/1': pack atoms under the budget. 'adjacent-cosine/1' (experimental): embed each atom with
   * the boundary encoder (atoms over its budget are hard-split first), start a new group where adjacent
   * cosine similarity is strictly below `threshold`, then pack within groups under the budget.
   */
  grouping: TokenPackGrouping | AdjacentCosineGrouping;
  /**
   * How an oversized atom's cut is chosen among the tokenizer's legal offsets. 'bracketed/1': recount the
   * 1st, 2nd, 4th, 8th, ... candidate prefix until one doesn't fit, then bisect - the largest fitting
   * prefix when counts never decrease as a prefix grows, O(log n) recounts per cut.
   */
  cutSearch: 'bracketed/1';
  tokenizer: TokenizerIdentity;
  /** Per chunk, including the encoder's special tokens and prefixes. */
  maxInputTokens: number;
  maxChunks: number;
  /** Tokenizer calls (counts and cut proposals) allowed while planning one document. */
  maxPlanningSteps: number;
  overlap: 0;
  aggregation: { algorithm: 'mean-chunk-vector/1'; normalization: 'none' };
  emptyInput: 'reject';
  overflow: 'reject';
}

export interface TokenPackGrouping {
  algorithm: 'token-pack/1';
}

export interface AdjacentCosineGrouping {
  algorithm: 'adjacent-cosine/1';
  /** Finite, in [-1, 1]. A new group starts where similarity < threshold; equality does not split. */
  threshold: number;
  /** A separate identity, unless it is exactly the classification encoder. */
  boundaryEncoderIdentity: FullEncoderIdentity;
  boundaryTokenizer: TokenizerIdentity;
  /** Oversized atoms are cut with legal offsets under the boundary budget; atoms are never merged. */
  boundaryPreparation: 'hard-split-atoms/1';
  boundaryMaxInputTokens: number;
}

export class PipelineError extends Error {
  override readonly name = 'PipelineError';
  readonly problems: readonly string[];
  constructor(problems: readonly string[]) {
    super(`invalid document pipeline: ${problems.join('; ')}`);
    this.problems = Object.freeze([...problems]);
  }
}

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const exactKeys = (o: Record<string, unknown>, keys: string[], what: string, p: string[]) => { for (const k of Object.keys(o)) if (!keys.includes(k)) p.push(`${what} has unknown field ${k}`); };
const positiveInt = (v: unknown, max: number) => Number.isInteger(v) && (v as number) >= 1 && (v as number) <= max;

/** Problems with a pipeline (empty when valid). */
export function pipelineProblems(raw: unknown): string[] {
  if (!isObject(raw)) return ['the pipeline must be an object'];
  if (raw.schema !== PIPELINE_SCHEMA) return [`unsupported pipeline schema ${JSON.stringify(raw.schema)} (this library reads '${PIPELINE_SCHEMA}')`];
  const p: string[] = [];
  exactKeys(raw, ['schema', 'splitter', 'grouping', 'cutSearch', 'tokenizer', 'maxInputTokens', 'maxChunks', 'maxPlanningSteps', 'overlap', 'aggregation', 'emptyInput', 'overflow'], 'pipeline', p);
  const sp = raw.splitter;
  if (!isObject(sp) || !(SPLITTER_ALGORITHMS as readonly unknown[]).includes(sp.algorithm)) p.push(`splitter.algorithm must be one of ${SPLITTER_ALGORITHMS.join(', ')}`);
  else exactKeys(sp, ['algorithm'], 'splitter', p);
  const g = raw.grouping;
  if (isObject(g) && g.algorithm === 'adjacent-cosine/1') {
    exactKeys(g, ['algorithm', 'threshold', 'boundaryEncoderIdentity', 'boundaryTokenizer', 'boundaryPreparation', 'boundaryMaxInputTokens'], 'grouping', p);
    if (!(typeof g.threshold === 'number' && Number.isFinite(g.threshold) && g.threshold >= -1 && g.threshold <= 1)) p.push('grouping.threshold must be a finite number in [-1, 1]');
    const idp = fullEncoderIdentityProblems(g.boundaryEncoderIdentity);
    p.push(...idp.map((e) => `grouping.boundaryEncoderIdentity ${e}`));
    const tp = tokenizerIdentityProblems(g.boundaryTokenizer);
    p.push(...tp.map((e) => `grouping.boundaryTokenizer: ${e}`));
    if (g.boundaryPreparation !== 'hard-split-atoms/1') p.push("grouping.boundaryPreparation must be 'hard-split-atoms/1'");
    if (!positiveInt(g.boundaryMaxInputTokens, 1 << 20)) p.push('grouping.boundaryMaxInputTokens must be a positive integer');
    else if (!idp.length && !tp.length) {
      p.push(...budgetedEncoderProblems(g.boundaryEncoderIdentity as FullEncoderIdentity, { maxInputTokens: g.boundaryMaxInputTokens as number, tokenizerRevision: (g.boundaryTokenizer as TokenizerIdentity).revision }, 'the boundary encoder'));
    }
  } else if (!isObject(g) || g.algorithm !== 'token-pack/1') p.push("grouping.algorithm must be 'token-pack/1' or 'adjacent-cosine/1'");
  else exactKeys(g, ['algorithm'], 'grouping', p);
  if (raw.cutSearch !== 'bracketed/1') p.push("cutSearch must be 'bracketed/1'");
  p.push(...tokenizerIdentityProblems(raw.tokenizer));
  if (!positiveInt(raw.maxInputTokens, 1 << 20)) p.push('maxInputTokens must be a positive integer');
  if (!positiveInt(raw.maxChunks, 1 << 16)) p.push('maxChunks must be an integer in [1, 65536]');
  if (!positiveInt(raw.maxPlanningSteps, 1 << 24)) p.push('maxPlanningSteps must be a positive integer');
  if (raw.overlap !== 0) p.push('overlap must be 0');
  const a = raw.aggregation;
  if (!isObject(a) || a.algorithm !== 'mean-chunk-vector/1' || a.normalization !== 'none') p.push("aggregation must be { algorithm: 'mean-chunk-vector/1', normalization: 'none' }");
  else exactKeys(a, ['algorithm', 'normalization'], 'aggregation', p);
  if (raw.emptyInput !== 'reject') p.push("emptyInput must be 'reject'");
  if (raw.overflow !== 'reject') p.push("overflow must be 'reject'");
  return p;
}

/** Throws PipelineError with every problem. */
export function validatePipeline(raw: unknown): DocumentPipeline {
  const p = pipelineProblems(raw);
  if (p.length) throw new PipelineError(p);
  return raw as DocumentPipeline;
}

/** A pipeline with its fixed fields and default limits made explicit, validated. Identities keep only their defined fields. */
export function createPipeline(options: {
  splitter: SplitterAlgorithm;
  tokenizer: TokenizerIdentity;
  maxInputTokens: number;
  maxChunks?: number;
  maxPlanningSteps?: number;
  /** Experimental semantic grouping; omit for 'token-pack/1'. */
  semantic?: { threshold: number; boundaryEncoderIdentity: FullEncoderIdentity; boundaryTokenizer: TokenizerIdentity; boundaryMaxInputTokens: number };
}): DocumentPipeline {
  const sem = options.semantic;
  return validatePipeline({
    schema: PIPELINE_SCHEMA,
    splitter: { algorithm: options.splitter },
    grouping: sem
      ? {
        algorithm: 'adjacent-cosine/1', threshold: sem.threshold, boundaryEncoderIdentity: { ...identityFields(sem.boundaryEncoderIdentity), layers: sem.boundaryEncoderIdentity.layers ? [...sem.boundaryEncoderIdentity.layers] : null },
        boundaryTokenizer: { schema: sem.boundaryTokenizer.schema, revision: sem.boundaryTokenizer.revision, inputPreparation: sem.boundaryTokenizer.inputPreparation },
        boundaryPreparation: 'hard-split-atoms/1', boundaryMaxInputTokens: sem.boundaryMaxInputTokens,
      }
      : { algorithm: 'token-pack/1' },
    cutSearch: 'bracketed/1',
    tokenizer: { schema: options.tokenizer.schema, revision: options.tokenizer.revision, inputPreparation: options.tokenizer.inputPreparation },
    maxInputTokens: options.maxInputTokens,
    maxChunks: options.maxChunks ?? 64,
    maxPlanningSteps: options.maxPlanningSteps ?? 20_000,
    overlap: 0,
    aggregation: { algorithm: 'mean-chunk-vector/1', normalization: 'none' },
    emptyInput: 'reject',
    overflow: 'reject',
  });
}

/** The pipeline's identity: SHA-256 of its canonical JSON (validated first). */
export const pipelineDigest = (pipeline: DocumentPipeline): string => canonicalDigest(validatePipeline(pipeline));
