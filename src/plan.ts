/**
 * Deterministic chunk planning.
 *
 * 'token-pack/1':
 *   1. Split into atomic spans with the pipeline's splitter.
 *   2. Walk atoms in order: try appending the next atom to the current core and count the COMPLETE
 *      prospective input (token counts are not additive, so atom counts are never summed). Append if
 *      it fits; otherwise flush the core ('structure') and retry the atom on an empty core.
 *   3. An atom that alone doesn't fit is cut at one of the tokenizer's legal offsets, chosen by the
 *      pipeline's cut search ('bracketed/1'): candidates ascending; recount the 1st, 2nd, 4th, 8th, ...
 *      prefix until one doesn't fit (or the last fits), then bisect between the last fitting and the
 *      first non-fitting candidate. When counts never decrease as a prefix grows, that is the largest
 *      fitting prefix; for any tokenizer it is deterministic and the chosen prefix was counted and fits.
 *      It becomes a chunk ('budget') and the remainder is retried. The first candidate doesn't fit:
 *      TOKEN_BUDGET_UNSATISFIABLE.
 *   4. Recount every chunk; each must fit. Cores partition the text exactly; context = core (no overlap).
 *
 * 'adjacent-cosine/1' (experimental) adds, before packing:
 *   a. hard-split-atoms/1: every atom over the BOUNDARY budget is cut by the same legal-cut search under
 *      the boundary tokenizer (never merged); the subatoms partition the atom and keep its index.
 *   b. one validated boundary embedding per subatom (batched; results placed by index, so batch size and
 *      completion order never change anything), then adjacent cosine similarities by direct dot
 *      products - O(n·d), no similarity matrix. Zero norms or non-finite similarities fail.
 *   c. a new group starts where similarity is strictly below the threshold.
 *   d. packing as above over the subatoms, never across a group boundary (those chunks end 'semantic').
 *
 * Bounded: every tokenizer call and every boundary subatom is a planning step (maxPlanningSteps, checked
 * BEFORE boundary requests are issued), chunks are capped (maxChunks), input bytes are capped by the host.
 * Exhaustion fails the whole document - a suffix is never dropped. Planning yields to the event loop every
 * `yieldEvery` steps and checks the abort signal between steps; a timer cannot preempt one slow
 * synchronous tokenizer call.
 */
import { canonicalDigest, sourceDigest } from './digest.ts';
import { identityFields, vectorProblems, type EncoderAdapter } from './encoder.ts';
import { pipelineDigest, validatePipeline, type AdjacentCosineGrouping, type DocumentPipeline } from './pipeline.ts';
import { isCodePointBoundary, partitionProblems, startsWithMark, type SourceSpan } from './spans.ts';
import { splitSpans } from './splitters.ts';
import { tokenizerIdentityProblems, type TokenizerAdapter } from './tokenizer.ts';

export const PREPROCESSING_ERROR_CODES = [
  'INVALID_PIPELINE', 'TOKENIZER_MISMATCH', 'EMPTY_INPUT', 'INPUT_TOO_LARGE', 'TOKEN_BUDGET_UNSATISFIABLE',
  'MAX_CHUNKS_EXCEEDED', 'MAX_PLANNING_STEPS_EXCEEDED', 'TOKENIZER_FAILURE', 'ABORTED',
  'BOUNDARY_CAPABILITY_MISSING', 'BOUNDARY_ENCODER_FAILURE', 'INVALID_BOUNDARY_EMBEDDING', 'DEADLINE_EXCEEDED',
] as const;
export type PreprocessingErrorCode = (typeof PREPROCESSING_ERROR_CODES)[number];

export class PreprocessingError extends Error {
  override readonly name = 'PreprocessingError';
  readonly code: PreprocessingErrorCode;
  /** BOUNDARY_ENCODER_FAILURE: the encoder error's own `retryable`, when it states one. */
  readonly retryable: boolean | undefined;
  constructor(code: PreprocessingErrorCode, message: string, options: { retryable?: boolean; cause?: unknown } = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.code = code;
    this.retryable = options.retryable;
  }
}

export interface ChunkPlanEntry {
  index: number;
  core: SourceSpan;
  /** Equals core (the pipeline's overlap is 0). */
  context: SourceSpan;
  /** Atomic spans contributing (a hard-cut atom contributes to several chunks). */
  atomIndices: readonly number[];
  /** Exact encoder input tokens for the context, special tokens and prefixes included. */
  inputTokens: number;
  /** Why the chunk ended. */
  boundary: 'document-end' | 'structure' | 'budget' | 'semantic';
}

export interface ChunkPlan {
  pipelineDigest: string;
  sourceDigest: string;
  /** The encoder's source-character limit the plan honours (part of its identity), or null. */
  contextCharLimit: number | null;
  chunks: readonly ChunkPlanEntry[];
  /** e.g. a forced cut that separates a combining mark from its base. */
  diagnostics: readonly string[];
}

export interface PlanOptions {
  signal?: AbortSignal;
  /** The encoder identity's maxChars: every context must also be at most this many UTF-16 units. */
  contextCharLimit?: number | null;
  /** Host limit on the document's UTF-8 size (lone surrogates count 3 bytes, as TextEncoder encodes them). */
  maxInputUtf8Bytes?: number;
  /** Yield to the event loop every this many tokenizer calls (default 256). */
  yieldEvery?: number;
  /**
   * An absolute deadline on `performance.now()`'s clock (Date.now() where unavailable), checked at EVERY
   * planning step and around boundary requests - so a deadline is honoured even between event-loop
   * yields, when a timer-driven abort signal can't fire - and timed while a boundary request is pending,
   * so a request that never returns can't outlive it. Overrunning it fails DEADLINE_EXCEEDED.
   */
  deadline?: number;
  /** 'adjacent-cosine/1' only: adapters for the pipeline's boundary encoder and tokenizer identities. */
  boundary?: { encoder: EncoderAdapter; tokenizer: TokenizerAdapter; batchSize?: number; concurrency?: number };
}

function utf8Length(text: string): number {
  let n = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 0x80) n += 1;
    else if (c < 0x800) n += 2;
    else if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length && text.charCodeAt(i + 1) >= 0xdc00 && text.charCodeAt(i + 1) <= 0xdfff) { n += 4; i++; }
    else n += 3;
  }
  return n;
}

const yieldToEventLoop = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
const clock = () => (globalThis as { performance?: { now(): number } }).performance?.now() ?? Date.now();

interface Unit { start: number; end: number; atom: number; groupStart: boolean }

/** Shared planning state: the step budget, cancellation and checked tokenizer calls. */
class Planner {
  steps = 0;
  readonly diagnostics: string[] = [];
  readonly text: string;
  readonly maxSteps: number;
  readonly signal: AbortSignal | undefined;
  readonly yieldEvery: number;
  readonly deadline: number | undefined;
  constructor(text: string, maxSteps: number, signal: AbortSignal | undefined, yieldEvery: number, deadline: number | undefined) {
    this.text = text;
    this.maxSteps = maxSteps;
    this.signal = signal;
    this.yieldEvery = yieldEvery;
    this.deadline = deadline;
  }

  checkAbort() {
    if (this.signal?.aborted) throw new PreprocessingError('ABORTED', 'planning was aborted');
    if (this.deadline !== undefined && clock() > this.deadline) throw new PreprocessingError('DEADLINE_EXCEEDED', 'planning overran its deadline');
  }

  async step(n = 1) {
    this.checkAbort();
    const before = this.steps;
    this.steps += n;
    if (this.steps > this.maxSteps) throw new PreprocessingError('MAX_PLANNING_STEPS_EXCEEDED', `planning needed more than ${this.maxSteps} steps`);
    if (Math.floor(before / this.yieldEvery) !== Math.floor(this.steps / this.yieldEvery)) {
      await yieldToEventLoop();
      this.checkAbort();
    }
  }

  count(tokenizer: TokenizerAdapter, s: string): number {
    let n: number;
    try {
      n = tokenizer.countInput(s);
    } catch (e) {
      throw new PreprocessingError('TOKENIZER_FAILURE', `countInput threw: ${(e as Error).message}`, { cause: e });
    }
    // A slow synchronous call may have overrun the deadline: check right after it, before trusting it.
    this.checkAbort();
    if (!Number.isInteger(n) || n < 0) throw new PreprocessingError('TOKENIZER_FAILURE', `countInput returned ${n}, not a non-negative integer`);
    return n;
  }

  /** Token count when [start, end) fits `max` tokens and `limit` characters under `tokenizer`, else null. */
  async fits(tokenizer: TokenizerAdapter, start: number, end: number, max: number, limit: number | null): Promise<number | null> {
    if (limit !== null && end - start > limit) return null;
    await this.step();
    const n = this.count(tokenizer, this.text.slice(start, end));
    return n <= max ? n : null;
  }

  /**
   * A legal prefix of [start, end) that fits, by the 'bracketed/1' cut search: candidates ascending; test
   * the 1st, 2nd, 4th, 8th, ... until one doesn't fit (or the last fits), then bisect between the last
   * fitting and the first non-fitting candidate. O(log n) recounts of prefixes near the budget.
   */
  async largestPrefix(tokenizer: TokenizerAdapter, start: number, end: number, max: number, limit: number | null, what: string): Promise<{ at: number; tokens: number }> {
    const piece = this.text.slice(start, end);
    await this.step();
    let offsets: readonly number[];
    try {
      offsets = tokenizer.cutOffsets(piece);
    } catch (e) {
      throw new PreprocessingError('TOKENIZER_FAILURE', `cutOffsets threw: ${(e as Error).message}`, { cause: e });
    }
    if (!Array.isArray(offsets)) throw new PreprocessingError('TOKENIZER_FAILURE', 'cutOffsets did not return an array');
    for (const o of offsets) if (!(Number.isInteger(o) && o > 0 && o < piece.length && isCodePointBoundary(piece, o))) throw new PreprocessingError('TOKENIZER_FAILURE', `cutOffsets returned illegal offset ${o}`);
    // Prefixes over the character limit can never fit; dropping them costs no tokenizer call.
    const candidates = [...new Set(offsets)].filter((o) => limit === null || o <= limit).sort((a, b) => a - b);
    const last = candidates.length - 1;
    let fit = -1, fitTokens = 0, over = candidates.length;
    for (let i = 0; i <= last; i = Math.min(last, 2 * i + 1)) {
      const t = await this.fits(tokenizer, start, start + candidates[i], max, limit);
      if (t === null) { over = i; break; }
      fit = i; fitTokens = t;
      if (i === last) break;
    }
    while (fit >= 0 && over - fit > 1) {
      const mid = (fit + over) >>> 1;
      const t = await this.fits(tokenizer, start, start + candidates[mid], max, limit);
      if (t === null) over = mid;
      else { fit = mid; fitTokens = t; }
    }
    if (fit < 0) throw new PreprocessingError('TOKEN_BUDGET_UNSATISFIABLE', `no legal nonempty prefix of ${what} fits ${max} input tokens${limit !== null ? ` and ${limit} characters` : ''}`);
    const at = start + candidates[fit];
    if (startsWithMark(this.text, at)) this.diagnostics.push(`a forced cut at ${at} separates a combining mark from its base`);
    return { at, tokens: fitTokens };
  }
}

/** Packs units in order under the classification budget; never across a unit marked groupStart. */
async function pack(pl: Planner, units: Unit[], tokenizer: TokenizerAdapter, max: number, limit: number | null, maxChunks: number): Promise<ChunkPlanEntry[]> {
  const chunks: ChunkPlanEntry[] = [];
  const flush = (core: SourceSpan, atomIndices: number[], inputTokens: number, boundary: ChunkPlanEntry['boundary']) => {
    if (chunks.length >= maxChunks) throw new PreprocessingError('MAX_CHUNKS_EXCEEDED', `the document needs more than ${maxChunks} chunks`);
    chunks.push({ index: chunks.length, core: { ...core }, context: { ...core }, atomIndices: [...atomIndices], inputTokens, boundary });
  };
  const queue = units.map((u) => ({ ...u }));
  let current: { start: number; end: number; atoms: number[]; tokens: number } | null = null;
  let i = 0;
  while (i < queue.length) {
    const u = queue[i];
    if (current && u.groupStart) {
      flush(current, current.atoms, current.tokens, 'semantic');
      current = null;
    }
    if (current) {
      const n = await pl.fits(tokenizer, current.start, u.end, max, limit);
      if (n !== null) {
        current.end = u.end;
        if (current.atoms[current.atoms.length - 1] !== u.atom) current.atoms.push(u.atom);
        current.tokens = n;
        i++;
      } else {
        flush(current, current.atoms, current.tokens, 'structure');
        current = null;
      }
      continue;
    }
    const n = await pl.fits(tokenizer, u.start, u.end, max, limit);
    if (n !== null) {
      current = { start: u.start, end: u.end, atoms: [u.atom], tokens: n };
      i++;
      continue;
    }
    // The unit alone is too large: its largest fitting legal prefix becomes its own chunk.
    const cut = await pl.largestPrefix(tokenizer, u.start, u.end, max, limit, `atom ${u.atom}`);
    flush({ start: u.start, end: cut.at }, [u.atom], cut.tokens, 'budget');
    queue[i] = { start: cut.at, end: u.end, atom: u.atom, groupStart: false };
  }
  if (current) flush(current, current.atoms, current.tokens, 'document-end');
  else if (chunks.length) chunks[chunks.length - 1].boundary = 'document-end';
  return chunks;
}

/** hard-split-atoms/1: every atom cut until each piece fits the boundary budget (pieces never merge). */
async function boundaryUnits(pl: Planner, atoms: SourceSpan[], g: AdjacentCosineGrouping, tokenizer: TokenizerAdapter): Promise<Unit[]> {
  const limit = g.boundaryEncoderIdentity.maxChars;
  const out: Unit[] = [];
  for (let a = 0; a < atoms.length; a++) {
    let start = atoms[a].start;
    const end = atoms[a].end;
    while (start < end) {
      if (await pl.fits(tokenizer, start, end, g.boundaryMaxInputTokens, limit) !== null) { out.push({ start, end, atom: a, groupStart: false }); break; }
      const cut = await pl.largestPrefix(tokenizer, start, end, g.boundaryMaxInputTokens, limit, `atom ${a} (boundary preparation)`);
      out.push({ start, end: cut.at, atom: a, groupStart: false });
      start = cut.at;
    }
  }
  return out;
}

/** Boundary vectors in unit order (batched, bounded concurrency, results placed by index). */
async function boundaryVectors(pl: Planner, units: Unit[], opts: NonNullable<PlanOptions['boundary']>): Promise<number[][]> {
  const batchSize = opts.batchSize ?? 16, concurrency = opts.concurrency ?? 2;
  if (!(Number.isInteger(batchSize) && batchSize >= 1 && batchSize <= 256) || !(Number.isInteger(concurrency) && concurrency >= 1 && concurrency <= 16)) {
    throw new PreprocessingError('INVALID_PIPELINE', 'boundary batchSize must be in [1, 256] and concurrency in [1, 16]');
  }
  // Every subatom is a step, charged before any request is issued.
  await pl.step(units.length);
  const identity = opts.encoder.identity;
  const controller = new AbortController();
  let rejectStop!: (e: unknown) => void;
  const stopped = new Promise<never>((_, reject) => { rejectStop = reject; });
  stopped.catch(() => {});
  const onAbort = () => { controller.abort(); rejectStop(new PreprocessingError('ABORTED', 'planning was aborted')); };
  pl.checkAbort();
  pl.signal?.addEventListener('abort', onAbort, { once: true });
  // The deadline also bounds a request that never returns: it stops the wait and aborts the encoder.
  const deadlineTimer = pl.deadline === undefined ? undefined : setTimeout(() => {
    controller.abort();
    rejectStop(new PreprocessingError('DEADLINE_EXCEEDED', 'planning overran its deadline'));
  }, Math.max(0, pl.deadline - clock()));
  const vectors = new Array<number[]>(units.length);
  const starts = Array.from({ length: Math.ceil(units.length / batchSize) }, (_, b) => b * batchSize);
  let next = 0;
  const worker = async () => {
    while (next < starts.length && !controller.signal.aborted) {
      const s0 = starts[next++];
      const batch = units.slice(s0, s0 + batchSize).map((u) => pl.text.slice(u.start, u.end));
      let rows: unknown;
      try {
        pl.checkAbort();
        const call = Promise.resolve().then(() => opts.encoder.embed(batch, { signal: controller.signal }));
        call.catch(() => {});
        rows = await Promise.race([call, stopped]);
        pl.checkAbort();
      } catch (e) {
        if (e instanceof PreprocessingError) throw e;
        const retryable = typeof (e as { retryable?: unknown })?.retryable === 'boolean' ? (e as { retryable: boolean }).retryable : undefined;
        throw new PreprocessingError('BOUNDARY_ENCODER_FAILURE', `the boundary encoder failed: ${(e as Error)?.message ?? String(e)}`, { retryable, cause: e });
      }
      if (!Array.isArray(rows) || rows.length !== batch.length) throw new PreprocessingError('INVALID_BOUNDARY_EMBEDDING', `the boundary encoder returned ${Array.isArray(rows) ? rows.length : typeof rows} rows for ${batch.length} atoms`);
      rows.forEach((r, k) => {
        const why = vectorProblems(identity, r);
        if (why) throw new PreprocessingError('INVALID_BOUNDARY_EMBEDDING', `boundary atom ${s0 + k}: ${why}`);
        vectors[s0 + k] = Array.from(r as ArrayLike<number>);
      });
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(concurrency, starts.length) }, worker));
  } catch (e) {
    controller.abort();
    throw e;
  } finally {
    pl.signal?.removeEventListener('abort', onAbort);
    if (deadlineTimer !== undefined) clearTimeout(deadlineTimer);
  }
  pl.checkAbort();
  return vectors;
}

/** Adjacent cosine similarities: sims[i] between units i-1 and i (sims[0] unused). O(n·d). */
export function adjacentSimilarities(vectors: readonly (readonly number[])[]): number[] {
  const norms = vectors.map((v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)));
  norms.forEach((n, i) => { if (!(n > 0) || !Number.isFinite(n)) throw new PreprocessingError('INVALID_BOUNDARY_EMBEDDING', `boundary atom ${i} has a zero or non-finite norm`); });
  const sims = [Number.NaN];
  for (let i = 1; i < vectors.length; i++) {
    let dot = 0;
    for (let j = 0; j < vectors[i].length; j++) dot += vectors[i - 1][j] * vectors[i][j];
    const s = dot / (norms[i - 1] * norms[i]);
    if (!Number.isFinite(s)) throw new PreprocessingError('INVALID_BOUNDARY_EMBEDDING', `the similarity between boundary atoms ${i - 1} and ${i} is not finite`);
    sims.push(s);
  }
  return sims;
}

/** Plans a document's chunks. Throws PreprocessingError; never returns a plan that drops or exceeds anything. */
export async function planChunks(text: string, pipelineRaw: DocumentPipeline, tokenizer: TokenizerAdapter, options: PlanOptions = {}): Promise<ChunkPlan> {
  let pipeline: DocumentPipeline;
  try {
    pipeline = validatePipeline(pipelineRaw);
  } catch (e) {
    throw new PreprocessingError('INVALID_PIPELINE', (e as Error).message, { cause: e });
  }
  const idProblems = tokenizerIdentityProblems(tokenizer?.identity);
  if (idProblems.length || canonicalDigest(tokenizer.identity) !== canonicalDigest(pipeline.tokenizer)) {
    throw new PreprocessingError('TOKENIZER_MISMATCH', `the tokenizer adapter does not implement the pipeline's tokenizer identity${idProblems.length ? `: ${idProblems.join('; ')}` : ''}`);
  }
  const grouping = pipeline.grouping;
  if (grouping.algorithm === 'adjacent-cosine/1') {
    const b = options.boundary;
    if (!b?.encoder || typeof b.encoder.embed !== 'function' || !b.tokenizer) throw new PreprocessingError('BOUNDARY_CAPABILITY_MISSING', "semantic grouping needs the pipeline's boundary encoder and tokenizer adapters");
    if (canonicalDigest(identityFields(b.encoder.identity)) !== canonicalDigest(identityFields(grouping.boundaryEncoderIdentity))) throw new PreprocessingError('BOUNDARY_CAPABILITY_MISSING', 'the boundary encoder does not implement the pipeline boundary encoder identity');
    if (tokenizerIdentityProblems(b.tokenizer.identity).length || canonicalDigest(b.tokenizer.identity) !== canonicalDigest(grouping.boundaryTokenizer)) throw new PreprocessingError('BOUNDARY_CAPABILITY_MISSING', 'the boundary tokenizer does not implement the pipeline boundary tokenizer identity');
  }
  if (typeof text !== 'string') throw new PreprocessingError('EMPTY_INPUT', 'the document text must be a string');
  if (!/\S/u.test(text)) throw new PreprocessingError('EMPTY_INPUT', 'empty and whitespace-only documents are rejected');
  if (options.maxInputUtf8Bytes !== undefined && utf8Length(text) > options.maxInputUtf8Bytes) throw new PreprocessingError('INPUT_TOO_LARGE', `the document exceeds ${options.maxInputUtf8Bytes} UTF-8 bytes`);
  const limit = options.contextCharLimit ?? null;
  if (limit !== null && !(Number.isInteger(limit) && limit >= 1)) throw new PreprocessingError('INVALID_PIPELINE', 'contextCharLimit must be a positive integer or null');
  const yieldEvery = options.yieldEvery ?? 256;
  if (!(Number.isInteger(yieldEvery) && yieldEvery >= 1)) throw new PreprocessingError('INVALID_PIPELINE', 'yieldEvery must be a positive integer');
  if (options.deadline !== undefined && !Number.isFinite(options.deadline)) throw new PreprocessingError('INVALID_PIPELINE', 'deadline must be a finite time');
  const pl = new Planner(text, pipeline.maxPlanningSteps, options.signal, yieldEvery, options.deadline);
  pl.checkAbort();

  const atoms = splitSpans(text, pipeline.splitter.algorithm);
  let units: Unit[];
  if (grouping.algorithm === 'adjacent-cosine/1') {
    units = await boundaryUnits(pl, atoms, grouping, options.boundary!.tokenizer);
    const sims = adjacentSimilarities(await boundaryVectors(pl, units, options.boundary!));
    units.forEach((u, i) => { u.groupStart = i > 0 && sims[i] < grouping.threshold; });
  } else {
    units = atoms.map((s, atom) => ({ start: s.start, end: s.end, atom, groupStart: false }));
  }
  const chunks = await pack(pl, units, tokenizer, pipeline.maxInputTokens, limit, pipeline.maxChunks);

  // Final verification: exact recount of every complete input, and exact coverage.
  for (const c of chunks) {
    const n = pl.count(tokenizer, text.slice(c.context.start, c.context.end));
    if (n !== c.inputTokens) throw new PreprocessingError('TOKENIZER_FAILURE', `countInput is not deterministic (chunk ${c.index}: ${c.inputTokens} then ${n})`);
    if (n > pipeline.maxInputTokens || (limit !== null && c.context.end - c.context.start > limit)) throw new PreprocessingError('TOKEN_BUDGET_UNSATISFIABLE', `chunk ${c.index} exceeds the budget`);
  }
  const coverage = partitionProblems(text, chunks.map((c) => c.core));
  if (coverage.length) throw new PreprocessingError('TOKENIZER_FAILURE', `internal coverage failure: ${coverage.join('; ')}`);
  return { pipelineDigest: pipelineDigest(pipeline), sourceDigest: sourceDigest(text), contextCharLimit: limit, chunks, diagnostics: pl.diagnostics };
}

/**
 * Problems with a stored plan for reuse against `text` and `pipeline` (empty when it may be reused):
 * both digests, the character limit (declared and per context), exact coverage, indices, each chunk's
 * atoms, context = core, budgets and boundaries. With a tokenizer, every input is recounted too.
 */
export function chunkPlanProblems(plan: ChunkPlan, text: string, pipeline: DocumentPipeline, options: { contextCharLimit?: number | null; tokenizer?: TokenizerAdapter } = {}): string[] {
  const p: string[] = [];
  if (!plan || !Array.isArray(plan.chunks)) return ['not a chunk plan'];
  if (plan.pipelineDigest !== pipelineDigest(pipeline)) p.push('the plan was made with another pipeline');
  if (plan.sourceDigest !== sourceDigest(text)) p.push('the plan was made for another text');
  const limit = options.contextCharLimit ?? null;
  if ((plan.contextCharLimit ?? null) !== limit) p.push('the plan was made with another character limit');
  const coverage = partitionProblems(text, plan.chunks.map((c) => c.core));
  p.push(...coverage);
  if (plan.chunks.length > pipeline.maxChunks) p.push('the plan has more chunks than the pipeline allows');
  const allowed = pipeline.grouping.algorithm === 'adjacent-cosine/1' ? ['document-end', 'structure', 'budget', 'semantic'] : ['document-end', 'structure', 'budget'];
  // With exact coverage, each chunk's atoms are exactly the splitter's atoms its core overlaps.
  const atoms = coverage.length ? null : splitSpans(text, pipeline.splitter.algorithm);
  let first = 0;
  plan.chunks.forEach((c, i) => {
    if (c.index !== i) p.push(`chunk ${i} has index ${c.index}`);
    if (c.context?.start !== c.core?.start || c.context?.end !== c.core?.end) p.push(`chunk ${i}: context must equal core (overlap 0)`);
    if (limit !== null && c.context && c.context.end - c.context.start > limit) p.push(`chunk ${i} has ${c.context.end - c.context.start} characters, over the ${limit} limit`);
    if (!Number.isInteger(c.inputTokens) || c.inputTokens < 0 || c.inputTokens > pipeline.maxInputTokens) p.push(`chunk ${i} has ${c.inputTokens} input tokens, over the ${pipeline.maxInputTokens} budget`);
    if (!allowed.includes(c.boundary)) p.push(`chunk ${i} has boundary ${c.boundary}`);
    if ((i === plan.chunks.length - 1) !== (c.boundary === 'document-end')) p.push(`chunk ${i}: only the last chunk ends at 'document-end'`);
    if (atoms) {
      while (atoms[first].end <= c.core.start) first++;
      const expected: number[] = [];
      for (let a = first; a < atoms.length && atoms[a].start < c.core.end; a++) expected.push(a);
      if (!Array.isArray(c.atomIndices) || c.atomIndices.length !== expected.length || c.atomIndices.some((a: number, k: number) => a !== expected[k])) p.push(`chunk ${i}: atomIndices must be ${JSON.stringify(expected)}, the atoms its core covers`);
    }
    if (options.tokenizer && p.length === 0) {
      let n: number | null = null;
      try { n = options.tokenizer.countInput(text.slice(c.context.start, c.context.end)); } catch (e) { p.push(`chunk ${i}: recount failed: ${(e as Error).message}`); return; }
      if (n !== c.inputTokens) p.push(`chunk ${i}: recount differs`);
    }
  });
  return p;
}

/** The plan's identity, e.g. for document feature records and cache keys. */
export const planDigest = (plan: ChunkPlan): string => canonicalDigest(plan);
