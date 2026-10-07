import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  adjacentSimilarities, chunkPlanProblems, createPipeline, fixtureTokenizer, pipelineProblems, planChunks, PreprocessingError,
  type EncoderAdapter, type FullEncoderIdentity,
} from '../src/index.ts';

const tok = fixtureTokenizer();
const boundaryTok = fixtureTokenizer({ revision: 'boundary-tok/1', special: 1 });
const BOUNDARY: FullEncoderIdentity = {
  schema: 'liquidau-encoder/1', modelId: 'fake/boundary', revision: 'b1', dimensions: 2, precision: 'fp32', inputType: null, layers: null, pooling: 'mean',
  normalization: 'unit', tokenizerRevision: 'boundary-tok/1', maxChars: null, maxTokens: 32, truncation: 'none',
};

/** Topic vectors: cat -> [1,0], dog -> [0,1], otherwise the diagonal. Records batches; optional delays scramble completion order. */
function boundaryEncoder(options: { delay?: (call: number) => number; vector?: (t: string) => number[] } = {}): EncoderAdapter & { batches: string[][] } {
  const batches: string[][] = [];
  const v = options.vector ?? ((t: string) => (/cat/.test(t) ? [1, 0] : /dog/.test(t) ? [0, 1] : [Math.SQRT1_2, Math.SQRT1_2]));
  return {
    identity: BOUNDARY, batches,
    async embed(texts) {
      const call = batches.push([...texts]) - 1;
      const ms = options.delay?.(call) ?? 0;
      if (ms) await new Promise((r) => setTimeout(r, ms));
      return texts.map(v);
    },
  };
}

const pipeline = (threshold = 0.5, over: { maxInputTokens?: number; boundaryMaxInputTokens?: number; maxPlanningSteps?: number } = {}) => createPipeline({
  splitter: 'sentence-simple/1', tokenizer: tok.identity, maxInputTokens: over.maxInputTokens ?? 200, maxPlanningSteps: over.maxPlanningSteps,
  semantic: { threshold, boundaryEncoderIdentity: BOUNDARY, boundaryTokenizer: boundaryTok.identity, boundaryMaxInputTokens: over.boundaryMaxInputTokens ?? 32 },
});
const pieces = (text: string, plan: Awaited<ReturnType<typeof planChunks>>) => plan.chunks.map((c) => [text.slice(c.core.start, c.core.end), c.boundary]);

test('a new group starts strictly below the threshold; equality does not split', async () => {
  const text = 'A cat sat. Another cat. A dog ran. The dog barked.';
  const plan = await planChunks(text, pipeline(0.5), tok, { boundary: { encoder: boundaryEncoder(), tokenizer: boundaryTok } });
  assert.deepEqual(pieces(text, plan), [['A cat sat. Another cat.', 'semantic'], [' A dog ran. The dog barked.', 'document-end']]);
  // cat -> diagonal has similarity exactly SQRT1_2; a threshold of exactly that keeps them together.
  const mixed = 'A cat. Plain words.';
  const sims = adjacentSimilarities([[1, 0], [Math.SQRT1_2, Math.SQRT1_2]]);
  const atEqual = await planChunks(mixed, pipeline(sims[1]), tok, { boundary: { encoder: boundaryEncoder(), tokenizer: boundaryTok } });
  assert.equal(atEqual.chunks.length, 1);
  const above = await planChunks(mixed, pipeline(sims[1] + 1e-12), tok, { boundary: { encoder: boundaryEncoder(), tokenizer: boundaryTok } });
  assert.equal(above.chunks.length, 2);
});

test('batch size, concurrency and completion order never change boundaries, including pairs across batches', async () => {
  const text = Array.from({ length: 40 }, (_, i) => (Math.floor(i / 3) % 2 ? 'The dog barked.' : 'A cat sat.')).join(' ');
  const reference = await planChunks(text, pipeline(), tok, { boundary: { encoder: boundaryEncoder(), tokenizer: boundaryTok, batchSize: 256 } });
  for (const [batchSize, concurrency] of [[1, 1], [2, 4], [3, 3], [7, 16]]) {
    const enc = boundaryEncoder({ delay: (c) => (c % 3 === 0 ? 4 : 0) });
    const plan = await planChunks(text, pipeline(), tok, { boundary: { encoder: enc, tokenizer: boundaryTok, batchSize, concurrency } });
    assert.deepEqual(plan, reference, `batch ${batchSize} x ${concurrency}`);
    assert.ok(enc.batches.every((b) => b.length <= batchSize));
  }
  assert.ok(reference.chunks.filter((c) => c.boundary === 'semantic').length >= 10);
  assert.deepEqual(chunkPlanProblems(reference, text, pipeline()), []);
});

test('oversized atoms are hard-split for the boundary encoder; groups over the classification budget are cut too', async () => {
  const long = `cat ${'word '.repeat(60)}end.`;
  const enc = boundaryEncoder();
  const plan = await planChunks(long, pipeline(0.5, { maxInputTokens: 40, boundaryMaxInputTokens: 20 }), tok, { boundary: { encoder: enc, tokenizer: boundaryTok } });
  const sent = enc.batches.flat();
  assert.ok(sent.length > 1, 'the atom was split into several boundary inputs');
  for (const s of sent) assert.ok(boundaryTok.countInput(s) <= 20);
  assert.equal(sent.join(''), long, 'subatoms partition the atom');
  for (const c of plan.chunks) assert.ok(c.inputTokens <= 40 && c.atomIndices.every((a) => a === 0));
  assert.equal(plan.chunks.map((c) => long.slice(c.core.start, c.core.end)).join(''), long);
});

test('zero norms, bad vectors, missing capabilities and the step budget fail before or instead of grouping', async () => {
  const code = (c: string) => (e: unknown) => e instanceof PreprocessingError && e.code === c;
  const text = 'A cat. A dog.';
  await assert.rejects(planChunks(text, pipeline(), tok, { boundary: { encoder: boundaryEncoder({ vector: () => [0, 0] }), tokenizer: boundaryTok } }), code('INVALID_BOUNDARY_EMBEDDING'), 'zero norms fail (and fail the unit-norm check)');
  assert.throws(() => adjacentSimilarities([[1, 0], [0, 0]]), /zero or non-finite norm/);
  await assert.rejects(planChunks(text, pipeline(), tok, { boundary: { encoder: boundaryEncoder({ vector: () => [1] }), tokenizer: boundaryTok } }), code('INVALID_BOUNDARY_EMBEDDING'));
  await assert.rejects(planChunks(text, pipeline(), tok), code('BOUNDARY_CAPABILITY_MISSING'));
  await assert.rejects(planChunks(text, pipeline(), tok, { boundary: { encoder: { ...boundaryEncoder(), identity: { ...BOUNDARY, revision: 'b2' } }, tokenizer: boundaryTok } }), code('BOUNDARY_CAPABILITY_MISSING'));
  await assert.rejects(planChunks(text, pipeline(), tok, { boundary: { encoder: boundaryEncoder(), tokenizer: tok } }), code('BOUNDARY_CAPABILITY_MISSING'));
  await assert.rejects(planChunks(text, pipeline(), tok, { boundary: { encoder: { identity: BOUNDARY, embed: async () => { throw Object.assign(new Error('503'), { retryable: true }); } }, tokenizer: boundaryTok } }), (e: unknown) => e instanceof PreprocessingError && e.code === 'BOUNDARY_ENCODER_FAILURE' && e.retryable === true);
  // The step budget covers boundary preparation, charged before any request.
  const enc = boundaryEncoder();
  const many = 'A cat. '.repeat(50);
  await assert.rejects(planChunks(many, pipeline(0.5, { maxPlanningSteps: 60 }), tok, { boundary: { encoder: enc, tokenizer: boundaryTok } }), code('MAX_PLANNING_STEPS_EXCEEDED'));
  assert.equal(enc.batches.length, 0, 'no boundary request was issued');
});

test('semantic pipelines validate the boundary contract; every setting changes the digest', () => {
  const base = pipeline();
  assert.deepEqual(pipelineProblems(base), []);
  const g = base.grouping as Extract<typeof base.grouping, { algorithm: 'adjacent-cosine/1' }>;
  const bad: Array<[unknown, RegExp]> = [
    [{ ...base, grouping: { ...g, threshold: 1.5 } }, /threshold/], [{ ...base, grouping: { ...g, threshold: Number.NaN } }, /threshold/],
    [{ ...base, grouping: { ...g, boundaryEncoderIdentity: { ...BOUNDARY, truncation: 'head/1' } } }, /truncates/],
    [{ ...base, grouping: { ...g, boundaryMaxInputTokens: 64 } }, /exceeds the boundary encoder/],
    [{ ...base, grouping: { ...g, boundaryTokenizer: tok.identity } }, /tokenizer revision/],
    [{ ...base, grouping: { ...g, boundaryPreparation: 'merge-atoms/1' } }, /boundaryPreparation/],
    [{ ...base, grouping: { ...g, extra: 1 } }, /unknown field extra/],
  ];
  for (const [raw, pattern] of bad) assert.match(pipelineProblems(raw).join(), pattern);
});
