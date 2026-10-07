import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  canonicalJson, chunkPlanProblems, createPipeline, fixtureTokenizer, pipelineDigest, pipelineProblems, planChunks, PreprocessingError, splitSpans,
  type EncoderAdapter, type FullEncoderIdentity, type TokenizerAdapter,
} from '../src/index.ts';

const tok = fixtureTokenizer();
const code = (c: string) => (e: unknown) => e instanceof PreprocessingError && e.code === c;
const counting = (inner: TokenizerAdapter): TokenizerAdapter & { calls: number } => {
  const t = { identity: inner.identity, calls: 0, countInput: (s: string) => { t.calls++; return inner.countInput(s); }, cutOffsets: (s: string) => inner.cutOffsets(s) };
  return t;
};
/** n words with a single line break every 12 words: one paragraph under paragraph/1. */
const longParagraph = (n: number) => Array.from({ length: n }, (_, i) => `word${i % 97}${i % 12 === 11 ? '\n' : ' '}`).join('').trimEnd();

test('a long paragraph plans within the default limits, with O(log n) recounts per cut', async () => {
  const text = longParagraph(8000);
  const t = counting(tok);
  const plan = await planChunks(text, createPipeline({ splitter: 'paragraph/1', tokenizer: tok.identity, maxInputTokens: 384 }), t);
  assert.equal(plan.chunks.map((c) => text.slice(c.core.start, c.core.end)).join(''), text);
  assert.ok(plan.chunks.every((c) => c.inputTokens <= 384));
  assert.ok(t.calls < 40 * plan.chunks.length, `${t.calls} recounts for ${plan.chunks.length} chunks`);
});

test('bracketed/1 cuts at the largest fitting prefix when counts never decrease', async () => {
  for (const [text, max, limit] of [[longParagraph(600), 50, null], [longParagraph(300), 400, 90], ['x '.repeat(400), 17, null]] as const) {
    const pipeline = createPipeline({ splitter: 'paragraph/1', tokenizer: tok.identity, maxInputTokens: max });
    const plan = await planChunks(text, pipeline, tok, { contextCharLimit: limit });
    const atoms = splitSpans(text, 'paragraph/1');
    const cuts = plan.chunks.filter((c) => c.boundary === 'budget');
    assert.ok(cuts.length > 2);
    for (const c of cuts) {
      // The next legal offset in the piece being cut must not fit.
      const atomEnd = atoms.find((a) => a.start <= c.core.start && c.core.start < a.end)!.end;
      const next = tok.cutOffsets(text.slice(c.core.start, atomEnd)).find((o) => o > c.core.end - c.core.start);
      if (next === undefined) continue;
      assert.ok((limit !== null && next > limit) || tok.countInput(text.slice(c.core.start, c.core.start + next)) > max, `a longer prefix fits at chunk ${c.index}`);
    }
  }
});

test('a tokenizer whose counts are not monotone still gets deterministic, fitting, exact plans', async () => {
  // Odd-length inputs cost 3 extra tokens, so a longer prefix can count less than a shorter one.
  const odd: TokenizerAdapter = { identity: tok.identity, countInput: (s) => tok.countInput(s) + (s.length % 2) * 3, cutOffsets: tok.cutOffsets };
  const text = longParagraph(400);
  const pipeline = createPipeline({ splitter: 'paragraph/1', tokenizer: tok.identity, maxInputTokens: 40 });
  const a = await planChunks(text, pipeline, odd), b = await planChunks(text, pipeline, odd);
  assert.deepEqual(a, b);
  assert.deepEqual(chunkPlanProblems(a, text, pipeline, { tokenizer: odd }), []);
});

test('a deadline bounds a boundary request that never returns, and aborts it', async () => {
  const boundaryTok = fixtureTokenizer({ revision: 'boundary-tok/1', special: 1 });
  const identity: FullEncoderIdentity = {
    schema: 'liquidau-encoder/1', modelId: 'fake/boundary', revision: 'b1', dimensions: 2, precision: 'fp32', inputType: null, layers: null, pooling: 'mean',
    normalization: 'unit', tokenizerRevision: 'boundary-tok/1', maxChars: null, maxTokens: 32, truncation: 'none',
  };
  let seen: AbortSignal | undefined;
  const hanging: EncoderAdapter = { identity, embed: (_texts, { signal }) => { seen = signal; return new Promise(() => {}); } };
  const pipeline = createPipeline({ splitter: 'sentence-simple/1', tokenizer: tok.identity, maxInputTokens: 100, semantic: { threshold: 0.5, boundaryEncoderIdentity: identity, boundaryTokenizer: boundaryTok.identity, boundaryMaxInputTokens: 32 } });
  const started = performance.now();
  await assert.rejects(planChunks('A cat. A dog.', pipeline, tok, { boundary: { encoder: hanging, tokenizer: boundaryTok }, deadline: performance.now() + 50 }), code('DEADLINE_EXCEEDED'));
  assert.ok(performance.now() - started < 1000);
  assert.equal(seen?.aborted, true);
});

test('stored plans: contexts over the character limit, wrong atoms and a throwing tokenizer are problems', async () => {
  const text = 'alpha beta gamma delta. epsilon zeta eta theta.';
  const pipeline = createPipeline({ splitter: 'sentence-simple/1', tokenizer: tok.identity, maxInputTokens: 100 });
  const plan = await planChunks(text, pipeline, tok);
  assert.deepEqual(chunkPlanProblems(plan, text, pipeline), []);
  assert.match(chunkPlanProblems({ ...plan, contextCharLimit: 10 }, text, pipeline, { contextCharLimit: 10 }).join(), /over the 10 limit/);
  const wrongAtoms = { ...plan, chunks: plan.chunks.map((c) => ({ ...c, atomIndices: [999, -1] })) };
  assert.match(chunkPlanProblems(wrongAtoms, text, pipeline).join(), /atomIndices must be \[0,1\]/);
  const throwing: TokenizerAdapter = { ...tok, countInput: () => { throw new Error('boom'); } };
  assert.match(chunkPlanProblems(plan, text, pipeline, { tokenizer: throwing }).join(), /recount failed: boom/);
});

test('pipelines: the cut search is an explicit field and identities keep only their defined fields', () => {
  const base = createPipeline({ splitter: 'sentence-simple/1', tokenizer: tok.identity, maxInputTokens: 12 });
  assert.equal(base.cutSearch, 'bracketed/1');
  const { cutSearch: _omitted, ...withoutCutSearch } = base;
  assert.match(pipelineProblems(withoutCutSearch).join(), /cutSearch/);
  assert.match(pipelineProblems({ ...base, schema: 'liquidau-document-pipeline/1' }).join(), /unsupported pipeline schema/);
  const extra = createPipeline({ splitter: 'sentence-simple/1', tokenizer: { ...tok.identity, note: 'x' } as unknown as typeof tok.identity, maxInputTokens: 12 });
  assert.equal(pipelineDigest(extra), pipelineDigest(base));
  const identity: FullEncoderIdentity = {
    schema: 'liquidau-encoder/1', modelId: 'm', revision: 'r', dimensions: 2, precision: 'fp32', inputType: null, layers: null, pooling: 'mean',
    normalization: 'unit', tokenizerRevision: 'b/1', maxChars: null, maxTokens: 32, truncation: 'none',
  };
  const semantic = (id: FullEncoderIdentity) => createPipeline({ splitter: 'sentence-simple/1', tokenizer: tok.identity, maxInputTokens: 12, semantic: { threshold: 0.5, boundaryEncoderIdentity: id, boundaryTokenizer: fixtureTokenizer({ revision: 'b/1' }).identity, boundaryMaxInputTokens: 32 } });
  assert.equal(pipelineDigest(semantic({ ...identity, provider: 'x' } as FullEncoderIdentity)), pipelineDigest(semantic(identity)));
});

test('only registered splitters run, and only plain data is hashed', () => {
  assert.throws(() => splitSpans('a.b', 'constructor' as never), /unknown splitter/);
  assert.throws(() => canonicalJson({ at: new Date(0) }), /Date/);
  assert.throws(() => canonicalJson(new Float32Array(2)), /Float32Array/);
  assert.equal(canonicalJson(Object.assign(Object.create(null), { b: 1, a: 2 })), '{"a":2,"b":1}');
});
