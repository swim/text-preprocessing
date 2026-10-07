import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';

import {
  canonicalDigest, canonicalJson, chunkPlanProblems, createPipeline, fixtureTokenizer, partitionProblems, pipelineDigest, pipelineProblems, planChunks,
  PreprocessingError, sha256Hex, sourceDigest, splitSpans, tokenizerContractProblems, type DocumentPipeline, type TokenizerAdapter,
} from '../src/index.ts';

const tok = fixtureTokenizer();
const pipeline = (over: Partial<DocumentPipeline> = {}, splitter: 'paragraph/1' | 'sentence-simple/1' = 'sentence-simple/1'): DocumentPipeline => ({ ...createPipeline({ splitter, tokenizer: tok.identity, maxInputTokens: 12 }), ...over });
const texts = (text: string, spans: { start: number; end: number }[]) => spans.map((s) => text.slice(s.start, s.end));

test('sentence-simple/1: separator runs belong to the left span; leading runs stand alone; exact reconstruction', () => {
  const text = '..Hello world. How are you?! Fine;\r\nthanks 3.5 kg';
  assert.deepEqual(texts(text, splitSpans(text, 'sentence-simple/1')), ['..', 'Hello world.', ' How are you?!', ' Fine;\r\n', 'thanks 3.', '5 kg']);
  assert.deepEqual(splitSpans('', 'sentence-simple/1'), []);
  assert.deepEqual(texts('no separators', splitSpans('no separators', 'sentence-simple/1')), ['no separators']);
});

test('paragraph/1: two or more line breaks split, one does not; indentation belongs to the run', () => {
  const text = 'First line\nsame paragraph\n\n  \tSecond\r\n\r\nThird\n \n\nFourth\n';
  assert.deepEqual(texts(text, splitSpans(text, 'paragraph/1')), ['First line\nsame paragraph\n\n  \t', 'Second\r\n\r\n', 'Third\n \n\n', 'Fourth\n']);
  assert.deepEqual(texts('\n\nlead', splitSpans('\n\nlead', 'paragraph/1')), ['\n\n', 'lead']);
});

/** A reproducible generator over hard characters: emoji, combining marks, full-width punctuation, CRLF, separators. */
function* corpus(n: number) {
  let s = 12345;
  const rand = () => { s = (s * 1103515245 + 12345) >>> 0; return s / 2 ** 32; };
  const parts = ['word', ' ', '  ', '.', '!?', ';', '\n', '\r\n', '\n\n', '\t', '🙂', 'é', '＂', '。', 'Ｈｅｌｌｏ', 'ﬁ', 'repeat repeat', '3.5', '\uD800'];
  for (let k = 0; k < n; k++) {
    const len = 1 + Math.floor(rand() * 60);
    yield Array.from({ length: len }, () => parts[Math.floor(rand() * parts.length)]).join('');
  }
}

test('property: splitters and plans partition the original text exactly on code-point boundaries', async () => {
  for (const text of corpus(300)) {
    for (const alg of ['paragraph/1', 'sentence-simple/1'] as const) {
      assert.deepEqual(partitionProblems(text, splitSpans(text, alg)), [], JSON.stringify(text));
      if (!/\S/u.test(text)) continue;
      const plan = await planChunks(text, pipeline({ maxChunks: 1000 }, alg), tok);
      assert.equal(plan.chunks.map((c) => text.slice(c.core.start, c.core.end)).join(''), text);
      for (const c of plan.chunks) {
        assert.ok(c.inputTokens <= 12 && c.inputTokens === tok.countInput(text.slice(c.core.start, c.core.end)));
        assert.deepEqual(c.context, c.core);
      }
      assert.deepEqual(chunkPlanProblems(plan, text, pipeline({ maxChunks: 1000 }, alg), { tokenizer: tok }), []);
    }
  }
});

test('hard budgets count the complete prepared input, prefix and special tokens included', async () => {
  const prefixed = fixtureTokenizer({ prefix: 'query: ', special: 3 });
  const p = createPipeline({ splitter: 'sentence-simple/1', tokenizer: prefixed.identity, maxInputTokens: 10 });
  const text = 'alpha beta gamma. delta epsilon zeta eta theta iota kappa lambda mu.';
  const plan = await planChunks(text, p, prefixed);
  for (const c of plan.chunks) assert.ok(prefixed.countInput(text.slice(c.core.start, c.core.end)) <= 10);
  assert.ok(plan.chunks.some((c) => c.boundary === 'budget'), 'an oversized sentence was cut');
  assert.equal(plan.chunks.at(-1)!.boundary, 'document-end');
  // Counts are recounted, never summed: joining two fitting atoms can still overflow.
  assert.ok(tok.countInput('a. b.') !== tok.countInput('a.') + tok.countInput(' b.'));
});

test('impossible budgets, limits and cancellation fail the whole document explicitly', async () => {
  const code = (c: string) => (e: unknown) => e instanceof PreprocessingError && e.code === c;
  await assert.rejects(planChunks('anything', pipeline({ maxInputTokens: 2 }), tok), code('TOKEN_BUDGET_UNSATISFIABLE'), 'special tokens alone fill the budget');
  await assert.rejects(planChunks('x'.repeat(50), pipeline({ maxInputTokens: 100 }), tok, { contextCharLimit: 10 }), code('TOKEN_BUDGET_UNSATISFIABLE'), 'no legal cut inside one long token');
  await assert.rejects(planChunks('a. b. c. d. e. f.', pipeline({ maxInputTokens: 4, maxChunks: 2 }), tok), code('MAX_CHUNKS_EXCEEDED'));
  await assert.rejects(planChunks('a. b. c. d. e. f.', pipeline({ maxPlanningSteps: 3 }), tok), code('MAX_PLANNING_STEPS_EXCEEDED'));
  await assert.rejects(planChunks('   \n\t ', pipeline(), tok), code('EMPTY_INPUT'));
  await assert.rejects(planChunks('', pipeline(), tok), code('EMPTY_INPUT'));
  await assert.rejects(planChunks('héllo', pipeline(), tok, { maxInputUtf8Bytes: 5 }), code('INPUT_TOO_LARGE'));
  const aborted = new AbortController();
  aborted.abort();
  await assert.rejects(planChunks('a. b.', pipeline(), tok, { signal: aborted.signal }), code('ABORTED'));
  // A timer abort lands between steps because planning yields.
  const later = new AbortController();
  setTimeout(() => later.abort(), 1);
  await assert.rejects(planChunks('a. '.repeat(5000), pipeline({ maxChunks: 65536, maxPlanningSteps: 1 << 24 }), tok, { signal: later.signal, yieldEvery: 16 }), code('ABORTED'));
  await assert.rejects(planChunks('a', pipeline(), fixtureTokenizer({ revision: 'other' })), code('TOKENIZER_MISMATCH'));
  const broken: TokenizerAdapter = { identity: tok.identity, countInput: () => 1.5, cutOffsets: () => [] };
  await assert.rejects(planChunks('a', pipeline(), broken), code('TOKENIZER_FAILURE'));
  const badCuts: TokenizerAdapter = { identity: tok.identity, countInput: (t) => t.length, cutOffsets: (t) => [t.indexOf('\uDE42')] };
  await assert.rejects(planChunks('xx🙂xxxxxxxxxxxxxxxxxx', pipeline({ maxInputTokens: 5 }), badCuts), code('TOKENIZER_FAILURE'), 'a cut inside a surrogate pair');
});

test('the character limit is honoured, and a forced cut before a combining mark is diagnosed', async () => {
  const text = 'abcdefghij klmnopqrst. uvw.';
  const plan = await planChunks(text, pipeline({ maxInputTokens: 100 }), tok, { contextCharLimit: 12 });
  assert.ok(plan.chunks.every((c) => c.core.end - c.core.start <= 12));
  assert.equal(plan.contextCharLimit, 12);
  const marks: TokenizerAdapter = { identity: tok.identity, countInput: (t) => t.length, cutOffsets: (t) => [...t].map((_, i) => i).filter((i) => i > 0) };
  const m = await planChunks('éééé', pipeline({ maxInputTokens: 3 }), marks);
  assert.ok(m.diagnostics.some((d) => /combining mark/.test(d)));
});

test('plans are deterministic and bound to text, pipeline and limits', async () => {
  const text = 'One. Two two. Three three three. Four four four four.';
  const a = await planChunks(text, pipeline(), tok), b = await planChunks(text, pipeline(), tok);
  assert.deepEqual(a, b);
  assert.match(chunkPlanProblems(a, `${text} `, pipeline()).join(), /another text/);
  assert.match(chunkPlanProblems(a, text, pipeline({ maxInputTokens: 13 })).join(), /another pipeline/);
  assert.match(chunkPlanProblems(a, text, pipeline(), { contextCharLimit: 40 }).join(), /character limit/);
  const tampered = { ...a, chunks: a.chunks.slice(1) };
  assert.ok(chunkPlanProblems(tampered, text, pipeline()).length);
});

test('pipelines: explicit fields only, unsupported algorithms and overlap refused, digest changes with every setting', () => {
  const base = pipeline();
  assert.deepEqual(pipelineProblems(base), []);
  const bad: Array<[unknown, RegExp]> = [
    [{ ...base, overlap: 1 }, /overlap/], [{ ...base, extra: true }, /unknown field extra/], [{ ...base, schema: 'liquidau-document-pipeline/3' }, /unsupported pipeline schema/],
    [{ ...base, splitter: { algorithm: 'intl-segmenter/1' } }, /splitter/], [{ ...base, grouping: { algorithm: 'adjacent-cosine/1', threshold: 0.5 } }, /boundaryEncoderIdentity/],
    [{ ...base, tokenizer: { ...base.tokenizer, revision: 'provider-managed' } }, /provider-managed/], [{ ...base, maxInputTokens: 0 }, /maxInputTokens/],
    [{ ...base, aggregation: { algorithm: 'max-chunk-vector/1', normalization: 'none' } }, /aggregation/], [{ ...base, emptyInput: 'pass' }, /emptyInput/],
  ];
  for (const [raw, pattern] of bad) assert.match(pipelineProblems(raw).join(), pattern);
  const digests = new Set([base, { ...base, maxChunks: 65 }, { ...base, maxPlanningSteps: 1 }, { ...base, maxInputTokens: 13 }, pipeline({}, 'paragraph/1'), { ...base, tokenizer: { ...base.tokenizer, inputPreparation: 'x' } }].map(pipelineDigest));
  assert.equal(digests.size, 6);
});

test('digests: SHA-256 matches node:crypto; canonical JSON sorts keys; source digests are injective on lone surrogates', () => {
  for (const s of ['', 'abc', 'x'.repeat(1000), '🙂 ünïcode']) {
    const bytes = new TextEncoder().encode(s);
    assert.equal(sha256Hex(bytes), createHash('sha256').update(bytes).digest('hex'));
  }
  assert.equal(canonicalJson({ b: 1, a: [2, { d: 1, c: 0 }] }), '{"a":[2,{"c":0,"d":1}],"b":1}');
  assert.equal(canonicalDigest({ b: 1, a: 2 }), canonicalDigest({ a: 2, b: 1 }));
  assert.throws(() => canonicalJson({ a: Number.NaN }));
  assert.notEqual(sourceDigest('\uD800'), sourceDigest('\uD801'), 'TextEncoder would map both to U+FFFD');
  assert.notEqual(sourceDigest('a'), sourceDigest('a '));
});

test('tokenizer contract: the fixture conforms, broken adapters are caught', () => {
  assert.deepEqual(tokenizerContractProblems(tok), []);
  let n = 0;
  assert.match(tokenizerContractProblems({ identity: tok.identity, countInput: () => n++, cutOffsets: () => [] }).join(), /not deterministic/);
  assert.match(tokenizerContractProblems({ identity: tok.identity, countInput: () => 1, cutOffsets: (t) => [t.length] }).join(), /illegal offset/);
  assert.match(tokenizerContractProblems({ identity: { ...tok.identity, revision: 'provider-managed' }, countInput: () => 1, cutOffsets: () => [] }).join(), /provider-managed/);
});

test('a deadline is checked at every planning step, even when no timer can fire', async () => {
  const slow: TokenizerAdapter = { identity: tok.identity, countInput: (t) => { const end = performance.now() + 15; while (performance.now() < end) { /* busy */ } return tok.countInput(t); }, cutOffsets: tok.cutOffsets };
  const deadline = performance.now() + 10;
  await assert.rejects(planChunks('One. Two. Three.', pipeline(), slow, { deadline, yieldEvery: 1_000_000 }), (e: unknown) => e instanceof PreprocessingError && e.code === 'DEADLINE_EXCEEDED');
  assert.ok((await planChunks('One. Two.', pipeline(), tok, { deadline: performance.now() + 60_000 })).chunks.length >= 1);
});

test('the contract can require special tokens on the empty string', () => {
  assert.deepEqual(tokenizerContractProblems(tok, undefined, { minEmptyCount: 2 }), []);
  assert.match(tokenizerContractProblems(fixtureTokenizer({ special: 0 }), undefined, { minEmptyCount: 1 }).join(), /empty string counts 0/);
});
