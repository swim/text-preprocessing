import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createStaticEmbedder, createWordPiece, DEFAULT_LEXICAL, lexicalFeatures, lexicalSpecProblems, sha256Hex, staticIdentityProblems } from '../src/index.ts';

test('lexicalFeatures: deterministic, unit length, sorted unique indices within 2^bits', () => {
  const a = lexicalFeatures('My card was STOLEN, my card!');
  const b = lexicalFeatures('my card was stolen my card');
  assert.deepEqual([...a.index], [...b.index], 'case and punctuation do not change features');
  assert.deepEqual([...a.value], [...b.value]);
  const norm = Math.sqrt([...a.value].reduce((s, v) => s + v * v, 0));
  assert.ok(Math.abs(norm - 1) < 1e-6);
  for (let i = 1; i < a.index.length; i++) assert.ok(a.index[i] > a.index[i - 1]);
  assert.ok([...a.index].every((k) => k < 2 ** DEFAULT_LEXICAL.bits));
  assert.equal(lexicalFeatures('').index.length, 0);
  // Repeated words weigh √count: "card" appears twice.
  assert.ok(Math.max(...a.value) > Math.min(...a.value));
  assert.deepEqual(lexicalSpecProblems(DEFAULT_LEXICAL), []);
  assert.ok(lexicalSpecProblems({ schema: 'x', bits: 40 }).length === 2);
});

const VOCAB = ['[PAD]', '[UNK]', 'the', 'card', 'was', 'stol', '##en', 'cafe', ',', '!', '中', 'un', '##aff', '##able'];
const TOKENIZER = {
  normalizer: { type: 'BertNormalizer', clean_text: true, handle_chinese_chars: true, strip_accents: null, lowercase: true },
  pre_tokenizer: { type: 'BertPreTokenizer' },
  model: { type: 'WordPiece', unk_token: '[UNK]', continuing_subword_prefix: '##', max_input_chars_per_word: 100, vocab: Object.fromEntries(VOCAB.map((t, i) => [t, i])) },
};
const id = (t: string) => VOCAB.indexOf(t);

test('createWordPiece: BERT normalisation, punctuation splits, greedy ## pieces, unknown words', () => {
  const wp = createWordPiece(TOKENIZER);
  assert.deepEqual(wp.encode('The card was STOLEN!'), [id('the'), id('card'), id('was'), id('stol'), id('##en'), id('!')]);
  assert.deepEqual(wp.encode('Café,card'), [id('cafe'), id(','), id('card')], 'accents stripped, punctuation isolated');
  assert.deepEqual(wp.encode('unaffable'), [id('un'), id('##aff'), id('##able')]);
  assert.deepEqual(wp.encode('zzz card'), [id('[UNK]'), id('card')], 'a word with no pieces is one unknown token');
  assert.deepEqual(wp.encode('中card'), [id('中'), id('card')], 'CJK characters are split out');
  assert.deepEqual(wp.encode('card\u0000​\tcard'), [id('card'), id('card')], 'control characters dropped, whitespace normalised');
  assert.throws(() => createWordPiece({ ...TOKENIZER, model: { type: 'BPE' } }), /WordPiece/);
});

test('createStaticEmbedder: unit-length mean of known tokens, maxTokens, identity digests', async () => {
  const dims = 2;
  const weights = new Float32Array(VOCAB.length * dims);
  weights.set([1, 0], id('card') * dims);
  weights.set([0, 1], id('the') * dims);
  weights.set([5, 5], id('[UNK]') * dims);
  const bytes = new TextEncoder().encode(JSON.stringify(TOKENIZER));
  const e = await createStaticEmbedder({ modelId: 'fixture', tokenizerBytes: bytes, weights, dimensions: dims });
  const v = e.embed('the card zzz');
  assert.ok(Math.abs(v[0] - Math.SQRT1_2) < 1e-6 && Math.abs(v[1] - Math.SQRT1_2) < 1e-6, 'unknown tokens are dropped');
  const one = await createStaticEmbedder({ modelId: 'fixture', tokenizerBytes: bytes, weights, dimensions: dims, maxTokens: 1 });
  assert.deepEqual([...one.embed('the card')], [0, 1], 'only the first maxTokens known tokens count');
  assert.deepEqual([...e.embed('zzz')], [0, 0], 'no known tokens: the zero vector');
  assert.equal(e.identity.tokenizerSha256, sha256Hex(bytes));
  assert.deepEqual(staticIdentityProblems(e.identity), []);
  assert.ok(staticIdentityProblems({ ...e.identity, weightsSha256: 'x' }).length === 1);
  await assert.rejects(createStaticEmbedder({ modelId: 'f', tokenizerBytes: bytes, weights: new Float32Array(3), dimensions: 2 }), /multiple/);
});

test('createStaticEmbedder refuses vocabulary ids past the weight table, and lexicalFeatures an out-of-range spec', async () => {
  const gappy = { ...TOKENIZER, model: { ...TOKENIZER.model, vocab: Object.fromEntries(VOCAB.map((t, i) => [t, i === VOCAB.length - 1 ? VOCAB.length + 5 : i])) } };
  const weights = new Float32Array(VOCAB.length * 2);
  await assert.rejects(createStaticEmbedder({ modelId: 'fixture', tokenizerBytes: new TextEncoder().encode(JSON.stringify(gappy)), weights, dimensions: 2 }), /largest id is \d+, but the weights have only/);
  assert.throws(() => lexicalFeatures('hello', { ...DEFAULT_LEXICAL, bits: 31 }), /bits must be an integer in \[10, 24\]/);
});
