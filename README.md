# @liquidau/text-preprocessing

Reproducible document preprocessing shared by the Liquidau libraries. It provides exact source spans, versioned deterministic splitters, an exact-tokenizer contract, hard token-budget chunk packing and canonical digests. It has no dependencies and no Node built-ins. Tokenizers and encoders are injected.

```ts
import { createPipeline, planChunks, pipelineDigest } from '@liquidau/text-preprocessing';

const pipeline = createPipeline({ splitter: 'sentence-simple/1', tokenizer: tokenizer.identity, maxInputTokens: 384 });
const plan = await planChunks(text, pipeline, tokenizer, { signal, contextCharLimit: encoder.identity.maxChars });
// plan.chunks: ordered, disjoint cores covering [0, text.length) exactly; each inputTokens <= 384
```

- **Spans** are UTF-16 code-unit offsets into the original string. They never split a surrogate pair, and concatenating the cores reconstructs the input exactly.
- **Splitters** are registry entries, never configurable regexes.
  - `paragraph/1` splits on two or more line breaks.
  - `sentence-simple/1` splits on `[.!?;\r\n]+` in the original text.
  - Each separator run belongs to the span on its left.
- **Tokenizers**: `countInput` must count the exact encoder input (prefixes and special tokens included), and `cutOffsets` proposes legal cut points.
  - `fixtureTokenizer` is a deterministic test fixture, not a model tokenizer.
  - `tokenizerContractProblems` is the conformance check adapters should pass.
- **Packing** (`token-pack/1`) recounts every prospective input rather than adding counts. It cuts an oversized atom at a legal offset chosen by the pipeline's `cutSearch`.
  - `bracketed/1` recounts the 1st, 2nd, 4th, 8th, ... candidate prefix until one doesn't fit, then bisects. That is the largest fitting prefix whenever counts never decrease as a prefix grows, at O(log n) recounts per cut, so long paragraphs plan quickly. For any tokenizer the choice is deterministic and the chosen prefix is counted and fits.
  - Planning fails with `TOKEN_BUDGET_UNSATISFIABLE`, `MAX_CHUNKS_EXCEEDED`, `MAX_PLANNING_STEPS_EXCEEDED`, `INPUT_TOO_LARGE`, `EMPTY_INPUT`, `TOKENIZER_FAILURE`, `TOKENIZER_MISMATCH` or `ABORTED`.
  - It never silently drops or truncates text.
- **Identity**: `pipelineDigest` is the SHA-256 of the pipeline's canonical JSON. `sourceDigest` hashes the exact UTF-16 string, which keeps it injective for lone surrogates. `chunkPlanProblems` validates a stored plan before reuse: digests, coverage, budgets, each context against the character limit and each chunk's atoms.
- **Semantic grouping** (`adjacent-cosine/1`, experimental) needs a boundary encoder and a boundary tokenizer, injected through `planChunks`'s `boundary` option.
  - Atoms over the boundary budget are first hard-split with the same legal-cut search (`hard-split-atoms/1`).
  - Each piece gets one validated vector, in batches whose size and completion order never change the result.
  - Adjacent cosine similarities use direct dot products (O(n·d), no similarity matrix). A new group starts strictly below the threshold.
  - Packing then runs within groups; those chunks end `'semantic'`.
  - Zero norms, non-finite similarities and missing or mismatched boundary adapters fail explicitly.
  - Every boundary piece counts against `maxPlanningSteps` before any request is issued.
  - A `deadline` also bounds pending boundary requests: one that never returns fails `DEADLINE_EXCEEDED` and its signal is aborted.
  - The boundary identities, preparation and budget are part of the pipeline digest.

Overlap is fixed at 0 and aggregation at `mean-chunk-vector/1`. `fullEncoderIdentityProblems`, `budgetedEncoderProblems` and `vectorProblems` are the shared encoder identity and output checks.
