/**
 * The versioned splitter registry. Each algorithm is fixed behaviour, never a configurable regex:
 *
 *   paragraph/1        separator runs /(?:\r\n|\r|\n)[\t ]*(?:(?:\r\n|\r|\n)[\t ]*)+/gu - two or more
 *                      line breaks, with spaces/tabs between and after them. A single line break stays
 *                      inside its paragraph. The run's trailing spaces/tabs (the next paragraph's
 *                      indentation) belong to the run.
 *   sentence-simple/1  separator runs /[.!?;\r\n]+/gu on the ORIGINAL text. Abbreviations and decimals
 *                      split ("3.5", "e.g."). A simple baseline: not a linguistic sentence detector, and
 *                      not rule-miner's canonicaliser (which splits after NFKC normalisation).
 *
 * For both: cut immediately after each separator run (the run belongs to the span on its left),
 * starting at offset 0, omitting zero-length spans and appending any nonempty remainder. A leading
 * separator run is its own span. Spans therefore partition the text exactly.
 */
import type { SourceSpan } from './spans.ts';

export const SPLITTERS = {
  'paragraph/1': () => /(?:\r\n|\r|\n)[\t ]*(?:(?:\r\n|\r|\n)[\t ]*)+/gu,
  'sentence-simple/1': () => /[.!?;\r\n]+/gu,
} as const;

export type SplitterAlgorithm = keyof typeof SPLITTERS;

export const SPLITTER_ALGORITHMS = Object.keys(SPLITTERS) as SplitterAlgorithm[];

/** Atomic spans of `text` under a registered splitter (empty for empty text). */
export function splitSpans(text: string, algorithm: SplitterAlgorithm): SourceSpan[] {
  if (!Object.hasOwn(SPLITTERS, algorithm)) throw new Error(`unknown splitter ${String(algorithm)}`);
  const make = SPLITTERS[algorithm];
  const re = make();
  const spans: SourceSpan[] = [];
  let start = 0;
  for (const m of text.matchAll(re)) {
    const end = m.index + m[0].length;
    if (end > start) spans.push({ start, end });
    start = end;
  }
  if (start < text.length) spans.push({ start, end: text.length });
  return spans;
}
