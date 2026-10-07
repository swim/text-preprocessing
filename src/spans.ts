/**
 * Source spans: [start, end) in UTF-16 code units of the ORIGINAL JavaScript string - not bytes,
 * code-point indices or positions in a normalised string. Convert at external boundaries.
 */

export interface SourceSpan {
  /** Inclusive UTF-16 code-unit offset in the original string. */
  start: number;
  /** Exclusive UTF-16 code-unit offset. */
  end: number;
}

export interface TextDocument {
  id: string;
  text: string;
  /** The split-independence unit (conversation, author cluster, paraphrase group). */
  groupId: string;
  authorId?: string;
}

const isHigh = (c: number) => c >= 0xd800 && c <= 0xdbff;
const isLow = (c: number) => c >= 0xdc00 && c <= 0xdfff;

/** True when cutting `text` at `offset` doesn't split a valid surrogate pair (0 and length are boundaries). */
export function isCodePointBoundary(text: string, offset: number): boolean {
  if (!Number.isInteger(offset) || offset < 0 || offset > text.length) return false;
  if (offset === 0 || offset === text.length) return true;
  return !(isHigh(text.charCodeAt(offset - 1)) && isLow(text.charCodeAt(offset)));
}

/** True when the code point at `offset` is a combining mark (a cut there separates it from its base). */
export const startsWithMark = (text: string, offset: number): boolean => offset < text.length && /^\p{M}/u.test(text.slice(offset, offset + 2));

/**
 * Problems with spans as a partition of `text` (empty when they are ordered, nonempty, disjoint, on
 * code-point boundaries and cover [0, text.length) exactly).
 */
export function partitionProblems(text: string, spans: readonly SourceSpan[]): string[] {
  const p: string[] = [];
  let at = 0;
  spans.forEach((s, i) => {
    if (!s || !Number.isInteger(s.start) || !Number.isInteger(s.end)) { p.push(`span ${i} has non-integer offsets`); return; }
    if (s.start !== at) p.push(`span ${i} starts at ${s.start}, expected ${at}`);
    if (s.end <= s.start) p.push(`span ${i} is empty or reversed`);
    if (!isCodePointBoundary(text, s.start) || !isCodePointBoundary(text, s.end)) p.push(`span ${i} splits a surrogate pair`);
    at = s.end;
  });
  if (at !== text.length) p.push(`spans cover [0, ${at}), the text has ${text.length} code units`);
  return p;
}

/** The original text of a span. */
export const spanText = (text: string, span: SourceSpan): string => text.slice(span.start, span.end);
