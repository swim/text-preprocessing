/**
 * Canonical identities: sorted-key JSON of finite values, UTF-8, SHA-256 (FIPS 180-4, synchronous and
 * dependency-free so validation works on any runtime and can be called synchronously by loaders).
 */

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n));

/** Lowercase hex SHA-256 of exact bytes. */
export function sha256Hex(data: Uint8Array): string {
  const blocks = Math.ceil((data.length + 9) / 64);
  const msg = new Uint8Array(blocks * 64);
  msg.set(data);
  msg[data.length] = 0x80;
  const view = new DataView(msg.buffer);
  const bits = data.length * 8;
  view.setUint32(msg.length - 8, Math.floor(bits / 0x100000000));
  view.setUint32(msg.length - 4, bits >>> 0);
  const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const W = new Uint32Array(64);
  for (let b = 0; b < blocks; b++) {
    for (let t = 0; t < 16; t++) W[t] = view.getUint32(b * 64 + t * 4);
    for (let t = 16; t < 64; t++) {
      const s0 = rotr(W[t - 15], 7) ^ rotr(W[t - 15], 18) ^ (W[t - 15] >>> 3);
      const s1 = rotr(W[t - 2], 17) ^ rotr(W[t - 2], 19) ^ (W[t - 2] >>> 10);
      W[t] = (W[t - 16] + s0 + W[t - 7] + s1) >>> 0;
    }
    let [a, bb, c, d, e, f, g, h] = H;
    for (let t = 0; t < 64; t++) {
      const t1 = (h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[t] + W[t]) >>> 0;
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c))) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = bb; bb = a; a = (t1 + t2) >>> 0;
    }
    H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + bb) >>> 0; H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
    H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0; H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
  }
  return Array.from(H, (x) => x.toString(16).padStart(8, '0')).join('');
}

const utf8 = new TextEncoder();

/**
 * Sorted-key JSON of plain objects, arrays, strings, finite numbers, booleans and null. Arrays keep their
 * order; undefined object fields are dropped (callers normalise defaults into explicit fields before
 * hashing). Anything else (a Date, Map, typed array or class instance) throws rather than hashing as {}.
 */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('canonicalJson: non-finite number');
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (typeof value === 'object') {
    const proto = Object.getPrototypeOf(value);
    if (proto !== Object.prototype && proto !== null) throw new Error(`canonicalJson: cannot encode a ${proto?.constructor?.name ?? 'non-plain'} object (plain objects and arrays only)`);
    const o = value as Record<string, unknown>;
    return `{${Object.keys(o).sort().filter((k) => o[k] !== undefined).map((k) => `${JSON.stringify(k)}:${canonicalJson(o[k])}`).join(',')}}`;
  }
  throw new Error(`canonicalJson: cannot encode ${typeof value}`);
}

/** SHA-256 of a value's canonical JSON. */
export const canonicalDigest = (value: unknown): string => sha256Hex(utf8.encode(canonicalJson(value)));

/**
 * Identity of an exact JS string: SHA-256 over its UTF-16 code units (little-endian) under a domain
 * prefix. Unlike UTF-8, this is injective for malformed strings: TextEncoder maps every lone
 * surrogate to U+FFFD, so two different inputs would otherwise share a digest.
 */
export function sourceDigest(text: string): string {
  const prefix = utf8.encode('liquidau-source/utf16le/1\0');
  const bytes = new Uint8Array(prefix.length + text.length * 2);
  bytes.set(prefix);
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    bytes[prefix.length + 2 * i] = c & 0xff;
    bytes[prefix.length + 2 * i + 1] = c >> 8;
  }
  return sha256Hex(bytes);
}

export const isSha256Hex = (v: unknown): v is string => typeof v === 'string' && /^[0-9a-f]{64}$/.test(v);
