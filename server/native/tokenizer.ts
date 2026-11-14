/**
 * NEXA Native Runtime - Tokenizer Implementation (TypeScript)
 * Byte-level vocabulary tokenizer with UTF-8 byte fallback and reversible decoding.
 */

export interface NexaSpecialTokens {
  unkId: number;
  bosId: number;
  eosId: number;
  padId: number;
}

export class NexaTokenizer {
  private tokenToId: Map<string, number> = new Map();
  private idToToken: string[] = [];
  private idToBytes: Map<number, Uint8Array> = new Map();

  readonly special: NexaSpecialTokens = {
    unkId: 0,
    bosId: 1,
    eosId: 2,
    padId: 3
  };

  constructor() {
    this.addSpecial('<unk>', this.special.unkId);
    this.addSpecial('<|begin_of_text|>', this.special.bosId);
    this.addSpecial('<|end_of_text|>', this.special.eosId);
    this.addSpecial('<pad>', this.special.padId);

    // 256 individual raw bytes mapped to IDs 4..259
    for (let b = 0; b < 256; b++) {
      const id = 4 + b;
      const rawByteArr = new Uint8Array([b]);
      this.idToToken[id] = `<0x${b.toString(16).padStart(2, '0').toUpperCase()}>`;
      this.idToBytes.set(id, rawByteArr);
      // Also map 1-byte ASCII strings directly
      this.tokenToId.set(String.fromCharCode(b), id);
    }
  }

  private addSpecial(token: string, id: number) {
    if (id >= this.idToToken.length) {
      this.idToToken.length = id + 1;
    }
    this.idToToken[id] = token;
    this.tokenToId.set(token, id);
  }

  addToken(token: string, id: number): void {
    if (id >= this.idToToken.length) {
      this.idToToken.length = id + 1;
    }
    this.idToToken[id] = token;
    this.tokenToId.set(token, id);
    this.idToBytes.set(id, Buffer.from(token, 'utf-8'));
  }

  loadVocabulary(vocab: string[]): void {
    for (let i = 0; i < vocab.length; i++) {
      this.addToken(vocab[i], 260 + i);
    }
  }

  encode(text: string, addBos: boolean = true, addEos: boolean = false): number[] {
    const tokens: number[] = [];
    if (addBos) tokens.push(this.special.bosId);

    let i = 0;
    while (i < text.length) {
      let matched = false;
      const maxLen = Math.min(32, text.length - i);

      // Longest token match for vocabulary words
      for (let len = maxLen; len >= 2; len--) {
        const sub = text.substring(i, i + len);
        const id = this.tokenToId.get(sub);
        if (id !== undefined) {
          tokens.push(id);
          i += len;
          matched = true;
          break;
        }
      }

      if (!matched) {
        // Multi-byte UTF-8 character fallback
        const char = text[i];
        const charBuf = Buffer.from(char, 'utf-8');
        for (let b = 0; b < charBuf.length; b++) {
          const byteVal = charBuf[b];
          tokens.push(4 + byteVal);
        }
        i += 1;
      }
    }

    if (addEos) tokens.push(this.special.eosId);
    return tokens;
  }

  decode(tokens: number[], skipSpecial: boolean = true): string {
    const byteChunks: number[] = [];

    for (const id of tokens) {
      if (skipSpecial) {
        if (
          id === this.special.bosId ||
          id === this.special.eosId ||
          id === this.special.padId ||
          id === this.special.unkId
        ) {
          continue;
        }
      }

      // Check if it's a byte token (4..259)
      if (id >= 4 && id <= 259) {
        byteChunks.push(id - 4);
      } else {
        const bytes = this.idToBytes.get(id);
        if (bytes) {
          for (let b = 0; b < bytes.length; b++) {
            byteChunks.push(bytes[b]);
          }
        } else if (this.idToToken[id]) {
          const buf = Buffer.from(this.idToToken[id], 'utf-8');
          for (let b = 0; b < buf.length; b++) {
            byteChunks.push(buf[b]);
          }
        }
      }
    }

    return Buffer.from(byteChunks).toString('utf-8');
  }

  get vocabSize(): number {
    return this.idToToken.length;
  }
}
