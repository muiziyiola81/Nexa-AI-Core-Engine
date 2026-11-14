/**
 * NEXA AI Core Engine - API Key Store & Authentication Manager
 * Secure server-side API key generation, timing-safe SHA-256 verification,
 * Cloud Run environment-injected key support, scoping, and usage tracking.
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { getDataDir } from './config.js';

export type KeyScope = 'chat:completions' | 'models:read' | 'models:manage' | 'engine:admin';

export interface ApiKeyRecord {
  id: string;
  name: string;
  keyPrefix: string;          // e.g. "nexa_live_9a7b..."
  keyHash: string;            // SHA-256 hash of secret key
  scopes: KeyScope[];
  createdAt: number;
  lastUsedAt?: number;
  totalRequests: number;
  rateLimitPerMin?: number;
}

const ALL_SCOPES: KeyScope[] = ['chat:completions', 'models:read', 'models:manage', 'engine:admin'];

export class KeyStore {
  private keys: Map<string, ApiKeyRecord> = new Map();
  private envKeys: Map<string, ApiKeyRecord> = new Map();
  private storageFile: string;

  constructor() {
    this.storageFile = path.join(getDataDir(), 'api_keys.json');
    this.ensureDataDir();
    this.load();
    this.loadEnvKeys();
    this.ensureDefaultKey();
  }

  private ensureDataDir() {
    const dataDir = getDataDir();
    if (!fs.existsSync(dataDir)) {
      try {
        fs.mkdirSync(dataDir, { recursive: true });
      } catch {
        // ignore if read-only container fs
      }
    }
  }

  private loadEnvKeys() {
    const candidates: Array<{ label: string; raw: string }> = [];
    if (process.env.NEXA_API_KEY && process.env.NEXA_API_KEY.trim()) {
      candidates.push({ label: 'NEXA_API_KEY (Cloud Run Secret)', raw: process.env.NEXA_API_KEY.trim() });
    }
    if (process.env.MASTER_API_KEY && process.env.MASTER_API_KEY.trim()) {
      candidates.push({ label: 'MASTER_API_KEY (Env)', raw: process.env.MASTER_API_KEY.trim() });
    }
    if (process.env.NEXA_API_KEYS && process.env.NEXA_API_KEYS.trim()) {
      const items = process.env.NEXA_API_KEYS.split(',').map(s => s.trim()).filter(Boolean);
      items.forEach((raw, idx) => {
        candidates.push({ label: `NEXA_API_KEYS[${idx}] (Env)`, raw });
      });
    }

    candidates.forEach((entry, idx) => {
      const hash = this.hashKey(entry.raw);
      const id = `key_env_${idx}_${hash.slice(0, 8)}`;
      const prefix =
        entry.raw.length >= 12
          ? `${entry.raw.slice(0, 10)}...${entry.raw.slice(-4)}`
          : 'env_key...';
      this.envKeys.set(id, {
        id,
        name: entry.label,
        keyPrefix: prefix,
        keyHash: hash,
        scopes: [...ALL_SCOPES],
        createdAt: Date.now(),
        totalRequests: 0
      });
    });
  }

  private load() {
    try {
      if (fs.existsSync(this.storageFile)) {
        const raw = fs.readFileSync(this.storageFile, 'utf-8');
        const list = JSON.parse(raw) as ApiKeyRecord[];
        for (const item of list) {
          this.keys.set(item.id, item);
        }
      }
    } catch {
      // ignore
    }
  }

  private save() {
    try {
      const list = Array.from(this.keys.values());
      fs.writeFileSync(this.storageFile, JSON.stringify(list, null, 2), 'utf-8');
    } catch (err) {
      console.error('Failed to save API keys:', err);
    }
  }

  private ensureDefaultKey() {
    // Do not persist a predictable default key in production when env keys are configured
    if (this.keys.size === 0 && this.envKeys.size === 0 && process.env.NODE_ENV !== 'production') {
      const initialKey = 'nexa_live_dev_master_key_local';
      const hash = this.hashKey(initialKey);
      const record: ApiKeyRecord = {
        id: 'key_default_master',
        name: 'Default Master Key (Local Dev)',
        keyPrefix: 'nexa_live_dev...',
        keyHash: hash,
        scopes: [...ALL_SCOPES],
        createdAt: Date.now(),
        totalRequests: 0
      };
      this.keys.set(record.id, record);
      this.save();
    }
  }

  public hashKey(rawKey: string): string {
    return crypto.createHash('sha256').update(rawKey).digest('hex');
  }

  private hashesEqual(hexA: string, hexB: string): boolean {
    if (hexA.length !== hexB.length) return false;
    const bufA = Buffer.from(hexA, 'hex');
    const bufB = Buffer.from(hexB, 'hex');
    if (bufA.length !== bufB.length) return false;
    return crypto.timingSafeEqual(bufA, bufB);
  }

  public hasConfiguredKeys(): boolean {
    return this.envKeys.size > 0 || this.keys.size > 0;
  }

  /**
   * Generates a new cryptographically secure API key.
   * Returns the raw key ONLY once.
   */
  public createKey(name: string, scopes: KeyScope[], rateLimitPerMin?: number): { record: ApiKeyRecord; rawKey: string } {
    const randomHex = crypto.randomBytes(24).toString('hex');
    const rawKey = `nexa_live_${randomHex}`;
    const keyHash = this.hashKey(rawKey);
    const keyPrefix = `${rawKey.slice(0, 14)}...${rawKey.slice(-4)}`;

    const record: ApiKeyRecord = {
      id: `key_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`,
      name,
      keyPrefix,
      keyHash,
      scopes,
      createdAt: Date.now(),
      totalRequests: 0,
      rateLimitPerMin
    };

    this.keys.set(record.id, record);
    this.save();
    return { record, rawKey };
  }

  public validateKey(rawKey: string, requiredScope?: KeyScope): { valid: boolean; record?: ApiKeyRecord; reason?: string } {
    if (!rawKey) {
      return { valid: false, reason: 'Empty API key.' };
    }

    const hash = this.hashKey(rawKey);

    // 1. Check environment-injected keys first (NEXA_API_KEY, MASTER_API_KEY, NEXA_API_KEYS)
    for (const record of this.envKeys.values()) {
      if (this.hashesEqual(record.keyHash, hash)) {
        if (requiredScope && !record.scopes.includes(requiredScope) && !record.scopes.includes('engine:admin')) {
          return { valid: false, record, reason: `API Key missing required permission: "${requiredScope}".` };
        }
        record.totalRequests++;
        record.lastUsedAt = Date.now();
        return { valid: true, record };
      }
    }

    // 2. Check persisted keys
    for (const record of this.keys.values()) {
      if (this.hashesEqual(record.keyHash, hash)) {
        if (requiredScope && !record.scopes.includes(requiredScope) && !record.scopes.includes('engine:admin')) {
          return { valid: false, record, reason: `API Key missing required permission: "${requiredScope}".` };
        }
        record.totalRequests++;
        record.lastUsedAt = Date.now();
        this.save();
        return { valid: true, record };
      }
    }

    return { valid: false, reason: 'Invalid or revoked API key.' };
  }

  public getAll(): Omit<ApiKeyRecord, 'keyHash'>[] {
    const combined = [...this.envKeys.values(), ...this.keys.values()];
    return combined.map(k => ({
      id: k.id,
      name: k.name,
      keyPrefix: k.keyPrefix,
      scopes: k.scopes,
      createdAt: k.createdAt,
      lastUsedAt: k.lastUsedAt,
      totalRequests: k.totalRequests,
      rateLimitPerMin: k.rateLimitPerMin
    }));
  }

  public revokeKey(id: string): boolean {
    if (!this.keys.has(id)) return false;
    this.keys.delete(id);
    this.save();
    return true;
  }
}
