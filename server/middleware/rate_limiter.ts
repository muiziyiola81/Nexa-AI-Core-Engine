/**
 * NEXA AI Core Engine - Token Bucket Rate Limiter
 * Per-Key and Per-IP rate limiting enforcing RPM limits with standard HTTP 429 response.
 * Extracts real client IP behind Cloud Run / reverse proxies via X-Forwarded-For.
 */

import { Response, NextFunction } from 'express';
import { AuthenticatedRequest } from './auth.js';
import { ConfigStore } from '../store/config.js';

interface ClientWindow {
  count: number;
  resetAt: number;
}

function getClientIp(req: AuthenticatedRequest): string {
  const xff = req.headers['x-forwarded-for'];
  if (typeof xff === 'string' && xff.trim()) {
    return xff.split(',')[0].trim();
  }
  if (Array.isArray(xff) && xff.length > 0) {
    return xff[0].trim();
  }
  return req.ip || 'global_client';
}

export function createRateLimiter(configStore: ConfigStore) {
  const windows: Map<string, ClientWindow> = new Map();

  // Periodic cleanup (.unref() ensures CLI test scripts terminate cleanly)
  const cleanupTimer = setInterval(() => {
    const now = Date.now();
    for (const [key, win] of windows.entries()) {
      if (now > win.resetAt) {
        windows.delete(key);
      }
    }
  }, 30000);
  cleanupTimer.unref?.();

  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    const config = configStore.get();
    const identifier = req.apiKeyRecord?.id || getClientIp(req);
    const limit = req.apiKeyRecord?.rateLimitPerMin || config.rateLimitPerMinute;

    const now = Date.now();
    let clientWin = windows.get(identifier);

    if (!clientWin || now > clientWin.resetAt) {
      clientWin = {
        count: 0,
        resetAt: now + 60000 // 1 minute window
      };
      windows.set(identifier, clientWin);
    }

    clientWin.count++;
    const remaining = Math.max(0, limit - clientWin.count);
    const resetSeconds = Math.ceil((clientWin.resetAt - now) / 1000);

    res.setHeader('X-RateLimit-Limit', limit.toString());
    res.setHeader('X-RateLimit-Remaining', remaining.toString());
    res.setHeader('X-RateLimit-Reset', resetSeconds.toString());

    if (clientWin.count > limit) {
      res.setHeader('Retry-After', resetSeconds.toString());
      res.status(429).json({
        error: {
          message: `Rate limit reached for requests. Limit: ${limit} req/min. Please retry in ${resetSeconds}s.`,
          type: 'requests',
          param: null,
          code: 'rate_limit_exceeded'
        }
      });
      return;
    }

    next();
  };
}
