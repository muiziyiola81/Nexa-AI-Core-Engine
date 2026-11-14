/**
 * NEXA AI Core Engine - Server-Side Authentication Middleware
 * Enforces Bearer token verification and scope checking.
 */

import { Request, Response, NextFunction } from 'express';
import { KeyStore, KeyScope, ApiKeyRecord } from '../store/key_store.js';
import { ConfigStore } from '../store/config.js';

export interface AuthenticatedRequest extends Request {
  apiKeyRecord?: ApiKeyRecord;
}

export function createAuthMiddleware(keyStore: KeyStore, configStore: ConfigStore, requiredScope?: KeyScope) {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    const config = configStore.get();
    const authHeader = req.headers.authorization;

    // If API key is not strictly enforced globally AND no auth header is sent, allow request
    if (!config.requireApiKey && !authHeader) {
      return next();
    }

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({
        error: {
          message: 'You didn\'t provide an API key. You need to provide your API key in an Authorization header using Bearer auth (e.g. Authorization: Bearer nexa_live_...).',
          type: 'invalid_request_error',
          param: null,
          code: 'invalid_api_key'
        }
      });
      return;
    }

    const token = authHeader.slice(7).trim();
    const validation = keyStore.validateKey(token, requiredScope);

    if (!validation.valid) {
      const isScopeError = Boolean(validation.record && validation.reason?.includes('missing required permission'));
      res.status(isScopeError ? 403 : 401).json({
        error: {
          message: validation.reason || 'Incorrect API key provided.',
          type: 'invalid_request_error',
          param: null,
          code: isScopeError ? 'insufficient_scope' : 'invalid_api_key'
        }
      });
      return;
    }

    req.apiKeyRecord = validation.record;
    next();
  };
}
