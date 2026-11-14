/**
 * NEXA AI Core Engine - Central Error Handler
 * Formats all exceptions into standard OpenAI-compatible API error envelopes.
 */

import { Request, Response, NextFunction } from 'express';

export function errorHandler(err: unknown, _req: Request, res: Response, _next: NextFunction): void {
  const error = err as Error;
  const message = error.message || 'An internal error occurred in the NEXA Core Engine.';

  let statusCode = 500;
  let code = 'internal_server_error';
  let type = 'server_error';

  if (message.includes('[NEXA_ERR_MODEL_NOT_FOUND]')) {
    statusCode = 404;
    code = 'model_not_found';
    type = 'invalid_request_error';
  } else if (message.includes('[NEXA_ERR_MODEL_NOT_LOADED]')) {
    statusCode = 400;
    code = 'model_not_loaded';
    type = 'invalid_request_error';
  } else if (message.includes('[NEXA_ERR_MAX_CONCURRENCY]')) {
    statusCode = 503;
    code = 'server_overloaded';
    type = 'server_error';
  } else if (message.includes('[NEXA_ERR_MODEL_LOAD_FAILED]')) {
    statusCode = 503;
    code = 'model_load_failed';
    type = 'server_error';
  } else if (message.includes('Hardware Load Error')) {
    statusCode = 400;
    code = 'hardware_resource_missing';
    type = 'invalid_request_error';
  }

  res.status(statusCode).json({
    error: {
      message: message.replace(/\[NEXA_ERR_[A-Z_]+\]\s*/, ''),
      type,
      param: null,
      code
    }
  });
}
