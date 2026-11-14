/**
 * NEXA AI Core Engine - Request Validation Middleware
 * Strict schema validation for OpenAI chat completions and model registration.
 */

import { Request, Response, NextFunction } from 'express';

export function validateChatCompletionRequest(req: Request, res: Response, next: NextFunction): void {
  const { model, messages, temperature, top_p, max_tokens } = req.body;

  if (!model || typeof model !== 'string') {
    res.status(400).json({
      error: {
        message: 'Missing or invalid "model" parameter. Must be a valid model ID string.',
        type: 'invalid_request_error',
        param: 'model',
        code: 'param_required'
      }
    });
    return;
  }

  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({
      error: {
        message: 'Missing or empty "messages" parameter. Must be an array of message objects.',
        type: 'invalid_request_error',
        param: 'messages',
        code: 'param_required'
      }
    });
    return;
  }

  const validRoles = ['system', 'user', 'assistant', 'tool'];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (!msg || typeof msg !== 'object' || !validRoles.includes(msg.role)) {
      res.status(400).json({
        error: {
          message: `Invalid role "${msg?.role}" in message at index ${i}. Must be one of: ${validRoles.join(', ')}.`,
          type: 'invalid_request_error',
          param: `messages[${i}].role`,
          code: 'invalid_value'
        }
      });
      return;
    }

    if (typeof msg.content !== 'string') {
      res.status(400).json({
        error: {
          message: `Missing or non-string "content" in message at index ${i}.`,
          type: 'invalid_request_error',
          param: `messages[${i}].content`,
          code: 'invalid_value'
        }
      });
      return;
    }
  }

  if (temperature !== undefined) {
    if (typeof temperature !== 'number' || temperature < 0 || temperature > 2.0) {
      res.status(400).json({
        error: {
          message: 'Invalid "temperature" parameter. Must be a number between 0.0 and 2.0.',
          type: 'invalid_request_error',
          param: 'temperature',
          code: 'invalid_value'
        }
      });
      return;
    }
  }

  if (top_p !== undefined) {
    if (typeof top_p !== 'number' || top_p < 0 || top_p > 1.0) {
      res.status(400).json({
        error: {
          message: 'Invalid "top_p" parameter. Must be a number between 0.0 and 1.0.',
          type: 'invalid_request_error',
          param: 'top_p',
          code: 'invalid_value'
        }
      });
      return;
    }
  }

  if (max_tokens !== undefined) {
    if (!Number.isInteger(max_tokens) || max_tokens < 1) {
      res.status(400).json({
        error: {
          message: 'Invalid "max_tokens" parameter. Must be a positive integer.',
          type: 'invalid_request_error',
          param: 'max_tokens',
          code: 'invalid_value'
        }
      });
      return;
    }
  }

  next();
}

export function validateModelRegistration(req: Request, res: Response, next: NextFunction): void {
  const { id, name, architecture, weightPath, contextWindow } = req.body;

  if (!id || typeof id !== 'string' || !/^[a-zA-Z0-9._-]+$/.test(id)) {
    res.status(400).json({
      error: {
        message: 'Invalid model "id". Must be an alphanumeric slug containing only dashes, dots, or underscores.',
        type: 'invalid_request_error',
        param: 'id',
        code: 'invalid_value'
      }
    });
    return;
  }

  if (!name || typeof name !== 'string') {
    res.status(400).json({
      error: {
        message: 'Missing or invalid "name" string.',
        type: 'invalid_request_error',
        param: 'name',
        code: 'param_required'
      }
    });
    return;
  }

  if (!weightPath || typeof weightPath !== 'string') {
    res.status(400).json({
      error: {
        message: 'Missing or invalid "weightPath" string (e.g. models/llama-3-8b.Q4_K_M.gguf).',
        type: 'invalid_request_error',
        param: 'weightPath',
        code: 'param_required'
      }
    });
    return;
  }

  if (contextWindow !== undefined && (!Number.isInteger(contextWindow) || contextWindow < 512)) {
    res.status(400).json({
      error: {
        message: 'Invalid "contextWindow". Must be an integer >= 512.',
        type: 'invalid_request_error',
        param: 'contextWindow',
        code: 'invalid_value'
      }
    });
    return;
  }

  next();
}
