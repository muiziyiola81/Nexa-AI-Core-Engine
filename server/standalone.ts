/**
 * NEXA AI Core Engine - Standalone Headless API Server
 * Starts the pure Express backend without Vite or frontend dependencies.
 * Optimized for headless deployment, Termux (Android), Docker, and edge nodes.
 */

process.env.STANDALONE_API = 'true';

// Dynamically import full server with standalone flag enabled
import('../server.js').catch(err => {
  console.error('[NEXA STANDALONE] Failed to bootstrap standalone server:', err);
  process.exit(1);
});
