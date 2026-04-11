/**
 * Dev server: static file serving + local file proxy for Ollama GGUF blobs.
 *
 * Serves the project directory as static files (like http-server) and adds a
 * special /api/local-file endpoint that reads a local file by path with Range
 * header support. This lets the browser load Ollama's on-disk GGUF blobs
 * without symlinking or copying.
 *
 * Usage: node server.js [port]
 */

import { createServer } from 'node:http';
import { open, readFile, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const PORT = parseInt(process.argv[2] || '8080', 10);
const ROOT = resolve(fileURLToPath(import.meta.url), '..');

const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Range');
  res.setHeader('Access-Control-Expose-Headers', 'Content-Range, Content-Length');
}

/** Serve a local file by path with Range support (for Ollama blobs). */
async function handleLocalFile(req, res) {
  cors(res);
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const url = new URL(req.url, `http://${req.headers.host}`);
  const filePath = url.searchParams.get('path');
  if (!filePath) { res.writeHead(400); res.end('Missing ?path='); return; }

  // Security: only allow files under ~/.ollama
  const home = process.env.HOME || process.env.USERPROFILE || '';
  const allowed = resolve(home, '.ollama');
  const resolved = resolve(filePath);
  // Windows paths are case-insensitive; normalise before comparing
  const norm = process.platform === 'win32'
    ? s => s.toLowerCase()
    : s => s;
  if (!norm(resolved).startsWith(norm(allowed))) {
    res.writeHead(403);
    res.end('Forbidden: only files under ~/.ollama are accessible');
    return;
  }

  let info;
  try { info = await stat(resolved); } catch { res.writeHead(404); res.end('Not found'); return; }
  if (!info.isFile()) { res.writeHead(400); res.end('Not a file'); return; }

  const total = info.size;

  // HEAD: return size only (stat already succeeded, no read needed)
  if (req.method === 'HEAD') {
    res.writeHead(200, {
      'Content-Type': 'application/octet-stream',
      'Content-Length': total,
      'Accept-Ranges': 'bytes',
    });
    res.end();
    return;
  }

  // Verify read access before sending success headers (macOS may EPERM on open)
  let fd;
  try {
    fd = await open(resolved, 'r');
  } catch (err) {
    const isPerm = err.code === 'EPERM' || err.code === 'EACCES';
    res.writeHead(isPerm ? 403 : 500);
    res.end(isPerm
      ? 'Permission denied: grant Full Disk Access to your terminal app in System Settings → Privacy & Security → Full Disk Access, then restart the server.'
      : `Cannot read file: ${err.message}`);
    return;
  } finally {
    if (fd) await fd.close();
  }

  const range = req.headers.range;

  if (range) {
    const m = range.match(/bytes=(\d+)-(\d*)/);
    if (!m) { res.writeHead(416); res.end(); return; }
    const start = parseInt(m[1], 10);
    const end = m[2] ? parseInt(m[2], 10) : total - 1;
    if (start >= total || end >= total) { res.writeHead(416); res.end(); return; }
    res.writeHead(206, {
      'Content-Type': 'application/octet-stream',
      'Content-Range': `bytes ${start}-${end}/${total}`,
      'Content-Length': end - start + 1,
      'Accept-Ranges': 'bytes',
    });
    const stream = createReadStream(resolved, { start, end });
    stream.on('error', (err) => {
      console.error('Stream error:', err.message);
      res.end();
    });
    stream.pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Type': 'application/octet-stream',
      'Content-Length': total,
      'Accept-Ranges': 'bytes',
    });
    const stream = createReadStream(resolved);
    stream.on('error', (err) => {
      console.error('Stream error:', err.message);
      res.end();
    });
    stream.pipe(res);
  }
}

/** Serve static files from the project root. */
async function handleStatic(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  let filePath = join(ROOT, decodeURIComponent(url.pathname));
  if (filePath.endsWith('/') || filePath.endsWith('\\')) filePath = join(filePath, 'index.html');

  const ext = extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';

  try {
    const data = await readFile(filePath);
    res.writeHead(200, {
      'Content-Type': mime,
      'Content-Length': data.length,
      'Cache-Control': 'no-cache',
    });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('Not found');
  }
}

const server = createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const handler = url.pathname === '/api/local-file' ? handleLocalFile : handleStatic;
  handler(req, res).catch((err) => {
    console.error('Request handler error:', err);
    if (!res.headersSent) { res.writeHead(500); res.end('Internal server error'); }
  });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`LLM Inspector running at http://127.0.0.1:${PORT}`);
  console.log(`Serving static files from ${ROOT}`);
  console.log(`Local file proxy at /api/local-file?path=...`);
});

