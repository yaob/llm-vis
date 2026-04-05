/**
 * Integration tests for server.js
 *
 * Spawns a dedicated test server on a random port to avoid conflicts.
 * Tests the /api/local-file proxy endpoint and static file serving.
 *
 * Blob-specific tests query Ollama to discover a real blob path.
 * If Ollama is unavailable those tests are skipped gracefully.
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const PORT = 19876; // unlikely to conflict
const BASE = `http://127.0.0.1:${PORT}`;
const PROJECT_ROOT = resolve(fileURLToPath(import.meta.url), '../..');

let serverProcess;
let blobPath = null;
let blobReadable = false;

async function discoverBlobPath() {
  try {
    const res = await fetch('http://localhost:11434/api/tags', { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return null;
    const data = await res.json();
    const models = data.models || [];
    if (models.length === 0) return null;
    const showRes = await fetch('http://localhost:11434/api/show', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: models[0].name }),
      signal: AbortSignal.timeout(5000),
    });
    if (!showRes.ok) return null;
    const showData = await showRes.json();
    const mf = showData.modelfile || '';
    const match = mf.match(/^FROM\s+(\/[^\s]+sha256-[0-9a-f]{64})/mi);
    return match ? match[1] : null;
  } catch { return null; }
}

describe('server.js /api/local-file proxy', () => {
  before(async () => {
    // Spawn our custom server.js on the test port
    serverProcess = spawn('node', ['server.js', String(PORT)], {
      cwd: PROJECT_ROOT,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Wait for "running at" message
    await new Promise((ok, fail) => {
      const timeout = setTimeout(() => fail(new Error('Server start timeout')), 8000);
      let output = '';
      serverProcess.stdout.on('data', (d) => {
        output += d.toString();
        if (output.includes('running at')) { clearTimeout(timeout); ok(); }
      });
      serverProcess.stderr.on('data', (d) => { output += d.toString(); });
      serverProcess.on('error', (e) => { clearTimeout(timeout); fail(e); });
      serverProcess.on('exit', (code) => {
        if (!output.includes('running at')) {
          clearTimeout(timeout);
          fail(new Error(`Server exited with code ${code}: ${output}`));
        }
      });
    });

    blobPath = await discoverBlobPath();

    // Check if blob is actually readable (not just stat-able)
    if (blobPath) {
      try {
        const probe = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent(blobPath)}`, {
          headers: { Range: 'bytes=0-3' },
          signal: AbortSignal.timeout(5000),
        });
        if (probe.ok || probe.status === 206) {
          const buf = await probe.arrayBuffer();
          blobReadable = buf.byteLength === 4;
        }
      } catch { blobReadable = false; }
    }
  });

  after(async () => {
    if (serverProcess) {
      serverProcess.kill('SIGTERM');
      await new Promise(r => serverProcess.on('close', r));
    }
  });

  // --- Blob endpoint tests (require Ollama + a real blob) ---

  it('HEAD returns Content-Length and Accept-Ranges for a real blob', async (t) => {
    if (!blobPath) return t.skip('No Ollama blob available');
    const res = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent(blobPath)}`, { method: 'HEAD' });
    assert.equal(res.status, 200);
    const len = parseInt(res.headers.get('Content-Length'), 10);
    assert.ok(len > 0, `Expected positive Content-Length, got ${len}`);
    assert.equal(res.headers.get('Accept-Ranges'), 'bytes');
  });

  it('GET with Range returns 206 with GGUF magic bytes', async (t) => {
    if (!blobReadable) return t.skip('No readable Ollama blob available');
    const res = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent(blobPath)}`, {
      headers: { Range: 'bytes=0-3' },
    });
    assert.equal(res.status, 206);
    const buf = await res.arrayBuffer();
    const magic = new Uint8Array(buf);
    assert.equal(magic[0], 0x47); // G
    assert.equal(magic[1], 0x47); // G
    assert.equal(magic[2], 0x55); // U
    assert.equal(magic[3], 0x46); // F
    assert.ok(res.headers.get('Content-Range').startsWith('bytes 0-3/'));
    assert.equal(res.headers.get('Content-Length'), '4');
  });

  it('Range request for middle of file returns correct byte count', async (t) => {
    if (!blobReadable) return t.skip('No readable Ollama blob available');
    const res = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent(blobPath)}`, {
      headers: { Range: 'bytes=100-199' },
    });
    assert.equal(res.status, 206);
    const buf = await res.arrayBuffer();
    assert.equal(buf.byteLength, 100);
  });

  it('includes CORS headers on blob response', async (t) => {
    if (!blobPath) return t.skip('No Ollama blob available');
    const res = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent(blobPath)}`, {
      headers: { Range: 'bytes=0-0' },
    });
    assert.equal(res.headers.get('Access-Control-Allow-Origin'), '*');
  });

  it('OPTIONS returns 204 (preflight)', async (t) => {
    if (!blobPath) return t.skip('No Ollama blob available');
    const res = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent(blobPath)}`, { method: 'OPTIONS' });
    assert.equal(res.status, 204);
  });

  // --- Error cases (don't require Ollama) ---

  it('returns 400 when path parameter is missing', async () => {
    const res = await fetch(`${BASE}/api/local-file`);
    assert.equal(res.status, 400);
  });

  it('returns 403 for paths outside ~/.ollama', async () => {
    const res = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent('/etc/passwd')}`);
    assert.equal(res.status, 403);
  });

  it('returns 403 for path traversal attempt', async () => {
    const malicious = join(homedir(), '.ollama', '..', '..', 'etc', 'passwd');
    const res = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent(malicious)}`);
    assert.equal(res.status, 403);
  });

  it('returns 403 for /tmp path', async () => {
    const res = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent('/tmp/some-file')}`);
    assert.equal(res.status, 403);
  });

  it('returns 404 for non-existent file under ~/.ollama', async () => {
    const fakePath = join(homedir(), '.ollama', 'nonexistent-blob-xyz');
    const res = await fetch(`${BASE}/api/local-file?path=${encodeURIComponent(fakePath)}`);
    assert.equal(res.status, 404);
  });

  // --- Static file serving ---

  it('serves index.html at root', async () => {
    const res = await fetch(`${BASE}/`);
    assert.equal(res.status, 200);
    assert.ok(res.headers.get('Content-Type').includes('text/html'));
  });

  it('serves JS files with correct MIME type', async () => {
    const res = await fetch(`${BASE}/src/ollama/client.js`);
    assert.equal(res.status, 200);
    assert.ok(res.headers.get('Content-Type').includes('javascript'));
  });

  it('returns 404 for non-existent static file', async () => {
    const res = await fetch(`${BASE}/does-not-exist.xyz`);
    assert.equal(res.status, 404);
  });

  it('sets Cache-Control: no-cache on static files', async () => {
    const res = await fetch(`${BASE}/src/ollama/client.js`);
    assert.equal(res.headers.get('Cache-Control'), 'no-cache');
  });
});

