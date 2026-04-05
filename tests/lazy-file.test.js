import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { LazyFile } from '../src/ollama/lazy-file.js';

// ─── LazyFile constructor ───────────────────────────────────────────

describe('LazyFile constructor', () => {
  it('sets name, size, and internal path', () => {
    const lf = new LazyFile('/some/path', 'test-file', 1024);
    assert.equal(lf.name, 'test-file');
    assert.equal(lf.size, 1024);
    assert.equal(lf._path, '/some/path');
  });

  it('size is a number', () => {
    const lf = new LazyFile('/p', 'n', 999);
    assert.equal(typeof lf.size, 'number');
  });
});

// ─── LazyFile.slice ─────────────────────────────────────────────────

describe('LazyFile.slice', () => {
  let originalFetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('returns an object with arrayBuffer method', () => {
    const lf = new LazyFile('/path/to/blob', 'test', 1000);
    const slice = lf.slice(0, 100);
    assert.equal(typeof slice.arrayBuffer, 'function');
  });

  it('arrayBuffer() calls fetch with correct Range header', async () => {
    let capturedUrl, capturedOpts;
    globalThis.fetch = async (url, opts) => {
      capturedUrl = url;
      capturedOpts = opts;
      return {
        ok: true,
        arrayBuffer: async () => new ArrayBuffer(100),
      };
    };

    const lf = new LazyFile('/data/.ollama/blobs/sha256-abc', 'test', 5000);
    await lf.slice(100, 200).arrayBuffer();

    assert.ok(capturedUrl.includes(encodeURIComponent('/data/.ollama/blobs/sha256-abc')));
    assert.equal(capturedOpts.headers.Range, 'bytes=100-199');
  });

  it('arrayBuffer() throws on non-ok response', async () => {
    globalThis.fetch = async () => ({ ok: false, status: 404 });
    const lf = new LazyFile('/bad/path', 'test', 1000);
    await assert.rejects(() => lf.slice(0, 10).arrayBuffer(), /Failed to read bytes/);
  });

  it('builds correct URL with special characters in path', async () => {
    let capturedUrl;
    globalThis.fetch = async (url) => {
      capturedUrl = url;
      return { ok: true, arrayBuffer: async () => new ArrayBuffer(1) };
    };

    const lf = new LazyFile('/path with spaces/blob', 'test', 100);
    await lf.slice(0, 1).arrayBuffer();
    assert.ok(capturedUrl.includes('path%20with%20spaces'));
  });
});

// ─── LazyFile.create ────────────────────────────────────────────────

describe('LazyFile.create', () => {
  let originalFetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  // Helper: successful probe response (Range GET bytes=0-3)
  const probeOk = { ok: true, status: 206, headers: new Map([['Content-Range', 'bytes 0-3/815310432']]) };

  it('creates LazyFile from HEAD response Content-Length', async () => {
    globalThis.fetch = async (url, opts) => {
      if (opts && opts.method === 'HEAD') {
        return { ok: true, headers: new Map([['Content-Length', '815310432']]) };
      }
      // Probe Range request
      return probeOk;
    };

    const lf = await LazyFile.create('/some/blob', 'ollama:gemma3:1b');
    assert.equal(lf.size, 815310432);
    assert.equal(lf.name, 'ollama:gemma3:1b');
    assert.equal(lf._path, '/some/blob');
  });

  it('falls back to probe Content-Range when HEAD fails', async () => {
    let fetchCalls = 0;
    globalThis.fetch = async (url, opts) => {
      fetchCalls++;
      if (opts && opts.method === 'HEAD') {
        return { ok: false };
      }
      // Probe Range request succeeds with Content-Range
      return { ok: true, status: 206, headers: new Map([['Content-Range', 'bytes 0-3/500000']]) };
    };

    const lf = await LazyFile.create('/blob', 'test');
    assert.equal(fetchCalls, 2); // probe + HEAD (size from probe Content-Range)
    assert.equal(lf.size, 500000);
  });

  it('throws when probe returns 403 (permission denied)', async () => {
    globalThis.fetch = async () => ({
      ok: false, status: 403,
      text: async () => 'Permission denied: grant Full Disk Access',
    });
    await assert.rejects(
      () => LazyFile.create('/bad', 'test'),
      /Permission denied/,
    );
  });

  it('throws when both HEAD and Range fail', async () => {
    globalThis.fetch = async (url, opts) => {
      if (opts && opts.method === 'HEAD') return { ok: false };
      // Probe succeeds but with no useful Content-Range
      return { ok: true, status: 206, headers: new Map() };
    };
    await assert.rejects(
      () => LazyFile.create('/bad', 'test'),
      /Cannot determine size/,
    );
  });

  it('throws when size cannot be determined from HEAD', async () => {
    globalThis.fetch = async (url, opts) => {
      if (opts && opts.method === 'HEAD') {
        return { ok: true, headers: new Map([['Content-Length', '0']]) };
      }
      // Probe succeeds but with no Content-Range
      return { ok: true, status: 206, headers: new Map() };
    };
    await assert.rejects(
      () => LazyFile.create('/blob', 'test'),
      /Cannot determine size/,
    );
  });
});

