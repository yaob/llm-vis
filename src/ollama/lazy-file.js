/**
 * LazyFile — a File-like wrapper that reads bytes on demand via HTTP Range
 * requests to the dev server's /api/local-file endpoint.
 *
 * Implements the subset of the File/Blob API used by the GGUF parser and
 * tensor decoder: .name, .size, .slice(start, end).arrayBuffer().
 */
export class LazyFile {
  /**
   * @param {string} localPath - absolute file path on the server's filesystem
   * @param {string} name - display name (e.g. "ollama:gemma3:1b")
   * @param {number} size - total file size in bytes
   */
  constructor(localPath, name, size) {
    this._path = localPath;
    this.name = name;
    this.size = size;
  }

  /**
   * Create a LazyFile by probing the server for the file's size.
   * @param {string} localPath
   * @param {string} name
   * @returns {Promise<LazyFile>}
   */
  static async create(localPath, name) {
    const url = `/api/local-file?path=${encodeURIComponent(localPath)}`;
    // HEAD-like: fetch 0 bytes to get Content-Range with total size
    const res = await fetch(url, { method: 'HEAD' });
    if (!res.ok) {
      // Fallback: try a Range request to get total size from Content-Range
      const rangeRes = await fetch(url, { headers: { Range: 'bytes=0-0' } });
      if (!rangeRes.ok) throw new Error(`Cannot access blob file: ${localPath}`);
      const cr = rangeRes.headers.get('Content-Range') || '';
      const m = cr.match(/\/(\d+)$/);
      const size = m ? parseInt(m[1], 10) : 0;
      if (!size) throw new Error(`Cannot determine size of blob file: ${localPath}`);
      return new LazyFile(localPath, name, size);
    }
    // If HEAD is supported, use Content-Length
    const size = parseInt(res.headers.get('Content-Length') || '0', 10);
    if (!size) throw new Error(`Cannot determine size of blob file: ${localPath}`);
    return new LazyFile(localPath, name, size);
  }

  /**
   * Return a Blob-like slice that supports .arrayBuffer().
   * Compatible with how decodeTensorSlice calls file.slice(start, end).arrayBuffer().
   */
  slice(start, end) {
    const path = this._path;
    return {
      arrayBuffer() {
        const url = `/api/local-file?path=${encodeURIComponent(path)}`;
        return fetch(url, {
          headers: { Range: `bytes=${start}-${end - 1}` },
        }).then(res => {
          if (!res.ok) throw new Error(`Failed to read bytes ${start}-${end} from ${path}`);
          return res.arrayBuffer();
        });
      },
    };
  }
}

