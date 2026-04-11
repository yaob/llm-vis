/**
 * Ollama API Client
 * Communicates with a local Ollama instance to list and inspect downloaded models.
 * Default endpoint: http://localhost:11434
 */

const DEFAULT_BASE = 'http://localhost:11434';

export class OllamaClient {
  constructor(baseUrl = DEFAULT_BASE) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  /** Check if Ollama is reachable. */
  async isAvailable() {
    try {
      const res = await fetch(`${this.baseUrl}/api/version`, { signal: AbortSignal.timeout(2000) });
      return res.ok;
    } catch {
      return false;
    }
  }

  /**
   * List locally available models.
   * Returns array of { name, size, family, parameterSize, quantizationLevel, modifiedAt, digest }
   */
  async listModels() {
    const res = await fetch(`${this.baseUrl}/api/tags`);
    if (!res.ok) throw new Error(`Ollama /api/tags failed: ${res.status}`);
    const data = await res.json();
    return (data.models || []).map(m => ({
      name: m.name,
      model: m.model,
      size: m.size,
      digest: m.digest,
      modifiedAt: m.modified_at,
      family: m.details?.family || 'unknown',
      families: m.details?.families || [],
      parameterSize: m.details?.parameter_size || '',
      quantizationLevel: m.details?.quantization_level || '',
      format: m.details?.format || '',
    }));
  }

  /**
   * Get detailed model info including GGUF metadata KV pairs.
   * @param {string} modelName - e.g. "llama3.2:latest"
   * @returns {{ details, modelInfo, template, parameters, license, modelfile }}
   */
  async showModel(modelName) {
    const res = await fetch(`${this.baseUrl}/api/show`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelName }),
    });
    if (!res.ok) throw new Error(`Ollama /api/show failed: ${res.status}`);
    const data = await res.json();
    return {
      details: data.details || {},
      modelInfo: data.model_info || {},
      template: data.template || '',
      parameters: data.parameters || '',
      license: data.license || '',
      capabilities: data.capabilities || [],
      modelfile: data.modelfile || '',
    };
  }

  /**
   * Extract the local GGUF blob file path from the modelfile's FROM line.
   * Ollama's /api/show returns a modelfile with lines like:
   *   FROM /path/to/blobs/sha256-<hex>          (Unix)
   *   FROM C:\path\to\blobs\sha256-<hex>        (Windows backslash)
   *   FROM C:/path/to/blobs/sha256-<hex>        (Windows forward slash)
   * @param {string} modelfile
   * @returns {string|null} absolute file path, or null
   */
  static extractBlobPath(modelfile) {
    if (!modelfile) return null;
    // Match Unix absolute paths (/...) and Windows absolute paths (C:\... or C:/...)
    const match = modelfile.match(/^FROM\s+((?:[A-Za-z]:[\/\\]|\/)[^\s]+sha256-[0-9a-f]{64})/mi);
    if (!match) return null;
    return match[1];
  }

  /**
   * Get the local blob file path for a model.
   * @param {string} modelName
   * @param {object} [showData] - pre-fetched showModel response
   * @returns {Promise<string|null>}
   */
  async getBlobPath(modelName, showData = null) {
    try {
      const data = showData || await this.showModel(modelName);
      return OllamaClient.extractBlobPath(data.modelfile);
    } catch {
      return null;
    }
  }
}

