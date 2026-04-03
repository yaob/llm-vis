# LLM Visualizer

A browser-based tool for inspecting and visualizing the internal structure of Large Language Models. Drag-and-drop a GGUF file or connect to a local Ollama instance and instantly explore the model's architecture — layers, tensors, quantization, weight heatmaps, and dataflow diagrams. Everything runs client-side; no server, no uploads, no API keys.

## Features

- **GGUF file parsing** — reads the binary header, metadata, and tensor index without loading weights into memory, so it works with multi-gigabyte models
- **Ollama integration** — auto-detects locally running Ollama and lists installed models for one-click inspection
- **Architecture-aware residual flow diagrams** — interactive SVG dataflow visualizations for Attention, MLP, MoE (Mixture of Experts), and SSM (Mamba) blocks
- **Weight heatmaps** — dequantizes and renders tensor slices as color-mapped heatmaps with paginated row navigation (supports F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K)
- **Model summary card** — name, architecture, params, memory, blocks, embedding dim, heads, context length, vocab size, GQA ratio, head dimension
- **Quantization profile** — visual distribution bar showing the percentage of each quant type across all tensors
- **RoPE parameters** — frequency base, scaling type, and dimension count
- **Tokenizer info** — model type (BPE/SPM), special token IDs (BOS/EOS/PAD), chat template presence
- **FFN details** — hidden dimension, expansion ratio, activation function (SwiGLU, GeGLU, ReGLU)
- **Norm type detection** — RMSNorm vs LayerNorm from architecture mapping
- **Attention variants** — sliding window, cross-attention, layer norm epsilon
- **Provenance metadata** — author, license, URL, description
- **Raw metadata browser** — expandable table of all GGUF key-value pairs

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) 18+ (22+ recommended)

### Install & Run

```bash
git clone https://github.com/yaob/llm-vis.git
cd llm-vis
npm install
npm start
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

### Load a Model

**Option A — GGUF file upload**: Drag-and-drop a `.gguf` file onto the upload area (or click to browse). Only the header is read; the full file is not loaded into memory.

**Option B — Ollama**: If [Ollama](https://ollama.com/) is running locally on port 11434, installed models appear automatically. Click any model card to inspect it. (Weight heatmaps require a GGUF file upload.)

## Running Tests

Tests use the Node.js built-in test runner — no extra dependencies needed:

```bash
node --test tests/*.test.js
```

A GitHub Actions workflow runs these automatically on every pull request to `main`.

## Project Structure

```
index.html                      Entry point and UI shell
src/
  parsers/
    gguf.js                     GGUF binary parser (header + metadata + tensor info)
  model/
    analyzer.js                 Group tensors → normalized model tree
    gguf-tensor-decoder.js      Dequantize tensor slices for heatmaps
    residual-flow-graph.js      Build residual dataflow graph from model
    ollama-adapter.js           Adapt Ollama /api/show response to model format
  ollama/
    client.js                   Ollama HTTP API client
  viz/
    app-view.js                 Top-level view: summary card + view tabs
    residual-flow.js            SVG residual flow renderer (attention, MLP, MoE, SSM)
    layer-stack.js              Vertical layer stack visualization
tests/
  analyzer.test.js              Tensor name parsing, value coercion, component aliases
  gguf.test.js                  Tensor byte size calculations
  tensor-decoder.test.js        FP16/BF16 conversion, stats, row decoders
.github/workflows/test.yml      CI workflow
```

## License

MIT