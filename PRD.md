# LLM Visualizer — Product Requirements Document

## Overview

LLM Visualizer is a browser-based tool that lets you drag-and-drop a model file and instantly see its internal structure — layers, tensor shapes, parameter counts, quantization types, and architecture-aware dataflow diagrams. No server, no install, no dependencies. Everything runs client-side by parsing only the file header/metadata, so it works even with multi-gigabyte models.

## Goals

- **Instant model inspection**: Drop a file, see the full architecture in under 2 seconds.
- **Zero infrastructure**: Runs entirely in the browser. No backend, no uploads, no API keys.
- **Multi-format**: Support the most common LLM distribution formats (GGUF, SafeTensors, GGML).
- **Deep introspection**: Go beyond layer lists — show weight distributions, quantization coverage, and architecture-aware dataflow.
- **Shareable**: Export a model summary as a portable snapshot (image or self-contained HTML).

## Non-Goals

- **Model inference or execution**: This is a read-only inspection tool.
- **Model editing or conversion**: We don't modify files.
- **Training or fine-tuning workflows**: Out of scope.
- **Model comparison / diffing**: Not in current scope.
- **Server-side processing**: All parsing happens in the browser.

## Target Users

| User | Need |
|---|---|
| **ML engineers** | Quickly verify a model's architecture, layer count, and quantization before deploying to production. |
| **Hobbyists running local LLMs** | Understand what's inside a GGUF file downloaded from HuggingFace before loading it into llama.cpp or Ollama. |
| **Researchers** | Inspect tensor shapes and parameter distributions across layers without writing Python scripts. |
| **Content creators / educators** | Generate clear visual diagrams of model architectures for articles, videos, and presentations. |

## User Stories

1. **US-1**: As a user, I can drag-and-drop a `.gguf` file and see a summary card (model name, architecture, total params, block count, context length, vocab size) within 2 seconds.
2. **US-2**: As a user, I can see every transformer block as a visual block in a vertical stack, with parameter count shown proportionally.
3. **US-3**: As a user, I can click a block to expand it and see its sub-components (Attention, MLP, Norm) with their individual parameter counts.
4. **US-4**: As a user, I can click a sub-component to see individual tensors with their names, shapes, and quantization types.
5. **US-5**: As a user, I can upload a `.safetensors` file and get the same visualization experience.
6. **US-6**: As a user, I can see an architecture-aware dataflow diagram showing how data flows through embedding → attention → MLP → output.
7. **US-7**: As a user, I can see a heatmap of quantization types across all layers to understand which parts of the model are more or less compressed.
8. **US-8**: As a user, I can see weight distribution histograms for individual tensors (sampled from the file).
9. **US-9**: As a user, I can export the current visualization as a PNG image or a self-contained HTML file to share with others.

## Feature Roadmap

### P0 — Foundation (current)
> Ship a working GGUF visualizer with layer-level inspection.

- [x] GGUF binary parser (header + metadata + tensor info, no weights)
- [x] Model structure analyzer (group tensors by block, classify by component)
- [x] Vertical layer stack visualization with expandable blocks
- [x] Summary card with key model metadata
- [ ] **Ollama model picker**: Detect locally installed Ollama models (via `GET http://localhost:11434/api/tags`) and let users select one to visualize instead of drag-and-drop. Resolve the model's GGUF file path from Ollama's blob storage and load it directly.
- [ ] Handle edge cases: sharded GGUF files, corrupt headers, very large metadata
- [ ] Basic error states and loading feedback

### P1 — Multi-format & Deeper Inspection
> Support SafeTensors and add quantization/architecture visualizations.

- [ ] **SafeTensors parser**: Read JSON header from `.safetensors` files (US-5)
- [ ] **Config JSON parser**: Parse HuggingFace `config.json` for architecture info
- [ ] **Quantization heatmap**: Color-coded grid showing quant type per tensor across all layers (US-7)
- [ ] **Architecture dataflow diagram**: D3-rendered diagram showing data path through the transformer (US-6)
- [ ] **Parameter distribution chart**: Bar/treemap showing where parameters are concentrated (embedding vs attention vs MLP)

### P2 — Deep Introspection & Sharing
> Weight-level analysis and export capabilities.

- [ ] **Weight distribution histograms**: Sample actual weight values from the file and render per-tensor histograms (US-8)
- [ ] **Export to PNG/HTML**: Snapshot current view as a shareable artifact (US-9)
- [ ] **URL-based state**: Encode view state in URL hash so visualizations can be bookmarked
- [ ] **GGML format support**: Parse legacy GGML/GGMF/GGJT files
- [ ] **Keyboard navigation**: Arrow keys to move between layers, Enter to expand/collapse

## Technical Architecture

```
┌─────────────────────────────────────────────────┐
│                   Browser                        │
│                                                  │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐  │
│  │  File     │──▶│  Parsers   │──▶│ Analyzer  │  │
│  │  Input    │   │            │   │           │  │
│  │ (drag/    │   │ gguf.js    │   │ Groups    │  │
│  │  drop)    │   │ safetens.js│   │ tensors   │  │
│  │           │   │ config.js  │   │ into tree │  │
│  └──────────┘   └────────────┘   └─────┬─────┘  │
│                                        │         │
│                                        ▼         │
│                               ┌────────────────┐ │
│                               │ Visualizations │ │
│                               │                │ │
│                               │ layer-stack.js │ │
│                               │ dataflow.js    │ │
│                               │ quant-map.js   │ │
│                               │ histograms.js  │ │
│                               └────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Key design principle**: Parsers only read metadata/headers — never load full weight tensors into memory. This keeps the tool fast and usable for models of any size. Weight sampling (P2) will use targeted `File.slice()` reads.

### File Structure

```
index.html                  Entry point and UI shell
src/
  parsers/
    gguf.js                 Parse .gguf binary header
    safetensors.js          Parse .safetensors JSON header (P1)
    config-json.js          Parse HuggingFace config.json (P1)
  model/
    analyzer.js             Group tensors → normalized model tree
  viz/
    layer-stack.js          Vertical block diagram
    dataflow.js             Architecture-aware dataflow diagram (P1)
    quant-map.js            Quantization heatmap (P1)
    histograms.js           Weight distribution charts (P2)
```

## Technical Constraints

| Constraint | Detail |
|---|---|
| **No server** | Everything runs in the browser. No file uploads, no backend API. |
| **No framework** | Vanilla JS + D3.js. No React/Vue/Svelte. Keeps the tool zero-dependency and fast to load. |
| **Memory** | Must not load full weight data. Header-only parsing. For a 70B Q4 model (~40GB), we read at most ~100MB of header. |
| **Browser compat** | Must work in modern Chrome, Firefox, Safari. Uses ES modules, `File.slice()`, `DataView`, `ArrayBuffer`. |
| **File size** | No upper limit on file size — `File.slice()` ensures we only read what we need. |

## Open Questions

1. **Should we support remote files?** e.g. paste a HuggingFace URL and fetch just the header via HTTP Range requests. This would avoid downloading the full model.
2. **How deep should weight sampling go?** Reading actual weight values requires understanding each quantization format's block layout. Worth the complexity for P2?
3. **Should the architecture dataflow diagram be auto-generated from tensor names, or should we maintain per-architecture templates** (e.g. a LLaMA template, a Mamba template)?
4. **Do we need a plugin system?** If community members want to add parsers for new formats (e.g. ONNX, TensorFlow SavedModel), how do they extend the tool?
5. **PWA / offline support?** The tool already works offline since it's all client-side, but should we add a service worker for installability?

