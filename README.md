# LLM-Semantic-Caching-Layer

A drop-in, OpenAI-compatible **semantic caching proxy** that reduces LLM API
cost and tail latency by serving cached responses for *semantically similar*
prompts.

Point your client at the proxy by changing a single setting:

```diff
- OPENAI_BASE_URL=https://api.openai.com/v1
+ OPENAI_BASE_URL=http://localhost:8000/v1
```

No application code changes are required.

> Full architecture, benchmarks, and deployment docs are added incrementally as
> the project matures. See [`docs/`](docs/).

## Status

🚧 Early development. Tracking the phased roadmap in [`docs/ROADMAP.md`](docs/ROADMAP.md).
