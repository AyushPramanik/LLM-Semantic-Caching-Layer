# Grafana Dashboards

`semantic-cache-overview.json` is auto-provisioned into Grafana (folder
**Semantic Cache**) by the Docker Compose stack. It is also importable manually:
**Dashboards → Import → Upload JSON**, then select the Prometheus datasource.

## Panels

| # | Panel | What it answers |
|---|-------|-----------------|
| 1 | Real-time Hit Rate | What fraction of requests are served from cache right now? |
| 2 | Estimated Cost Saved (USD) | Cumulative dollars avoided via cache hits |
| 3 | Tokens Saved | Tokens not regenerated thanks to the cache |
| 4 | Cache Growth | How fast is the cache filling up? |
| 5 | Cache Efficiency Over Time | Hit-rate trend / warmup convergence |
| 6 | Cache Lookup Latency | P50 / P95 / P99 of cache lookups |
| 7 | Provider Latency (P95) | Upstream latency we avoid on a hit |
| 8 | Similarity Score Distribution | Where do match scores cluster vs. the threshold? |
| 9 | Requests by Provider | Traffic split across OpenAI / Anthropic / Ollama |
| 10 | Requests by Model | Traffic split across models |

## Screenshots

Add captured PNGs to `docs/img/` and reference them from the top-level README
(`grafana-overview.png`).
