# BioCache adapter for TTT-LM JAX

This dependency-free package is a **state proxy**, not a complete VLM
integration. It deliberately separates:

- `persistent_state.dynamic_ttt_params`: per-sequence parameters updated by TTT
- `static_extra_state.static_model_params`: pretrained/static model parameters
- `temporary_state`: per-call scratch data

The upstream JAX layer performs its TTT scan internally and does not expose the
last dynamic parameters through the public model output. Supply an instrumented
`update_fn` or an explicit `TTTPersistentState` to capture them; this adapter
does not label the full Flax parameter tree as dynamic state.

Importing `biocache_adapt` does not import JAX or Flax. `jax_available` uses
module discovery only, preserving optional delayed import.

The common callable API is `reset()`, `update()`, `query()`,
`persistent_state()`, `static_extra_state()`, and `temporary_state()`. It also
includes recursive pytree byte accounting, timing, configurable budgets, a
manifest schema, and JSONL schemas for delayed recall and VNBench features.

No-model validation:

```bash
python -m biocache_adapt --dry-run --pretty
python -m biocache_adapt --help
```
