# Solar-Open2 support (fork)

This fork of ggml-org/llama.cpp adds inference support for the
Upstage Solar-Open2-250B architecture (`solar_open2`).

## What this adds
- New arch `solar-open2`: 48-layer hybrid (36 Kimi Delta Attention linear-attn
  layers + 12 gated-GQA NoPE layers) with a 320-expert DeepSeek-V3-style MoE
  (shared expert + sigmoid router + e_score correction bias).
- HF-to-GGUF converter class `SolarOpen2Model`.
- Model type `LLM_TYPE_250B_A15B`.

## Attribution
Built on ggml-org/llama.cpp (MIT). The `solar-open2` graph reuses established
llama.cpp patterns:
- KDA recurrent branch adapted from `src/models/kimi-linear.cpp`
- MoE routing from the DeepSeek-V3 implementation
- gated-attention pattern from `src/models/qwen3next.cpp`

## Validation
- 8/8 greedy top-1 logit parity vs Upstage `transformers` reference.
- Coherent generation on the full 250B (Q2_K).
- Q2_K perplexity (wikitext-2, c=512, 80 chunks): PPL = 5.9322 +/- 0.10022.

## GGUFs
Prebuilt quants: https://huggingface.co/vcruz305/Solar-Open2-250B-GGUF

## Running (single DGX Spark GB10, 128GB unified)
Use --no-mmap for large quants (default mmap double-allocates and can OOM):

    llama-cli -m Solar-Open2-250B-Q2_K.gguf --no-mmap -ngl 999 -p "..."
