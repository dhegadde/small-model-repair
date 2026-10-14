# Third-party code, data and models

- The code here was written for this study inside a working copy of
  [llm-research-kit](https://github.com/vukrosic/llm-research-kit) (MIT License, Copyright (c) 2025 Vuk Rosić); no kit
  file is included.
- `results/` holds model outputs and grades only. Benchmarks are downloaded at run time and keep their own licenses:
  HumanEval+ / MBPP+ (EvalPlus), MultiPL-E, MMLU. Training prompts come from Magicoder-OSS-Instruct-75K and
  UltraChat-200k (downloaded at run time; not included).
- Models are downloaded from Hugging Face and not included: Qwen3-4B-Instruct-2507 and Qwen3-Coder-30B-A3B-Instruct
  (Apache 2.0); the GGUF files evaluated are Unsloth's public quantizations of those models.
