# DF3 Evaluation Report (Generated)

This file is auto-generated from `results/*.jsonl` by `scripts/summarize_results.py`.
It is intended as a *reproducible* backbone for a paper-quality narrative report.

## How to Reproduce

Run:

```bash
python scripts/summarize_results.py --results-dir results --out artifacts/eval_summary.json --out-md docs/evaluation_report.generated.md
```

The full machine-readable output is `artifacts/eval_summary.json`.

## Inventory

| Agent Model | Vision Model | Mode | JSONL | n | Acc | Cov | Acc(ans) | Abstain | Errors |
|:---:|---|---|---|---|---|---|---|---|---|
| anthropic/claude-sonnet-4.5 | anthropic/claude-sonnet-4.5 | tools | B_or_sonnet45.jsonl | 200 | 0.475 | 0.600 | 0.792 | 0.400 | 0.000 |
| deepseek/deepseek-v3.2 | google/gemini-3-flash-preview | tools | G3vision_agent_deepseekv32.jsonl | 200 | 0.470 | 0.605 | 0.777 | 0.390 | 0.005 |
| google/gemini-3-flash-preview | google/gemini-3-flash-preview | no-tools | G3visiononly_notools.jsonl | 500 | 0.914 | 0.980 | 0.933 | 0.018 | 0.002 |
| google/gemini-3-flash-preview | google/gemini-3-flash-preview | tools | B_or_g3flashprev.jsonl | 500 | 0.782 | 0.934 | 0.837 | 0.062 | 0.004 |
| gpt-5-mini | gpt-5.2 | no-tools | A_openai_gpt5mini_v52.jsonl | 500 | 0.074 | 0.078 | 0.949 | 0.922 | 0.000 |
| gpt-5-mini | gpt-5.2 | tools | A_openai_gpt5mini_v52.jsonl | 500 | 0.240 | 0.294 | 0.816 | 0.700 | 0.006 |
| gpt-5.2 | gpt-5.2 | no-tools | A_openai_gpt52_visiononly.jsonl | 500 | 0.074 | 0.078 | 0.949 | 0.922 | 0.000 |
| gpt-5.2 | gpt-5.2 | tools | A_openai_gpt52_e2e_tools.jsonl | 500 | 0.080 | 0.112 | 0.714 | 0.888 | 0.000 |
| moonshotai/kimi-k2-thinking:nitro | google/gemini-3-flash-preview | tools | G3vision_agent_kimi_k2.jsonl | 200 | 0.295 | 0.355 | 0.831 | 0.630 | 0.015 |
| moonshotai/kimi-k2-thinking:nitro | z-ai/glm-4.6v:nitro | tools | Kimi_k2_del.jsonl | 500 | 0.416 | 0.484 | 0.860 | 0.510 | 0.006 |
| x-ai/grok-4.1-fast | google/gemini-3-flash-preview | tools | G3vision_agent_grok41fast.jsonl | 200 | 0.505 | 0.605 | 0.835 | 0.390 | 0.005 |
| xiaomi/mimo-v2-flash:free | google/gemini-3-flash-preview | tools | G3vision_agent_mimov2flash.jsonl | 200 | 0.650 | 0.860 | 0.756 | 0.140 | 0.000 |
| xiaomi/mimo-v2-flash:free | z-ai/glm-4.6v:nitro | tools | Mimo_del.jsonl | 500 | 0.190 | 0.596 | 0.319 | 0.402 | 0.002 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | no-tools | Glm46v-vision_agent_glm46.jsonl | 500 | 0.418 | 0.636 | 0.657 | 0.364 | 0.000 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | tools | Glm46v-vision_agent_glm46.jsonl | 500 | 0.448 | 0.750 | 0.597 | 0.232 | 0.018 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6v:nitro | tools | Glm_del.jsonl | 500 | 0.504 | 0.600 | 0.840 | 0.398 | 0.002 |
| z-ai/glm-4.7:nitro | google/gemini-3-flash-preview | no-tools | G3vision_agent_glm47.jsonl | 500 | 0.914 | 0.980 | 0.933 | 0.020 | 0.000 |
| z-ai/glm-4.7:nitro | google/gemini-3-flash-preview | tools | G3vision_agent_glm47.jsonl | 500 | 0.456 | 0.508 | 0.898 | 0.490 | 0.002 |

## Key Metrics (per artifact/config)

_Note: `acc` and `cov` are computed over all samples; `acc_ans`/`mcc_ans`/`bal_acc_ans` are computed on answered samples only (abstentions removed). `fake_slip` and `real_false_flag` are triage-style rates over all items in the class._

| Artifact | Agent Model | Vision Model | Vision Src | Dataset | n | Acc | Acc 95% CI | Cov | Cov 95% CI | Acc(ans) | MCC(ans) | BalAcc(ans) | FakeSlip | RealFalseFlag | Abstain | Lat p50(s) | Lat p95(s) | ECE | Brier | AURC |
|:---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-sonnet-4.5\|tools\|B_or_sonnet45.jsonl | anthropic/claude-sonnet-4.5 | anthropic/claude-sonnet-4.5 | agent_rule_sonnet | 1f78e35118013ed4 | 200 | 0.475 | [0.407,0.544] | 0.600 | [0.531,0.665] | 0.792 | 0.578 | 0.791 | 0.112 | 0.137 | 0.400 | 54.979 | 70.626 | 0.067 | 0.161 | 0.168 |
| deepseek/deepseek-v3.2\|tools\|G3vision_agent_deepseekv32.jsonl | deepseek/deepseek-v3.2 | google/gemini-3-flash-preview | filename_g3_prefix | 1f78e35118013ed4 | 200 | 0.470 | [0.402,0.539] | 0.605 | [0.536,0.670] | 0.777 | 0.555 | 0.772 | 0.184 | 0.088 | 0.390 | 59.138 | 289.982 | 0.032 | 0.168 | 0.162 |
| google/gemini-3-flash-preview\|no-tools\|G3visiononly_notools.jsonl | google/gemini-3-flash-preview | google/gemini-3-flash-preview | models_field | f987165daff0de70 | 500 | 0.914 | [0.886,0.936] | 0.980 | [0.964,0.989] | 0.933 | 0.870 | 0.931 | 0.121 | 0.012 | 0.018 | 6.200 | 8.036 | 0.031 | 0.057 | 0.022 |
| google/gemini-3-flash-preview\|tools\|B_or_g3flashprev.jsonl | google/gemini-3-flash-preview | google/gemini-3-flash-preview | filename_b_or_g3 | f987165daff0de70 | 500 | 0.782 | [0.744,0.816] | 0.934 | [0.909,0.953] | 0.837 | 0.674 | 0.837 | 0.166 | 0.138 | 0.062 | 33.320 | 47.370 | 0.041 | 0.134 | 0.096 |
| gpt-5-mini\|no-tools\|A_openai_gpt5mini_v52.jsonl | gpt-5-mini | gpt-5.2 | agent_rule_openai | f987165daff0de70 | 500 | 0.074 | [0.054,0.100] | 0.078 | [0.058,0.105] | 0.949 | 0.000 | 0.500 | 0.000 | 0.008 | 0.922 | 0.027 | 0.038 | 0.143 | 0.071 | 0.042 |
| gpt-5-mini\|tools\|A_openai_gpt5mini_v52.jsonl | gpt-5-mini | gpt-5.2 | agent_rule_openai | f987165daff0de70 | 500 | 0.240 | [0.205,0.279] | 0.294 | [0.256,0.335] | 0.816 | 0.659 | 0.806 | 0.008 | 0.099 | 0.700 | 88.935 | 128.741 | 0.160 | 0.173 | 0.238 |
| gpt-5.2\|no-tools\|A_openai_gpt52_visiononly.jsonl | gpt-5.2 | gpt-5.2 | agent_rule_openai | f987165daff0de70 | 500 | 0.074 | [0.054,0.100] | 0.078 | [0.058,0.105] | 0.949 | 0.000 | 0.500 | 0.000 | 0.008 | 0.922 | 0.037 | 0.048 | 0.143 | 0.071 | 0.042 |
| gpt-5.2\|tools\|A_openai_gpt52_e2e_tools.jsonl | gpt-5.2 | gpt-5.2 | agent_rule_openai | f987165daff0de70 | 500 | 0.080 | [0.059,0.107] | 0.112 | [0.087,0.143] | 0.714 | 0.332 | 0.579 | 0.000 | 0.063 | 0.888 | 35.935 | 57.543 | 0.060 | 0.214 | 0.299 |
| moonshotai/kimi-k2-thinking:nitro\|tools\|G3vision_agent_kimi_k2.jsonl | moonshotai/kimi-k2-thinking:nitro | google/gemini-3-flash-preview | filename_g3_prefix | 1f78e35118013ed4 | 200 | 0.295 | [0.236,0.362] | 0.355 | [0.292,0.423] | 0.831 | 0.652 | 0.829 | 0.051 | 0.069 | 0.630 | 32.391 | 52.559 | 0.044 | 0.144 | 0.144 |
| moonshotai/kimi-k2-thinking:nitro\|tools\|Kimi_k2_del.jsonl | moonshotai/kimi-k2-thinking:nitro | z-ai/glm-4.6v:nitro | models_field | c02071eee1ee544a | 500 | 0.416 | [0.374,0.460] | 0.484 | [0.440,0.528] | 0.860 | -0.005 | 0.498 | 0.020 | 0.446 | 0.510 | 36.205 | 63.647 | 0.024 | 0.121 | 0.112 |
| x-ai/grok-4.1-fast\|tools\|G3vision_agent_grok41fast.jsonl | x-ai/grok-4.1-fast | google/gemini-3-flash-preview | filename_g3_prefix | 1f78e35118013ed4 | 200 | 0.505 | [0.436,0.574] | 0.605 | [0.536,0.670] | 0.835 | 0.611 | 0.801 | 0.112 | 0.088 | 0.390 | 40.101 | 55.719 | 0.061 | 0.136 | 0.138 |
| xiaomi/mimo-v2-flash:free\|tools\|G3vision_agent_mimov2flash.jsonl | xiaomi/mimo-v2-flash:free | google/gemini-3-flash-preview | filename_g3_prefix | 1f78e35118013ed4 | 200 | 0.650 | [0.582,0.713] | 0.860 | [0.805,0.901] | 0.756 | 0.558 | 0.742 | 0.408 | 0.020 | 0.140 | 44.373 | 72.020 | 0.082 | 0.180 | 0.146 |
| xiaomi/mimo-v2-flash:free\|tools\|Mimo_del.jsonl | xiaomi/mimo-v2-flash:free | z-ai/glm-4.6v:nitro | models_field | c02071eee1ee544a | 500 | 0.190 | [0.158,0.227] | 0.596 | [0.552,0.638] | 0.319 | -0.050 | 0.463 | 0.432 | 0.196 | 0.402 | 32.273 | 52.175 | 0.521 | 0.478 | 0.554 |
| z-ai/glm-4.6:nitro\|no-tools\|Glm46v-vision_agent_glm46.jsonl | z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | filename_glm46_vision | f987165daff0de70 | 500 | 0.418 | [0.376,0.462] | 0.636 | [0.593,0.677] | 0.657 | 0.222 | 0.538 | 0.441 | 0.000 | 0.364 | 0.012 | 0.026 | 0.199 | 0.263 | 0.304 |
| z-ai/glm-4.6:nitro\|tools\|Glm46v-vision_agent_glm46.jsonl | z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | filename_glm46_vision | f987165daff0de70 | 500 | 0.448 | [0.405,0.492] | 0.750 | [0.710,0.786] | 0.597 | 0.142 | 0.533 | 0.587 | 0.024 | 0.232 | 32.253 | 49.041 | 0.215 | 0.283 | 0.322 |
| z-ai/glm-4.6:nitro\|tools\|Glm_del.jsonl | z-ai/glm-4.6:nitro | z-ai/glm-4.6v:nitro | models_field | c02071eee1ee544a | 500 | 0.504 | [0.460,0.548] | 0.600 | [0.556,0.642] | 0.840 | 0.084 | 0.543 | 0.056 | 0.411 | 0.398 | 28.984 | 43.728 | 0.050 | 0.136 | 0.100 |
| z-ai/glm-4.7:nitro\|no-tools\|G3vision_agent_glm47.jsonl | z-ai/glm-4.7:nitro | google/gemini-3-flash-preview | filename_g3_prefix | f987165daff0de70 | 500 | 0.914 | [0.886,0.936] | 0.980 | [0.964,0.989] | 0.933 | 0.869 | 0.931 | 0.117 | 0.016 | 0.020 | 0.029 | 0.039 | 0.031 | 0.057 | 0.026 |
| z-ai/glm-4.7:nitro\|tools\|G3vision_agent_glm47.jsonl | z-ai/glm-4.7:nitro | google/gemini-3-flash-preview | filename_g3_prefix | f987165daff0de70 | 500 | 0.456 | [0.413,0.500] | 0.508 | [0.464,0.552] | 0.898 | 0.793 | 0.899 | 0.040 | 0.063 | 0.490 | 34.514 | 55.146 | 0.156 | 0.117 | 0.095 |

## Rankings by Vision Model (within dataset)

### Dataset 1f78e35118013ed4

| Vision Model | Agent Model | Mode | Acc | Cov | Acc(ans) | Abstain |
|:---:|---|---|---|---|---|---|
| anthropic/claude-sonnet-4.5 | anthropic/claude-sonnet-4.5 | tools | 0.475 | 0.600 | 0.792 | 0.400 |
| google/gemini-3-flash-preview | xiaomi/mimo-v2-flash:free | tools | 0.650 | 0.860 | 0.756 | 0.140 |
| google/gemini-3-flash-preview | x-ai/grok-4.1-fast | tools | 0.505 | 0.605 | 0.835 | 0.390 |
| google/gemini-3-flash-preview | deepseek/deepseek-v3.2 | tools | 0.470 | 0.605 | 0.777 | 0.390 |
| google/gemini-3-flash-preview | moonshotai/kimi-k2-thinking:nitro | tools | 0.295 | 0.355 | 0.831 | 0.630 |

### Dataset c02071eee1ee544a

| Vision Model | Agent Model | Mode | Acc | Cov | Acc(ans) | Abstain |
|:---:|---|---|---|---|---|---|
| z-ai/glm-4.6v:nitro | z-ai/glm-4.6:nitro | tools | 0.504 | 0.600 | 0.840 | 0.398 |
| z-ai/glm-4.6v:nitro | moonshotai/kimi-k2-thinking:nitro | tools | 0.416 | 0.484 | 0.860 | 0.510 |
| z-ai/glm-4.6v:nitro | xiaomi/mimo-v2-flash:free | tools | 0.190 | 0.596 | 0.319 | 0.402 |

### Dataset f987165daff0de70

| Vision Model | Agent Model | Mode | Acc | Cov | Acc(ans) | Abstain |
|:---:|---|---|---|---|---|---|
| google/gemini-3-flash-preview | google/gemini-3-flash-preview | no-tools | 0.914 | 0.980 | 0.933 | 0.018 |
| google/gemini-3-flash-preview | z-ai/glm-4.7:nitro | no-tools | 0.914 | 0.980 | 0.933 | 0.020 |
| google/gemini-3-flash-preview | google/gemini-3-flash-preview | tools | 0.782 | 0.934 | 0.837 | 0.062 |
| google/gemini-3-flash-preview | z-ai/glm-4.7:nitro | tools | 0.456 | 0.508 | 0.898 | 0.490 |
| gpt-5.2 | gpt-5-mini | tools | 0.240 | 0.294 | 0.816 | 0.700 |
| gpt-5.2 | gpt-5.2 | tools | 0.080 | 0.112 | 0.714 | 0.888 |
| gpt-5.2 | gpt-5-mini | no-tools | 0.074 | 0.078 | 0.949 | 0.922 |
| gpt-5.2 | gpt-5.2 | no-tools | 0.074 | 0.078 | 0.949 | 0.922 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | tools | 0.448 | 0.750 | 0.597 | 0.232 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | no-tools | 0.418 | 0.636 | 0.657 | 0.364 |

## Tool Usage Analysis (All Tools Runs; Descriptive)

_Note: Conditional statistics only; tool selection is confounded by image difficulty and is not causal._

| Tool | Calls | Used Rate | Acc(ans) Used | Acc(ans) Not Used | DeltaAcc(ans) | Conf Used | Conf Not Used | DeltaConf |
|:---:|---|---|---|---|---|---|---|---|
| perform_trufor | 3695 | 0.739 | 0.833 | 0.564 | 0.269 | 0.809 | 0.832 | -0.024 |
| perform_ela | 2985 | 0.597 | 0.832 | 0.663 | 0.169 | 0.788 | 0.842 | -0.054 |
| metadata | 2983 | 0.597 | 0.842 | 0.622 | 0.220 | 0.812 | 0.822 | -0.010 |
| extract_residuals | 1505 | 0.301 | 0.840 | 0.703 | 0.137 | 0.816 | 0.817 | -0.001 |
| execute_python_code | 333 | 0.067 | 0.796 | 0.737 | 0.060 | 0.858 | 0.812 | 0.046 |
| analyze_frequency_domain | 277 | 0.055 | 0.774 | 0.741 | 0.033 | 0.794 | 0.818 | -0.024 |
| detect_jpeg_quantization | 45 | 0.009 | 0.800 | 0.742 | 0.058 | 0.730 | 0.817 | -0.087 |
| analyze_jpeg_compression | 17 | 0.003 | 0.667 | 0.743 | -0.076 | 0.815 | 0.817 | -0.002 |

## Paired Comparisons (McNemar exact; tools vs no-tools)

| Model | n_common | both_correct | vision_correct_tools_wrong | vision_wrong_tools_correct | both_wrong | McNemar p (2-sided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| z-ai/glm-4.7:nitro | 500 | 219 | 238 | 9 | 34 | 7.48e-59 |
| gpt-5-mini | 500 | 19 | 18 | 101 | 362 | 3.37e-15 |
| google/gemini-3-flash-preview | 500 | 370 | 87 | 21 | 22 | 9.9e-11 |
| z-ai/glm-4.6:nitro | 500 | 173 | 36 | 51 | 240 | 0.133 |
| gpt-5.2 | 500 | 11 | 26 | 29 | 434 | 0.788 |

## Rankings (Vision-only / no-tools)

| Agent Model | Vision Model | Mode | Acc | Cov | Acc(ans) | Abstain | Vision<250ms |
|:---:|---|---|---|---|---|---|---|
| google/gemini-3-flash-preview | google/gemini-3-flash-preview | no-tools | 0.914 | 0.980 | 0.933 | 0.018 | 0.000 |
| z-ai/glm-4.7:nitro | google/gemini-3-flash-preview | no-tools | 0.914 | 0.980 | 0.933 | 0.020 | 1.000 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | no-tools | 0.418 | 0.636 | 0.657 | 0.364 | 0.986 |
| gpt-5-mini | gpt-5.2 | no-tools | 0.074 | 0.078 | 0.949 | 0.922 | 0.996 |
| gpt-5.2 | gpt-5.2 | no-tools | 0.074 | 0.078 | 0.949 | 0.922 | 1.000 |

## Rankings (Tool-augmented / tools)

| Agent Model | Vision Model | Mode | Acc | Cov | Acc(ans) | Abstain | Vision<250ms |
|:---:|---|---|---|---|---|---|---|
| google/gemini-3-flash-preview | google/gemini-3-flash-preview | tools | 0.782 | 0.934 | 0.837 | 0.062 | 0.000 |
| xiaomi/mimo-v2-flash:free | google/gemini-3-flash-preview | tools | 0.650 | 0.860 | 0.756 | 0.140 | 1.000 |
| x-ai/grok-4.1-fast | google/gemini-3-flash-preview | tools | 0.505 | 0.605 | 0.835 | 0.390 | 1.000 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6v:nitro | tools | 0.504 | 0.600 | 0.840 | 0.398 | 0.992 |
| anthropic/claude-sonnet-4.5 | anthropic/claude-sonnet-4.5 | tools | 0.475 | 0.600 | 0.792 | 0.400 | 0.000 |
| deepseek/deepseek-v3.2 | google/gemini-3-flash-preview | tools | 0.470 | 0.605 | 0.777 | 0.390 | 1.000 |
| z-ai/glm-4.7:nitro | google/gemini-3-flash-preview | tools | 0.456 | 0.508 | 0.898 | 0.490 | 0.998 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | tools | 0.448 | 0.750 | 0.597 | 0.232 | 0.000 |
| moonshotai/kimi-k2-thinking:nitro | z-ai/glm-4.6v:nitro | tools | 0.416 | 0.484 | 0.860 | 0.510 | 1.000 |
| moonshotai/kimi-k2-thinking:nitro | google/gemini-3-flash-preview | tools | 0.295 | 0.355 | 0.831 | 0.630 | 1.000 |
| gpt-5-mini | gpt-5.2 | tools | 0.240 | 0.294 | 0.816 | 0.700 | 1.000 |
| xiaomi/mimo-v2-flash:free | z-ai/glm-4.6v:nitro | tools | 0.190 | 0.596 | 0.319 | 0.402 | 0.808 |
| gpt-5.2 | gpt-5.2 | tools | 0.080 | 0.112 | 0.714 | 0.888 | 1.000 |
