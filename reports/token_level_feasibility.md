# Token-Level Aggregation Feasibility Report

| Model / Backend | Feasible | Next-token scores API | Step-by-step | Shared sequence | Limitations |
|---|---|---|---|---|---|
| DirectVLMModel (Qwen2.5-VL) | yes | HF forward logits (model(**inputs).logits) and generate(output_scores=True) | yes | yes | Requires custom loop outside current predict(); no cached KV optimization in first prototype. |
| GRITModel (Qwen2.5-VL backbone) | yes | Same HF Qwen2.5-VL logits access as DirectVLMModel. | yes | yes | Current GRIT wrapper post-processes tags; token-level prototype should target short A/B/C/D answer generation first. |
| VisualCoTModel (LLaVA / VisCoT wrapper) | unclear | Underlying model likely has logits, but wrapper uses llava helpers and 2-turn crop pipeline. | unclear | unclear | Two-turn bbox->crop->answer flow complicates synchronized token loop; backend-specific refactor needed. |
| DeepEyesV2Model (agentic tool-calling) | no | Agentic loop expects tool-calls and turn-level parsing rather than pure next-token sync. | n/a | n/a | Not in current comparison target and incompatible with first token-level MVP. |
