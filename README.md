# RADiFF: Anatomy-Block Masked Diffusion for Structured Radiology Report Generation

RADiFF generates the structured Findings section of a chest X-ray report one anatomical
section at a time, using masked discrete diffusion within each section and a single shared
denoiser for all eight SRRG-Findings categories.

- **Project page:** https://maa1604.github.io/RADiFF/
- **Model (Hugging Face):** https://huggingface.co/maa1604/radiff

## Repository layout

| Path | Contents |
|---|---|
| `inference.py`, `radiff/`, `config.json` | Inference package, as released with the model |
| `training/train.py` | Training |
| `training/test.py` | Generation on SRRG-Findings splits |
| `training/tune_empty_thresholds.py` | Per-category emission threshold calibration |
| `training/eval_srrg.py` | StructEval scoring |
| `training/src/` | Data, model, diffusion and decoding modules used by training |

## Usage

Download `model.safetensors` and `tokenizer.json` from the Hugging Face model page (gated,
research use only), place them next to `inference.py`, then:

```bash
pip install -r requirements.txt
python inference.py --images frontal.jpg lateral.jpg
```

For training, set `RRG_TOK_DIR` to the directory containing `tokenizer.json`.

## License

Research-only, non-commercial. See `LICENSE`.
