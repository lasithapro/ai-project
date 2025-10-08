# ai-project

Minimal extractive Question Answering demo showcasing optional QLoRA (4-bit quantization + LoRA adapters) on a DistilBERT QA model.

## Features
* Loads `distilbert-base-uncased-distilled-squad` for extractive QA.
* Attempts 4-bit quantization via `bitsandbytes` if a CUDA GPU is available.
* Wraps the model with dummy LoRA adapters (untrained) when `peft` and GPU quantization are available.
* Falls back gracefully to a standard CPU fp32 model when quantization is not supported.

## Requirements
See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

If you are on a CPU-only environment the script will automatically skip 4-bit quantization.

## Run
Because `app.py` lives at the project root (not inside a package directory), invoke it directly:
```bash
python app.py
```

Optional: override the model via environment variable:
```bash
QA_MODEL=distilbert-base-uncased-distilled-squad python app.py
```

If you prefer `python -m app`, restructure like:
```
ai-project/
	app/__init__.py
	app/main.py  # rename current app.py here
```
and call `python -m app.main`.

## How QLoRA Fallback Works
1. Script checks for CUDA availability.
2. Tries to import `bitsandbytes`.
3. If both pass, builds a 4-bit `BitsAndBytesConfig` and loads the model in NF4.
4. If PEFT is available, attaches LoRA adapters to `q_lin` & `v_lin`.
5. Any failure -> plain CPU fp32 model (message printed).

## Troubleshooting
| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| `ModuleNotFoundError: bitsandbytes` | Not installed or unsupported platform | `pip install bitsandbytes` (GPU) or ignore (CPU fallback) |
| `No module named 'triton.ops'` | Missing/old `triton` for GPU kernels | `pip install --upgrade triton` |
| `ImportError ... load_in_8bit` | Missing `accelerate` | `pip install accelerate` |
| High RAM usage | No 4-bit quantization (CPU path) | Use a CUDA GPU or smaller model |

## Minimal LoRA Fine-Tuning Sketch (Not Implemented Here)
```python
from peft import get_peft_model
from torch.optim import AdamW

model, tokenizer, _ = load_model()  # from app.py functions
model.train()
optim = AdamW(model.parameters(), lr=2e-4)
for batch in dataloader:
		outputs = model(**batch)
		loss = outputs.loss
		loss.backward()
		optim.step(); optim.zero_grad()
model.save_pretrained("lora-adapter")
```
Then later: `model = PeftModel.from_pretrained(base_model, "lora-adapter")`.

## Quick Test (Smoke)
```bash
python app.py | grep 1948
```
Exit code 0 + line containing 1948 indicates success.

## Performance Tips
* For repeated inference, wrap multiple questions in a batch to reduce overhead.
* Pin exact versions in `requirements.txt` for reproducibility in CI.
* Use `torch.set_num_threads(N)` to control CPU threading if needed.

## Security / Safety Note
The script downloads a model from Hugging Face hub at runtime. For production, pre-download and checksum the model inside your build process.

## License
Add an explicit license file if you plan to share or publish this project.

## Notes
* LoRA adapters here are not trained; they simply wrap the model to demonstrate integration.
* For real fine-tuning you would run a training loop and then call `model.save_pretrained()` for the adapter weights.
* If you encounter `bitsandbytes` or `triton` import errors on CPU-only machines, this is expected; the script will fall back automatically.

## Next Steps / Ideas
* Add a small FastAPI service endpoint for QA.
* Provide a training script to fine-tune the LoRA adapters.
* Add unit tests around the fallback logic.
