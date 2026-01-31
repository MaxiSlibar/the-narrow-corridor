# Data

The training data (~1.4GB) is not included in this repository. You can generate it by running:
```bash
python src/training/experiment.py
```

This will create:

| File | Size | Contents |
|------|------|----------|
| `all_steps.bin` | ~100 MB | Per step: run ID, epoch, target, context, loss, gradient, change |
| `all_embeddings.bin` | ~1.3 GB | Embedding snapshot after every step (34 words × 10 dims) |
| `vocabulary.json` | ~5 KB | Vocabulary, word frequencies, config |

## Pre-generated data

If you want to skip training and use our exact data:

- **Zenodo**: [Coming soon]
- **Hugging Face**: [Coming soon]

## Binary format

### all_steps.bin

Each record (48 bytes):
```
- run_id: int32
- epoch: int32
- target_word: int32
- context_word: int32
- loss: float32
- gradient: float32[10]
- change_magnitude: float32
```

### all_embeddings.bin

Each snapshot (1360 bytes):
```
- embeddings: float32[34][10]
```
