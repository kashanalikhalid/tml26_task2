Assignment data lives here at runtime — it is gitignored.

Download with:

```bash
huggingface-cli download SprintML/tml26_task2 \
    --repo-type model \
    --local-dir . \
    --include "target_model/*" "suspect_models/*" "task_template.py"
```

Resulting layout:

```
data/
├── target_model/
│   ├── weights.safetensors
│   └── train_main_idx.json
├── suspect_models/
│   ├── suspect_000.safetensors
│   ├── ...
│   └── suspect_359.safetensors
└── cifar100/        (downloaded automatically by torchvision on first run)
```
