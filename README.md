# Neural networl optimization utils (RMSNorm, AutoGrad, Lion optimizer)

## Pre work

```
poetry install
poetry env
```

## Project structure

```
project-root/
│
├── src/
│   └── rmsnorm.py          
│
├── tests/
│   └── test_rmsnorm.py     # Сравнение с torch.nn.RMSNorm
│
├── pyproject.toml
└── README.md
```


## Tests

```
python tests/test_rmsnorm.py
```