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
├── source/
│   ├── rmsnorm.py         # Custom RMSNorm implementation
│   ├── autograd.py    # Custom AutoGrad Function for e^x + cos(y)
│   └── lion.py            # Lion optimizer implementation
│
├── tests/
│   ├── test_all.py 
│   ├── test_rmsnorm.py    # Tests for RMSNorm
│   ├── test_autograd.py   # Tests for ExpPlusCos
│   └── test_lion.py       # Tests for Lion optimizer
│
├── pyproject.toml
└── README.md
```

## Tests

```
python -m tests.test_all    
```