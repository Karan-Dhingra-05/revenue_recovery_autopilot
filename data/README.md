# Data Directory

This directory stores generated synthetic datasets used for development, training, and evaluation.

## Structure

```
data/
├── generated/        ← output of the synthetic data generator (gitignored)
│   ├── payments.csv
│   ├── customers.csv
│   └── failures.csv
└── README.md
```

## Usage

The synthetic dataset generator lives in `backend/scripts/generate_data.py` (Phase 1).

Run it with:

```bash
cd backend
python scripts/generate_data.py --records 5000 --seed 42
```

Files in `data/generated/` are gitignored. Regenerate from the fixed seed for reproducible results.
