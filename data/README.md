# Data Directory

This directory stores generated synthetic datasets used for development, training, and evaluation.

## Structure

```
data/
├── generated/        ← output of the synthetic data generator (gitignored)
│   ├── transactions.csv
│   ├── recovery_action_outcomes.csv
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── dataset_metadata.json
└── README.md
```

## Generate the Dataset

```bash
cd backend
python scripts/generate_dataset.py --records 5000 --seed 42
```

## Inspect the Dataset

```bash
cd backend
python scripts/inspect_dataset.py
```

## File Schemas

### transactions.csv

Every payment transaction (SUCCESS + FAILED). One row per payment.

| Column | Type | Description |
|---|---|---|
| transaction_id | str | Unique identifier (txn_NNNNNN) |
| customer_id | str | Customer identifier (cust_NNNN) |
| amount | float | Payment amount in INR (₹50–₹5,00,000) |
| currency | str | Always "INR" |
| payment_method | str | CARD / UPI / NET_BANKING / WALLET / EMI |
| status | str | SUCCESS or FAILED |
| attempt_number | int | Which attempt this is for the customer (≥1) |
| customer_success_rate | float | Historical success rate at time of txn |
| customer_previous_failures | int | Count of prior failures |
| customer_previous_recoveries | int | Count of prior recoveries |
| hours_since_last_success | float | Hours since customer's last success (-1 = never) |
| subscription_flag | int | 1 if subscription customer, 0 otherwise |
| created_at | ISO datetime | When the payment was attempted |
| created_hour | int | Hour of day (0–23) |
| created_day | str | Day of week (Monday–Sunday) |
| failure_reason | str/null | Reason code (null for SUCCESS) |
| failure_source | str/null | bank / gateway / customer / upi (null for SUCCESS) |
| failure_pattern | str/null | A_TEMP_BANK / B_INSUF_FUNDS / C_EXPIRED_INSTR / D_REPEATED_FAIL / E_LOW_VALUE |

### recovery_action_outcomes.csv

For each FAILED transaction, one row per candidate recovery action (4 actions × N_failed).

| Column | Type | Description |
|---|---|---|
| transaction_id | str | Foreign key to transactions.csv |
| action_type | str | PAYMENT_LINK / REMINDER / DELAYED_RETRY / ESCALATE |
| recovery_probability | float | Simulated P(recovery \| context, action) |
| recovered | int | **Label** — 1 if recovered, 0 otherwise |
| amount | float | Same as transaction amount |
| failure_pattern | str | Failure pattern label |

### train.csv / validation.csv / test.csv

Merged features + labels for ML. Same schema as recovery_action_outcomes joined with transaction features (minus `status`, which is always FAILED).

Split: 70% train / 15% val / 15% test, sorted by `created_at` (time-aware).

## Reproducibility

Files in `data/generated/` are gitignored. Regenerate from the fixed seed (`--seed 42`) for identical results.
