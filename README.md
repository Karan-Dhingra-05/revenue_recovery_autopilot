# Revenue Recovery Autopilot

> **Recover more revenue from failed payments, automatically — but safely.**

Razorpay AI Buildathon — Track 3: AI Revenue Recovery

---

## Problem

A failed payment is not necessarily lost revenue. Different failures need different interventions. This system detects failed payments, diagnoses why they failed, selects the highest-probability recovery action subject to deterministic policy guardrails, executes it via Razorpay Test Mode, and measures how much revenue was actually recovered.

---

## Architecture (Overview)

```
Failed Payment
    ↓
Revenue-at-Risk Detection
    ↓
ML Recovery Scoring  (action-conditional probability)
    ↓
LLM Reasoning        (structured decision, bounded tools)
    ↓
Policy Engine        (deterministic guardrails)
    ↓
Action Executor      (Payment Link / Reminder / Escalate)
    ↓
Razorpay Webhook     (outcome confirmation)
    ↓
Audit Log + Metrics
```

**Stack:**

| Layer       | Technology                               |
|-------------|------------------------------------------|
| Frontend    | Next.js 14, TypeScript, Tailwind CSS     |
| Backend     | Python 3.12, FastAPI, Pydantic           |
| Database    | PostgreSQL 16                            |
| Cache/Queue | Redis 7                                  |
| ML          | scikit-learn / LightGBM (Phase 3)        |
| AI          | LLM via direct API (Phase 4)             |
| Payments    | Razorpay Test Mode APIs + Webhooks       |

---

## Local Setup

### Prerequisites

- Docker & Docker Compose
- Python 3.12+
- Node.js 20+

### 1. Clone and configure environment

```bash
git clone <repo-url>
cd revenue-recovery-autopilot

# Create backend environment file
cp .env.example backend/.env
# Edit backend/.env and fill in values (PostgreSQL/Redis are pre-filled for Docker)
```

### 2. Start infrastructure

```bash
docker compose up -d
# Wait for postgres and redis to be healthy:
docker compose ps
```

### 3. Start the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4. Start the frontend

```bash
cd frontend
cp .env.local.example .env.local  # or create manually
npm install
npm run dev
# Opens at http://localhost:3000
```

### 5. Verify health

```bash
curl http://localhost:8000/api/health
# Expected: {"status":"ok","db":"ok","redis":"ok"}
```

---

## Environment Variables

See [`.env.example`](.env.example) for the full list. Copy to `backend/.env` for local development.

| Variable                | Required | Description                          |
|-------------------------|----------|--------------------------------------|
| `DATABASE_URL`          | Phase 0  | PostgreSQL connection string         |
| `REDIS_URL`             | Phase 0  | Redis connection string              |
| `RAZORPAY_KEY_ID`       | Phase 2  | Razorpay test key ID                 |
| `RAZORPAY_KEY_SECRET`   | Phase 2  | Razorpay test key secret             |
| `RAZORPAY_WEBHOOK_SECRET` | Phase 2 | Razorpay webhook signing secret     |
| `LLM_API_KEY`           | Phase 4  | LLM provider API key                 |

---

## Running Tests

```bash
cd backend
pytest -v
```

---

## Development Phases

| Phase | Description                  | Status      |
|-------|------------------------------|-------------|
| 0     | Repository + Infrastructure  | ✅ Done      |
| 1     | Database schema + seed data  | ⬜ Pending   |
| 2     | Razorpay integration         | ⬜ Pending   |
| 3     | ML recovery scoring          | ⬜ Pending   |
| 4     | Agent + policy engine        | ⬜ Pending   |
| 5     | Recovery execution           | ⬜ Pending   |
| 6     | Dashboard polish             | ⬜ Pending   |
| 7     | Evaluation + demo hardening  | ⬜ Pending   |
| 8     | Deployment + submission      | ⬜ Pending   |

---

## Known Limitations

- Everything runs in Razorpay **Test Mode** only — no real money moves.
- ML model is trained on synthetic data; probabilities are directionally correct, not calibrated to any real merchant.
- LLM output is always a proposal — the deterministic policy engine decides whether to execute.

---

## License

MIT
