from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router

app = FastAPI(
    title="Revenue Recovery Autopilot",
    description=(
        "Recover more revenue from failed payments, automatically — but safely. "
        "Razorpay AI Buildathon — Track 3: AI Revenue Recovery."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Allow the Next.js dev server to call the API during local development.
# In production, replace with the real deployed frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router, prefix="/api")
