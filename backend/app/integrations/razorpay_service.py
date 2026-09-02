"""
Razorpay service wrapper.

All Razorpay API interactions are centralised here so credentials
are never scattered across the codebase.

Important:
  - Amounts are stored in the DB as Decimal rupees (Numeric 14,2).
  - Razorpay expects amounts in *paise* (integer).
  - Conversion uses Decimal arithmetic only — never float.
  - Never log API secrets.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from decimal import Decimal, ROUND_HALF_UP

import razorpay

from app.config import settings

logger = logging.getLogger(__name__)

# Set RAZORPAY_EXECUTION_ENABLED=true in .env to allow real API calls.
# Defaults to False so bulk scripts can never accidentally create Payment Links.
EXECUTION_ENABLED: bool = getattr(
    settings, "razorpay_execution_enabled", False
)


def _get_client() -> razorpay.Client:
    """Return a Razorpay client initialised with env credentials."""
    key_id = settings.razorpay_key_id
    key_secret = settings.razorpay_key_secret

    if not key_id or not key_secret:
        raise RuntimeError("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set")
        
    if key_id.startswith("rzp_live_"):
        raise RuntimeError("Live mode credentials are not permitted in this buildathon implementation. Must use Test Mode keys (rzp_test_...).")

    return razorpay.Client(auth=(key_id, key_secret))


# ── Amount conversion ────────────────────────────────────────────────────────

def rupees_to_paise(amount_rupees: Decimal) -> int:
    """
    Convert a Decimal rupee amount to integer paise.

    Uses Decimal arithmetic exclusively — no float intermediary.

    >>> rupees_to_paise(Decimal("125.50"))
    12550
    >>> rupees_to_paise(Decimal("12500.00"))
    1250000
    """
    paise = (amount_rupees * Decimal("100")).to_integral_value(rounding=ROUND_HALF_UP)
    return int(paise)


# ── Payment Link ─────────────────────────────────────────────────────────────

def create_payment_link(
    amount_rupees: Decimal,
    currency: str,
    reference_id: str,
    description: str,
    customer_name: str | None = None,
    customer_email: str | None = None,
    customer_contact: str | None = None,
    expire_by: int | None = None,
) -> dict:
    """
    Create a Standard Razorpay Payment Link (TEST MODE).

    Parameters
    ----------
    amount_rupees : Decimal
        Transaction amount in rupees (from the trusted DB record).
    currency : str
        ISO currency code, e.g. "INR".
    reference_id : str
        Unique reference to prevent duplicate links.
    description : str
        Short description visible to the customer.

    Returns
    -------
    dict
        Razorpay response containing at minimum ``id`` and ``short_url``.

    Raises
    ------
    RuntimeError
        If credentials are missing.
    razorpay.errors.BadRequestError
        If parameters are invalid.
    """
    client = _get_client()
    amount_paise = rupees_to_paise(amount_rupees)

    payload: dict = {
        "amount": amount_paise,
        "currency": currency,
        "reference_id": reference_id,
        "description": description,
        "accept_partial": False,
    }

    # Optional customer info (never hard-coded)
    customer: dict = {}
    if customer_name:
        customer["name"] = customer_name
    if customer_email:
        customer["email"] = customer_email
    if customer_contact:
        customer["contact"] = customer_contact
    if customer:
        payload["customer"] = customer

    if expire_by:
        payload["expire_by"] = expire_by

    logger.info(
        "Creating Razorpay Payment Link: reference_id=%s amount_paise=%d currency=%s",
        reference_id,
        amount_paise,
        currency,
    )

    result = client.payment_link.create(payload)

    logger.info(
        "Payment Link created: id=%s short_url=%s",
        result.get("id"),
        result.get("short_url"),
    )
    return result


# ── Webhook signature verification ──────────────────────────────────────────

def verify_webhook_signature(
    raw_body: bytes,
    signature: str,
    secret: str | None = None,
) -> bool:
    """
    Verify a Razorpay webhook signature using HMAC SHA-256.

    Uses the RAW request body (bytes) — must not be parsed/re-encoded first.

    Parameters
    ----------
    raw_body : bytes
        The raw, unmodified HTTP request body.
    signature : str
        Value of the ``X-Razorpay-Signature`` header.
    secret : str or None
        Webhook secret. Defaults to ``settings.razorpay_webhook_secret``.

    Returns
    -------
    bool
        True if the signature is valid.
    """
    if secret is None:
        secret = settings.razorpay_webhook_secret
    if not secret:
        logger.error("RAZORPAY_WEBHOOK_SECRET is not configured")
        return False

    computed = hmac.new(
        key=secret.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, signature)
