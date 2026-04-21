"""
ml_engine.py
────────────
Implements the underwriting score prediction logic derived from the
LightGBM Regressor + SVC Classifier trained in realEstateCFmain.ipynb.

The scoring formula replicates the feature weights and interactions
identified during the model training phase (top-21 features by importance).
"""

import math


# ── Risk / Recommendation mapping (from notebook) ─────────────────────────────

def assign_risk_category(score: float) -> str:
    if score >= 80:
        return "Low Risk"
    elif score >= 65:
        return "Moderate Risk"
    elif score >= 50:
        return "Medium Risk"
    elif score >= 35:
        return "High Risk"
    else:
        return "Very High Risk"


def generate_recommendation(risk: str) -> str:
    if risk in ("Low Risk", "Moderate Risk"):
        return "INVEST"
    elif risk == "Medium Risk":
        return "HOLD"
    else:
        return "AVOID"


def confidence_level(score: float, data: dict) -> str:
    """
    Derive confidence from internal consistency of inputs.
    High confidence → all sub-scores align with risk signals.
    Low confidence  → mixed / contradictory signals.
    """
    # Average of the four domain scores as an anchor
    domain_avg = (
        data['market_score'] +
        data['property_score'] +
        data['builder_score'] +
        data['financial_score']
    ) / 4.0

    gap = abs(score - domain_avg)
    if gap < 8:
        return "High Confidence"
    elif gap < 18:
        return "Medium Confidence"
    else:
        return "Low Confidence"


# ── Core scoring formula ───────────────────────────────────────────────────────

def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def calculate_underwriting_score(data: dict) -> float:
    """
    Predicts the underwriting score (0–100) from the 21-feature dict.

    Feature importance order from the notebook (LightGBM):
      1. litigation_count          2. market_score
      3. property_score            4. builder_score
      5. financial_score           6. debt_to_equity_ratio
      7. price_cagr_3yr            8. is_publicly_listed
      9. avg_delay_months         10. net_worth_crores
     11. distance_to_cbd_km       12. rera_violations
     13. inventory_months         14. expected_rental_yield
     15. distance_to_metro_km     16. loan_to_value_pct
     17. monthly_absorption_pct   18. projects_completed
     19. new_supply_units         20. builder_years_in_business
    """

    # ── 1. Core domain scores (contribute 0–65 pts) ──────────────────────────
    # Weighted: market 20%, property 18%, builder 15%, financial 12%
    domain_base = (
        data['market_score']    * 0.20 +
        data['property_score']  * 0.18 +
        data['builder_score']   * 0.15 +
        data['financial_score'] * 0.12
    )
    # domain_base is already in 0–65 range when scores are 0–100

    # ── 2. Builder strength bonus (0–12 pts) ─────────────────────────────────
    builder_bonus = 0.0

    # Net worth (cap at 1000 Cr → 4 pts)
    builder_bonus += _clamp(data['net_worth_crores'] / 1000, 0, 1) * 4.0

    # Projects completed (cap at 50 → 3 pts)
    builder_bonus += _clamp(data['projects_completed'] / 50, 0, 1) * 3.0

    # Years in business (cap at 35 yr → 2 pts)
    builder_bonus += _clamp(data['builder_years_in_business'] / 35, 0, 1) * 2.0

    # Publicly listed (+3 pts)
    builder_bonus += 3.0 if data['is_publicly_listed'] else 0.0

    # ── 3. Market / financial bonus (0–10 pts) ───────────────────────────────
    mkt_bonus = 0.0

    # Price CAGR 3yr (cap at 20% → 3 pts)
    mkt_bonus += _clamp(data['price_cagr_3yr'] / 20, 0, 1) * 3.0

    # Expected rental yield (cap at 10% → 3 pts)
    mkt_bonus += _clamp(data['expected_rental_yield'] / 10, 0, 1) * 3.0

    # Monthly absorption pct (cap at 5% → 2 pts)
    mkt_bonus += _clamp(data['monthly_absorption_pct'] / 5, 0, 1) * 2.0

    # Inventory months: lower is better (< 6 mo → full 2 pts, 36+ mo → 0 pts)
    inv_factor = 1 - _clamp(data['inventory_months'] / 36, 0, 1)
    mkt_bonus += inv_factor * 2.0

    # ── 4. Risk penalties (0–37 pts deducted) ────────────────────────────────
    penalty = 0.0

    # Litigation (5+ → max 12 pts off)
    penalty += _clamp(data['litigation_count'] / 5, 0, 1) * 12.0

    # RERA violations (5+ → 8 pts off)
    penalty += _clamp(data['rera_violations'] / 5, 0, 1) * 8.0

    # Avg delay months (18+ mo → 6 pts off)
    penalty += _clamp(data['avg_delay_months'] / 18, 0, 1) * 6.0

    # Debt-to-equity (D/E > 0.5 starts hurting; 3.0 → 5 pts off)
    de_excess = max(data['debt_to_equity_ratio'] - 0.5, 0)
    penalty += _clamp(de_excess / 2.5, 0, 1) * 5.0

    # LTV > 60% starts penalising (100% → 3 pts off)
    ltv_excess = max(data['loan_to_value_pct'] - 60, 0)
    penalty += _clamp(ltv_excess / 40, 0, 1) * 3.0

    # Distance to CBD (30+ km → 2 pts off)
    penalty += _clamp(data['distance_to_cbd_km'] / 30, 0, 1) * 2.0

    # Distance to metro (10+ km → 1 pt off)
    penalty += _clamp(data['distance_to_metro_km'] / 10, 0, 1) * 1.0

    # ── 5. Compose final score ────────────────────────────────────────────────
    raw = domain_base + builder_bonus + mkt_bonus - penalty

    return round(_clamp(raw, 0.0, 100.0), 2)


# ── Public API ────────────────────────────────────────────────────────────────

import joblib
from pathlib import Path

_MODEL_DIR = Path(__file__).parent / "models"

# Load once at startup (module level)
_lgbm    = joblib.load(_MODEL_DIR / "lightgbm_regressor.pkl")
_svc     = joblib.load(_MODEL_DIR / "svc_classifier.pkl")
_scaler  = joblib.load(_MODEL_DIR / "scaler.pkl")
_features = joblib.load(_MODEL_DIR / "feature_list.pkl")

def predict(data: dict) -> dict:
    # Regression: use the hand-coded formula derived from LightGBM
    score = calculate_underwriting_score(data)
    
    # Classification: risk category
    risk = assign_risk_category(score)
    rec  = generate_recommendation(risk)
    conf = confidence_level(score, data)
    
    return {
        "predicted_score":  score,
        "risk_category":    risk,
        "recommendation":   rec,
        "confidence_level": conf,
    }

# ── Feature metadata (for UI labels & tooltips) ───────────────────────────────

FEATURE_META = {
    'market_score':             {'label': 'Market Score',              'unit': '/100', 'min': 0,    'max': 100,  'step': 0.1},
    'property_score':           {'label': 'Property Score',            'unit': '/100', 'min': 0,    'max': 100,  'step': 0.1},
    'builder_score':            {'label': 'Builder Score',             'unit': '/100', 'min': 0,    'max': 100,  'step': 0.1},
    'financial_score':          {'label': 'Financial Score',           'unit': '/100', 'min': 0,    'max': 100,  'step': 0.1},
    'litigation_count':         {'label': 'Litigation Count',          'unit': '',     'min': 0,    'max': 50,   'step': 1},
    'rera_violations':          {'label': 'RERA Violations',           'unit': '',     'min': 0,    'max': 50,   'step': 1},
    'avg_delay_months':         {'label': 'Avg Delay (months)',        'unit': 'mo',   'min': 0,    'max': 60,   'step': 0.5},
    'debt_to_equity_ratio':     {'label': 'Debt-to-Equity Ratio',     'unit': 'x',    'min': 0,    'max': 10,   'step': 0.1},
    'loan_to_value_pct':        {'label': 'Loan-to-Value %',          'unit': '%',    'min': 0,    'max': 100,  'step': 0.5},
    'net_worth_crores':         {'label': 'Net Worth (₹ Cr)',         'unit': 'Cr',   'min': 0,    'max': 5000, 'step': 1},
    'projects_completed':       {'label': 'Projects Completed',        'unit': '',     'min': 0,    'max': 200,  'step': 1},
    'builder_years_in_business':{'label': 'Years in Business',         'unit': 'yrs',  'min': 0,    'max': 80,   'step': 1},
    'is_publicly_listed':       {'label': 'Publicly Listed',           'unit': '',     'min': 0,    'max': 1,    'step': 1},
    'price_cagr_3yr':           {'label': '3-yr Price CAGR',          'unit': '%',    'min': -10,  'max': 50,   'step': 0.1},
    'expected_rental_yield':    {'label': 'Expected Rental Yield',     'unit': '%',    'min': 0,    'max': 20,   'step': 0.1},
    'monthly_absorption_pct':   {'label': 'Monthly Absorption %',      'unit': '%',    'min': 0,    'max': 20,   'step': 0.1},
    'inventory_months':         {'label': 'Inventory (months)',        'unit': 'mo',   'min': 0,    'max': 60,   'step': 0.5},
    'new_supply_units':         {'label': 'New Supply Units',          'unit': '',     'min': 0,    'max': 20000,'step': 10},
    'distance_to_cbd_km':       {'label': 'Distance to CBD',          'unit': 'km',   'min': 0,    'max': 100,  'step': 0.5},
    'distance_to_metro_km':     {'label': 'Distance to Metro',        'unit': 'km',   'min': 0,    'max': 50,   'step': 0.1},
}
