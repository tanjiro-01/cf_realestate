import json
import os

from django.conf import settings
from django.core.cache import cache


CACHE_TIMEOUT_SECONDS = 60 * 60


def build_prompt(assessment) -> str:
    data = assessment.get_feature_dict()

    positive_signals = []
    negative_signals = []

    if data['market_score'] >= 70:
        positive_signals.append(f"strong market score of {data['market_score']:.1f}/100")
    if data['property_score'] >= 70:
        positive_signals.append(f"solid property score of {data['property_score']:.1f}/100")
    if data['builder_score'] >= 70:
        positive_signals.append(f"credible builder score of {data['builder_score']:.1f}/100")
    if data['financial_score'] >= 70:
        positive_signals.append(f"healthy financial score of {data['financial_score']:.1f}/100")
    if data['is_publicly_listed']:
        positive_signals.append("builder is publicly listed (higher transparency)")
    if data['projects_completed'] >= 20:
        positive_signals.append(f"{data['projects_completed']} completed projects (track record)")
    if data['price_cagr_3yr'] >= 8:
        positive_signals.append(f"price CAGR of {data['price_cagr_3yr']:.1f}% over 3 years")
    if data['expected_rental_yield'] >= 4:
        positive_signals.append(f"rental yield of {data['expected_rental_yield']:.1f}%")
    if data['net_worth_crores'] >= 200:
        positive_signals.append(f"builder net worth ₹{data['net_worth_crores']:.0f} Cr")

    if data['litigation_count'] > 2:
        negative_signals.append(f"{data['litigation_count']} active litigations")
    if data['rera_violations'] > 1:
        negative_signals.append(f"{data['rera_violations']} RERA violations")
    if data['avg_delay_months'] > 6:
        negative_signals.append(f"average project delay of {data['avg_delay_months']:.0f} months")
    if data['debt_to_equity_ratio'] > 1.5:
        negative_signals.append(f"high D/E ratio of {data['debt_to_equity_ratio']:.2f}x")
    if data['loan_to_value_pct'] > 75:
        negative_signals.append(f"elevated LTV of {data['loan_to_value_pct']:.1f}%")
    if data['distance_to_cbd_km'] > 20:
        negative_signals.append(f"distance to CBD is {data['distance_to_cbd_km']:.1f} km")
    if data['inventory_months'] > 18:
        negative_signals.append(f"inventory overhang of {data['inventory_months']:.0f} months")

    positives_str = "; ".join(positive_signals) if positive_signals else "no strong positive signals were detected"
    negatives_str = "; ".join(negative_signals) if negative_signals else "no major red flags were detected"

    return f"""You are a senior real estate investment analyst at an Indian real estate underwriting firm.
Analyze the following property assessment and explain the prediction in clear, jargon-light language suitable for a real estate fund manager.

PROJECT: {assessment.project_name}
LOCATION: {assessment.location or 'Not specified'}

MODEL INPUTS (21 features):
{json.dumps(data, indent=2)}

MODEL OUTPUT:
  Underwriting Score : {assessment.predicted_score:.1f} / 100
  Risk Category      : {assessment.risk_category}
  Recommendation     : {assessment.recommendation}
  Confidence Level   : {assessment.confidence_level}

KEY SIGNALS:
  Positive: {positives_str}
  Negative: {negatives_str}

Write a concise (200–280 word) analysis covering:
1. Executive Summary - one sentence verdict.
2. Why this score - explain the 2-3 biggest drivers (positive and negative).
3. Key Risks - list 2-3 specific risks an investor should watch.
4. Recommendation rationale - why INVEST, HOLD, or AVOID.

Use clear headings. Do not repeat the raw numbers from the inputs unless they add insight. Write in a professional but conversational tone."""


def _provider_config():
    groq_api_key = getattr(settings, 'GROQ_API_KEY', None)
    if groq_api_key is None:
        groq_api_key = os.environ.get('GROQ_API_KEY', '')

    anthropic_api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
    if anthropic_api_key is None:
        anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY', '')

    if groq_api_key:
        return {
            'provider': 'groq',
            'model': 'llama-3.1-8b-instant',
            'api_key': groq_api_key,
            'configured': True,
        }

    if anthropic_api_key:
        return {
            'provider': 'anthropic',
            'model': 'claude-sonnet-4-20250514',
            'api_key': anthropic_api_key,
            'configured': True,
        }

    return {
        'provider': 'fallback',
        'model': 'rule-based-fallback',
        'api_key': '',
        'configured': False,
    }


def _cache_scope(assessment) -> str:
    updated_token = assessment.updated_at.isoformat() if assessment.updated_at else 'no-updated-at'
    return f'{assessment.pk}:{updated_token}'


def _payload_cache_key(assessment) -> str:
    return f'ai:payload:{_cache_scope(assessment)}'


def _explanation_cache_key(assessment) -> str:
    return f'ai:explanation:{_cache_scope(assessment)}'


def get_ai_request_payload(assessment) -> dict:
    cached_payload = cache.get(_payload_cache_key(assessment))
    if cached_payload is not None:
        return cached_payload

    provider = _provider_config()
    payload = {
        'provider': provider['provider'],
        'model': provider['model'],
        'configured': provider['configured'],
        'messages': [
            {
                'role': 'user',
                'content': build_prompt(assessment),
            }
        ],
    }

    if provider['configured']:
        payload['max_tokens'] = 700
    else:
        payload['preview_only'] = True

    cache.set(_payload_cache_key(assessment), payload, CACHE_TIMEOUT_SECONDS)
    return payload


def get_ai_explanation(assessment) -> str:
    cached_explanation = cache.get(_explanation_cache_key(assessment))
    if cached_explanation is not None:
        return cached_explanation

    provider = _provider_config()
    request_payload = get_ai_request_payload(assessment)

    if not provider['configured']:
        explanation = _fallback_explanation(assessment)
        cache.set(_explanation_cache_key(assessment), explanation, CACHE_TIMEOUT_SECONDS)
        return explanation

    prompt = request_payload['messages'][0]['content']

    try:
        if provider['provider'] == 'groq':
            from groq import Groq

            client = Groq(api_key=provider['api_key'])
            response = client.chat.completions.create(
                model=provider['model'],
                max_tokens=700,
                messages=[{'role': 'user', 'content': prompt}],
            )
            content = response.choices[0].message.content or ''
            if content.strip():
                explanation = content.strip()
                cache.set(_explanation_cache_key(assessment), explanation, CACHE_TIMEOUT_SECONDS)
                return explanation

        if provider['provider'] == 'anthropic':
            from anthropic import Anthropic

            client = Anthropic(api_key=provider['api_key'])
            response = client.messages.create(
                model=provider['model'],
                max_tokens=700,
                messages=[{'role': 'user', 'content': prompt}],
            )
            content = ''.join(getattr(block, 'text', '') for block in response.content).strip()
            if content:
                cache.set(_explanation_cache_key(assessment), content, CACHE_TIMEOUT_SECONDS)
                return content

        explanation = _fallback_explanation(assessment)
        cache.set(_explanation_cache_key(assessment), explanation, CACHE_TIMEOUT_SECONDS)
        return explanation
    except Exception as exc:
        explanation = f"AI explanation unavailable: {exc}\n\n{_fallback_explanation(assessment)}"
        cache.set(_explanation_cache_key(assessment), explanation, CACHE_TIMEOUT_SECONDS)
        return explanation


def _fallback_explanation(assessment) -> str:
    """Rule-based fallback when an API is not configured or unavailable."""
    data = assessment.get_feature_dict()
    score = assessment.predicted_score or 0
    risk = assessment.risk_category

    lines = [
        f"## Underwriting Summary - {assessment.project_name}",
        "",
        f"**Score: {score:.1f}/100 | {risk} | {assessment.recommendation}**",
        "",
        "### Score Drivers",
    ]

    if data['market_score'] >= 70:
        lines.append(f"- Strong market score ({data['market_score']:.0f}/100) supports demand quality.")
    if data['builder_score'] >= 70:
        lines.append(f"- Builder score ({data['builder_score']:.0f}/100) suggests a credible developer.")
    if data['price_cagr_3yr'] >= 8:
        lines.append(f"- 3-year price CAGR of {data['price_cagr_3yr']:.1f}% supports appreciation potential.")

    if data['litigation_count'] > 2:
        lines.append(f"- {data['litigation_count']} active litigations increase legal risk.")
    if data['avg_delay_months'] > 6:
        lines.append(f"- Average delay of {data['avg_delay_months']:.0f} months may affect delivery timelines.")
    if data['debt_to_equity_ratio'] > 1.5:
        lines.append(f"- D/E ratio of {data['debt_to_equity_ratio']:.2f}x signals financial leverage risk.")
    if data['rera_violations'] > 1:
        lines.append(f"- {data['rera_violations']} RERA violations raise compliance concerns.")

    lines += [
        "",
        "### Recommendation",
        {
            'INVEST': 'The risk-reward profile is favorable. Proceed with due diligence.',
            'HOLD': 'Mixed signals warrant monitoring before committing capital.',
            'AVOID': 'Risk factors outweigh potential returns. Consider alternatives.',
        }.get(assessment.recommendation, ''),
        "",
        '_This explanation is auto-generated. Set GROQ_API_KEY or ANTHROPIC_API_KEY for full AI-powered analysis._',
    ]

    return '\n'.join(lines)
