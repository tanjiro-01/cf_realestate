from unittest.mock import patch

from django.core.cache import cache
from django.test import Client, TestCase, override_settings

from .ai_explainer import get_ai_explanation, get_ai_request_payload
from .models import PropertyAssessment


class AIPayloadCachingTests(TestCase):
    def setUp(self):
        cache.clear()
        self.assessment = PropertyAssessment.objects.create(
            project_name='Test Project',
            location='Bengaluru',
            market_score=78.5,
            property_score=74.0,
            builder_score=80.0,
            financial_score=69.5,
            is_publicly_listed=True,
            net_worth_crores=450.0,
            projects_completed=28,
            builder_years_in_business=18,
            litigation_count=1,
            rera_violations=0,
            avg_delay_months=4.0,
            debt_to_equity_ratio=1.2,
            loan_to_value_pct=55.0,
            price_cagr_3yr=9.0,
            expected_rental_yield=4.8,
            monthly_absorption_pct=3.2,
            inventory_months=10.0,
            new_supply_units=1200,
            distance_to_cbd_km=8.0,
            distance_to_metro_km=1.8,
            predicted_score=76.4,
            risk_category='Moderate Risk',
            recommendation='INVEST',
            confidence_level='High Confidence',
        )
        self.client = Client()

    @override_settings(GROQ_API_KEY='', ANTHROPIC_API_KEY='')
    def test_ai_endpoint_returns_request_body(self):
        response = self.client.get(f'/assessments/{self.assessment.pk}/ai-explain/')
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('explanation', data)
        self.assertIn('request_body', data)
        self.assertEqual(data['provider'], 'fallback')
        self.assertEqual(data['request_body']['provider'], 'fallback')
        self.assertIn('messages', data['request_body'])
        self.assertIn('Test Project', data['request_body']['messages'][0]['content'])

    @override_settings(GROQ_API_KEY='', ANTHROPIC_API_KEY='')
    def test_request_payload_is_cached(self):
        with patch('underwriting.ai_explainer.build_prompt', wraps=get_ai_request_payload.__globals__['build_prompt']) as mock_prompt:
            first = get_ai_request_payload(self.assessment)
            second = get_ai_request_payload(self.assessment)

        self.assertEqual(first, second)
        self.assertEqual(mock_prompt.call_count, 1)

    @override_settings(GROQ_API_KEY='', ANTHROPIC_API_KEY='')
    def test_explanation_is_cached(self):
        with patch('underwriting.ai_explainer._fallback_explanation', return_value='cached fallback') as mock_fallback:
            first = get_ai_explanation(self.assessment)
            second = get_ai_explanation(self.assessment)

        self.assertEqual(first, 'cached fallback')
        self.assertEqual(second, 'cached fallback')
        self.assertEqual(mock_fallback.call_count, 1)
