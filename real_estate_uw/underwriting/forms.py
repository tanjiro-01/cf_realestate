from django import forms
from .models import PropertyAssessment


class PropertyAssessmentForm(forms.ModelForm):

    class Meta:
        model = PropertyAssessment
        exclude = ['predicted_score', 'risk_category', 'recommendation',
                   'confidence_level', 'created_at', 'updated_at']

    # ── Widget customisation ────────────────────────────────────────────────
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        float_fields = [
            'market_score', 'property_score', 'builder_score', 'financial_score',
            'debt_to_equity_ratio', 'price_cagr_3yr', 'net_worth_crores',
            'avg_delay_months', 'distance_to_cbd_km', 'expected_rental_yield',
            'distance_to_metro_km', 'loan_to_value_pct', 'monthly_absorption_pct',
            'inventory_months',
        ]
        int_fields = [
            'litigation_count', 'rera_violations', 'projects_completed',
            'new_supply_units', 'builder_years_in_business',
        ]

        for field_name in float_fields:
            self.fields[field_name].widget = forms.NumberInput(
                attrs={'class': 'form-control', 'step': '0.01'}
            )
        for field_name in int_fields:
            self.fields[field_name].widget = forms.NumberInput(
                attrs={'class': 'form-control', 'step': '1'}
            )

        # Text fields
        for field_name in ['project_name', 'location']:
            self.fields[field_name].widget = forms.TextInput(
                attrs={'class': 'form-control'}
            )

        # Boolean
        self.fields['is_publicly_listed'].widget = forms.CheckboxInput(
            attrs={'class': 'form-check-input'}
        )

        # Helpful placeholders
        self.fields['project_name'].widget.attrs['placeholder'] = 'e.g. Prestige Lakeside Habitat'
        self.fields['location'].widget.attrs['placeholder'] = 'e.g. Whitefield, Bengaluru'

    # ── Validation ──────────────────────────────────────────────────────────
    def clean_market_score(self):
        v = self.cleaned_data['market_score']
        if not (0 <= v <= 100):
            raise forms.ValidationError('Must be between 0 and 100.')
        return v

    def clean_property_score(self):
        v = self.cleaned_data['property_score']
        if not (0 <= v <= 100):
            raise forms.ValidationError('Must be between 0 and 100.')
        return v

    def clean_builder_score(self):
        v = self.cleaned_data['builder_score']
        if not (0 <= v <= 100):
            raise forms.ValidationError('Must be between 0 and 100.')
        return v

    def clean_financial_score(self):
        v = self.cleaned_data['financial_score']
        if not (0 <= v <= 100):
            raise forms.ValidationError('Must be between 0 and 100.')
        return v

    def clean_loan_to_value_pct(self):
        v = self.cleaned_data['loan_to_value_pct']
        if not (0 <= v <= 100):
            raise forms.ValidationError('Must be between 0 and 100.')
        return v
