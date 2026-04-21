from django.db import models
from django.utils import timezone


class PropertyAssessment(models.Model):
    """
    Stores all 21 input features for real estate underwriting
    plus computed prediction outputs.
    """

    # ── Meta ──────────────────────────────────────────────────────────────────
    project_name = models.CharField(max_length=200, help_text="Name of the real estate project")
    location      = models.CharField(max_length=200, blank=True, help_text="City / area")
    created_at    = models.DateTimeField(default=timezone.now)
    updated_at    = models.DateTimeField(auto_now=True)

    # ── Feature Group 1: Scores (0–100) ───────────────────────────────────────
    market_score    = models.FloatField(help_text="Market attractiveness score (0–100)")
    property_score  = models.FloatField(help_text="Property quality score (0–100)")
    builder_score   = models.FloatField(help_text="Builder credibility score (0–100)")
    financial_score = models.FloatField(help_text="Financial health score (0–100)")

    # ── Feature Group 2: Builder Profile ──────────────────────────────────────
    is_publicly_listed       = models.BooleanField(default=False, help_text="Is builder listed on stock exchange?")
    net_worth_crores         = models.FloatField(help_text="Builder net worth (₹ Crores)")
    projects_completed       = models.PositiveIntegerField(help_text="Number of projects completed")
    builder_years_in_business = models.PositiveIntegerField(help_text="Years builder has been in business")

    # ── Feature Group 3: Risk Indicators ──────────────────────────────────────
    litigation_count    = models.PositiveIntegerField(help_text="Number of active litigations")
    rera_violations     = models.PositiveIntegerField(help_text="Number of RERA violations")
    avg_delay_months    = models.FloatField(help_text="Average project delay (months)")
    debt_to_equity_ratio = models.FloatField(help_text="Builder debt-to-equity ratio")
    loan_to_value_pct   = models.FloatField(help_text="Loan-to-value percentage (%)")

    # ── Feature Group 4: Financial Metrics ────────────────────────────────────
    price_cagr_3yr        = models.FloatField(help_text="3-year price CAGR (%)")
    expected_rental_yield = models.FloatField(help_text="Expected rental yield (%)")

    # ── Feature Group 5: Market Dynamics ──────────────────────────────────────
    monthly_absorption_pct = models.FloatField(help_text="Monthly absorption rate (%)")
    inventory_months        = models.FloatField(help_text="Inventory overhang (months)")
    new_supply_units        = models.PositiveIntegerField(help_text="New supply units in micro-market")

    # ── Feature Group 6: Location ─────────────────────────────────────────────
    distance_to_cbd_km   = models.FloatField(help_text="Distance to Central Business District (km)")
    distance_to_metro_km = models.FloatField(help_text="Distance to nearest metro station (km)")

    # ── Predicted Outputs (computed on save) ──────────────────────────────────
    RISK_CHOICES = [
        ('Low Risk',       'Low Risk'),
        ('Moderate Risk',  'Moderate Risk'),
        ('Medium Risk',    'Medium Risk'),
        ('High Risk',      'High Risk'),
        ('Very High Risk', 'Very High Risk'),
    ]
    RECOMMENDATION_CHOICES = [
        ('INVEST', 'INVEST'),
        ('HOLD',   'HOLD'),
        ('AVOID',  'AVOID'),
    ]
    CONFIDENCE_CHOICES = [
        ('High Confidence',   'High Confidence'),
        ('Medium Confidence', 'Medium Confidence'),
        ('Low Confidence',    'Low Confidence'),
    ]

    predicted_score        = models.FloatField(null=True, blank=True)
    risk_category          = models.CharField(max_length=20, choices=RISK_CHOICES,
                                              blank=True, default='')
    recommendation         = models.CharField(max_length=10, choices=RECOMMENDATION_CHOICES,
                                              blank=True, default='')
    confidence_level       = models.CharField(max_length=20, choices=CONFIDENCE_CHOICES,
                                              blank=True, default='')

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Property Assessment'
        verbose_name_plural = 'Property Assessments'

    def __str__(self):
        return f"{self.project_name} — {self.risk_category or 'Pending'}"

    def get_feature_dict(self):
        """Return the 21 model features as a plain dict."""
        return {
            'litigation_count':         self.litigation_count,
            'market_score':             self.market_score,
            'property_score':           self.property_score,
            'builder_score':            self.builder_score,
            'financial_score':          self.financial_score,
            'debt_to_equity_ratio':     self.debt_to_equity_ratio,
            'price_cagr_3yr':           self.price_cagr_3yr,
            'is_publicly_listed':       int(self.is_publicly_listed),
            'avg_delay_months':         self.avg_delay_months,
            'net_worth_crores':         self.net_worth_crores,
            'distance_to_cbd_km':       self.distance_to_cbd_km,
            'rera_violations':          self.rera_violations,
            'inventory_months':         self.inventory_months,
            'expected_rental_yield':    self.expected_rental_yield,
            'distance_to_metro_km':     self.distance_to_metro_km,
            'loan_to_value_pct':        self.loan_to_value_pct,
            'monthly_absorption_pct':   self.monthly_absorption_pct,
            'projects_completed':       self.projects_completed,
            'new_supply_units':         self.new_supply_units,
            'builder_years_in_business': self.builder_years_in_business,
        }

    # ── Badge helpers for templates ───────────────────────────────────────────
    @property
    def risk_badge_class(self):
        mapping = {
            'Low Risk':       'success',
            'Moderate Risk':  'info',
            'Medium Risk':    'warning',
            'High Risk':      'danger',
            'Very High Risk': 'dark',
        }
        return mapping.get(self.risk_category, 'secondary')

    @property
    def recommendation_badge_class(self):
        return {'INVEST': 'success', 'HOLD': 'warning', 'AVOID': 'danger'}.get(
            self.recommendation, 'secondary'
        )

    @property
    def score_color(self):
        if self.predicted_score is None:
            return '#6c757d'
        s = self.predicted_score
        if s >= 80:   return '#198754'
        if s >= 65:   return '#0dcaf0'
        if s >= 50:   return '#ffc107'
        if s >= 35:   return '#dc3545'
        return '#212529'
