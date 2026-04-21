from django.contrib import admin
from .models import PropertyAssessment


@admin.register(PropertyAssessment)
class PropertyAssessmentAdmin(admin.ModelAdmin):
    list_display = ['project_name', 'location', 'predicted_score',
                    'risk_category', 'recommendation', 'confidence_level', 'created_at']
    list_filter  = ['risk_category', 'recommendation', 'confidence_level', 'is_publicly_listed']
    search_fields = ['project_name', 'location']
    readonly_fields = ['predicted_score', 'risk_category', 'recommendation',
                       'confidence_level', 'created_at', 'updated_at']
