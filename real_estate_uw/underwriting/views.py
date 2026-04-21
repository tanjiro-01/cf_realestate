import json
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.db.models import Avg, Count, Q
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .models import PropertyAssessment
from .forms import PropertyAssessmentForm
from .ml_engine import predict
from .ai_explainer import get_ai_explanation, get_ai_request_payload


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_prediction(instance: PropertyAssessment) -> None:
    """Compute predictions and save to the model instance (no double-save)."""
    result = predict(instance.get_feature_dict())
    instance.predicted_score  = result['predicted_score']
    instance.risk_category    = result['risk_category']
    instance.recommendation   = result['recommendation']
    instance.confidence_level = result['confidence_level']


# ── Dashboard ──────────────────────────────────────────────────────────────────

def home(request):
    qs = PropertyAssessment.objects.all()

    total     = qs.count()
    invest    = qs.filter(recommendation='INVEST').count()
    hold      = qs.filter(recommendation='HOLD').count()
    avoid     = qs.filter(recommendation='AVOID').count()
    avg_score = qs.aggregate(avg=Avg('predicted_score'))['avg'] or 0

    # Risk breakdown for chart
    risk_counts = {
        'Low Risk':       qs.filter(risk_category='Low Risk').count(),
        'Moderate Risk':  qs.filter(risk_category='Moderate Risk').count(),
        'Medium Risk':    qs.filter(risk_category='Medium Risk').count(),
        'High Risk':      qs.filter(risk_category='High Risk').count(),
        'Very High Risk': qs.filter(risk_category='Very High Risk').count(),
    }

    recent = qs[:5]

    return render(request, 'underwriting/home.html', {
        'total':       total,
        'invest':      invest,
        'hold':        hold,
        'avoid':       avoid,
        'avg_score':   round(avg_score, 1),
        'risk_counts': json.dumps(risk_counts),
        'recent':      recent,
    })


# ── List ────────────────────────────────────────────────────────────────────────

def property_list(request):
    qs = PropertyAssessment.objects.all()

    # Filtering
    q          = request.GET.get('q', '').strip()
    risk_filter = request.GET.get('risk', '')
    rec_filter  = request.GET.get('rec', '')

    if q:
        qs = qs.filter(Q(project_name__icontains=q) | Q(location__icontains=q))
    if risk_filter:
        qs = qs.filter(risk_category=risk_filter)
    if rec_filter:
        qs = qs.filter(recommendation=rec_filter)

    return render(request, 'underwriting/property_list.html', {
        'assessments':   qs,
        'q':             q,
        'risk_filter':   risk_filter,
        'rec_filter':    rec_filter,
        'risk_choices':  PropertyAssessment.RISK_CHOICES,
        'rec_choices':   PropertyAssessment.RECOMMENDATION_CHOICES,
    })


# ── Create ─────────────────────────────────────────────────────────────────────

def property_create(request):
    if request.method == 'POST':
        form = PropertyAssessmentForm(request.POST)
        if form.is_valid():
            instance = form.save(commit=False)
            _run_prediction(instance)
            instance.save()
            messages.success(request, f'Assessment for "{instance.project_name}" created successfully.')
            return redirect('property_detail', pk=instance.pk)
    else:
        form = PropertyAssessmentForm()

    return render(request, 'underwriting/property_form.html', {
        'form':  form,
        'title': 'New Property Assessment',
        'btn':   'Run Assessment',
    })


# ── Detail ─────────────────────────────────────────────────────────────────────

def property_detail(request, pk):
    assessment = get_object_or_404(PropertyAssessment, pk=pk)
    features   = assessment.get_feature_dict()

    # Score gauge: 0-100 → 0-180 degrees (half-circle)
    gauge_deg = (assessment.predicted_score or 0) * 1.8

    return render(request, 'underwriting/property_detail.html', {
        'assessment': assessment,
        'features':   features,
        'gauge_deg':  gauge_deg,
    })


# ── Update ─────────────────────────────────────────────────────────────────────

def property_update(request, pk):
    assessment = get_object_or_404(PropertyAssessment, pk=pk)

    if request.method == 'POST':
        form = PropertyAssessmentForm(request.POST, instance=assessment)
        if form.is_valid():
            instance = form.save(commit=False)
            _run_prediction(instance)
            instance.save()
            messages.success(request, f'Assessment for "{instance.project_name}" updated.')
            return redirect('property_detail', pk=instance.pk)
    else:
        form = PropertyAssessmentForm(instance=assessment)

    return render(request, 'underwriting/property_form.html', {
        'form':       form,
        'assessment': assessment,
        'title':      f'Edit — {assessment.project_name}',
        'btn':        'Re-run Assessment',
    })


# ── Delete ─────────────────────────────────────────────────────────────────────

def property_delete(request, pk):
    assessment = get_object_or_404(PropertyAssessment, pk=pk)

    if request.method == 'POST':
        name = assessment.project_name
        assessment.delete()
        messages.success(request, f'Assessment for "{name}" deleted.')
        return redirect('property_list')

    return render(request, 'underwriting/property_confirm_delete.html', {
        'assessment': assessment,
    })


# ── AI Explain (AJAX) ──────────────────────────────────────────────────────────

def ai_explain(request, pk):
    """Returns AI explanation as JSON — called via fetch() from the detail page."""
    assessment = get_object_or_404(PropertyAssessment, pk=pk)
    request_payload = get_ai_request_payload(assessment)
    explanation = get_ai_explanation(assessment)
    return JsonResponse({
        'explanation': explanation,
        'request_body': request_payload,
        'provider': request_payload.get('provider'),
        'model': request_payload.get('model'),
    })
