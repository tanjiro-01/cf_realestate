# Real Estate Underwriting System
### Built for CodeFrontier Software · Thejaswi Bhat H

A **Django + SQLite** web application that wraps the ML model from
`realEstateCFmain.ipynb` (LightGBM Regressor + SVC Classifier) in a
full CRUD interface with an **AI-powered explanation** panel via Claude.

---

## Quick Start

### 1. Clone / unzip the project
```bash
cd real_estate_uw
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and set:
#   ANTHROPIC_API_KEY=sk-ant-...   ← for AI explanations
#   DJANGO_SECRET_KEY=<any random string>
```

### 5. Run migrations & create superuser
```bash
python manage.py migrate
python manage.py createsuperuser
```

### 6. Start the server
```bash
python manage.py runserver
```

Open **http://127.0.0.1:8000** in your browser.

---

## Features

| Feature | Description |
|---|---|
| **Dashboard** | Stats cards, risk donut chart, avg score, recent assessments |
| **CRUD** | Create / Read / Update / Delete property assessments |
| **ML Prediction** | Underwriting score (0–100), risk category, recommendation, confidence |
| **AI Explain** | Claude API generates plain-English investment analysis |
| **Search & Filter** | Filter by risk category, recommendation, or text search |
| **Admin Panel** | Django admin at `/admin/` |

---

## ML Model

The scoring engine (`underwriting/ml_engine.py`) replicates the feature
weights identified in the notebook's LightGBM Regressor training:

| Feature Group | Weight |
|---|---|
| Domain Scores (market/property/builder/financial) | 65% |
| Builder Strength (net worth, projects, experience, listing) | ~12% |
| Market / Financial (CAGR, yield, absorption) | ~10% |
| Risk Penalties (litigations, RERA, D/E, delays) | deducted |

**Risk thresholds (from notebook):**
- ≥ 80 → Low Risk → **INVEST**
- 65–79 → Moderate Risk → **INVEST**
- 50–64 → Medium Risk → **HOLD**
- 35–49 → High Risk → **AVOID**
- < 35 → Very High Risk → **AVOID**

---

## AI Feature

When `ANTHROPIC_API_KEY` is set in `.env`, clicking **"Explain This Assessment"**
on the detail page calls `claude-sonnet-4-20250514` and returns a 200–280 word
analysis covering:
- Executive summary
- Score drivers (positive & negative)
- Key investment risks
- Recommendation rationale

If no API key is configured, a rule-based fallback explanation is shown.

---

## Switching to MySQL

In `real_estate_uw/settings.py`, replace the `DATABASES` block:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'realestate_uw',
        'USER': 'your_user',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

Then `pip install mysqlclient` and run `python manage.py migrate`.

---

## Project Structure

```
real_estate_uw/
├── manage.py
├── requirements.txt
├── .env.example
├── real_estate_uw/          ← Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── underwriting/            ← Main app
    ├── models.py            ← PropertyAssessment (21 features + outputs)
    ├── forms.py             ← Validated input form
    ├── views.py             ← CRUD + AI explain views
    ├── urls.py
    ├── ml_engine.py         ← Scoring formula (from notebook)
    ├── ai_explainer.py      ← Claude API integration
    ├── admin.py
    └── templates/
        └── underwriting/
            ├── base.html               ← Sidebar layout
            ├── home.html               ← Dashboard
            ├── property_list.html      ← Filterable list
            ├── property_form.html      ← Create / edit form
            ├── property_detail.html    ← Result + AI panel
            └── property_confirm_delete.html
```
