# Smart Analytics — Setup Guide

## 1. Requirements

Python 3.10+ recommended.

```bash
pip install streamlit pandas numpy plotly openpyxl python-docx python-pptx reportlab statsmodels scikit-learn google-generativeai
```

Minimum install (core dashboard, no PDF/DOCX/PPTX/Gemini/ML outlier support):
```bash
pip install streamlit pandas numpy plotly openpyxl
```
The app degrades gracefully — Isolation Forest falls back to Z-Score if
`scikit-learn` is missing, forecasting falls back to a linear trend if
`statsmodels` is missing, and each report format tells you exactly which
package to install if it's absent.

## 2. Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## 3. First-time setup

1. Log in with a demo account (see below).
2. Go to **Admin / Configuration → Sample Data** and click **Generate Sample
   Data** to populate the app with realistic test data, OR go to **Setup**
   first to define your real product hierarchy, then **Data Upload** to load
   real data via the templates.
3. Configure your **company name** and currency label under **Setup →
   Company Settings**.

## 4. Demo credentials

| Username | Password | Role  |
|----------|----------|-------|
| admin    | admin123 | Admin |
| ceo      | ceo123   | CEO   |
| coo      | coo123   | COO   |
| cfo      | cfo123   | CFO   |

**Change these immediately** via **Admin / Configuration → User Management**
before any production use — the reference credential store is a local
SHA-256 hashed JSON file (`smart_analytics_data/users.json`), suitable for
demos but not a substitute for a hardened identity provider.

## 5. Production-grade authentication (recommended upgrade path)

The bundled `authenticate_user()` function is intentionally dependency-free
so the app runs out of the box. For production, replace it with one of:

- **Streamlit native auth (OIDC)** — `st.login()` / `st.experimental_user`
  wired to Azure AD, Okta, Google Workspace, etc. See
  https://docs.streamlit.io/develop/concepts/connections/authentication
- **streamlit-authenticator** — cookie-based sessions with bcrypt hashing:
  `pip install streamlit-authenticator`

## 6. Gemini API configuration

1. Get an API key from Google AI Studio.
2. In the app, go to **Admin / Configuration → AI & Outlier Settings** and
   paste the key into **Gemini API Key**, then **Save**.
3. Default model is **`gemini-3.7-flash`** — editable in the same screen
   (**Gemini Model** field) if you need to point at a different model.
4. `pip install google-generativeai` if not already installed.
5. Without a key configured, **AI Insights** automatically falls back to a
   deterministic, rule-based executive summary — the feature works with
   zero external dependencies, and that same summary is what gets embedded
   in both the Portfolio Report PDF and the full Management Report.

Alternatively, set the key via environment variable and read it in
`load_config()`'s default dict instead of typing it into the UI each time.

## 7. Data model

Uploads must map to these core fields (see **Data Upload** for downloadable
templates per data type — Budget, Actual, Forecast, Claims, Production,
Expenses):

`month, year, group_of_product, product, sub_product, line_of_business,
branch, channel, agent, gross_written_premium, net_written_premium,
earned_premium, claims_reported, claims_paid, outstanding_claims,
commissions, brokerage_costs, direct_costs, indirect_costs,
reinsurance_costs, receivables, budget, actual, forecast`

The **Setup** screen's Group of Product → Product → Sub-product hierarchy
must be defined first — uploads are validated against it. Each Group of
Product also serves as the Line of Business, so the bundled demo hierarchy
covers all major LOBs out of the box: Motor, Property, Marine, Medical,
Engineering, Liability, Travel, and Group Life. Generate sample data for
any period from **Admin / Configuration → Sample Data** (up to 60 months)
and every LOB will be represented.

All figures throughout the app — dashboards, tables, exports, and reports —
are rounded to the nearest whole number.

## 8. Currency

All monetary values are formatted with the official CBO Rial Omani symbol
(﷼) as a prefix, per the Central Bank of Oman's published symbol guidance
(cbo.gov.om/omrsymbol). The symbol and code are centralized in the
`CURRENCY_SYMBOL` / `CURRENCY_CODE` constants at the top of `app.py` if you
need to adjust formatting.

## 9. File/data storage

By default the app persists to local JSON/CSV files under
`smart_analytics_data/` (hierarchy, config, users, transactions). For a
multi-user production deployment, swap these `_load_json` / `_save_json` /
`load_data` / `save_data` helpers for a real database (PostgreSQL, etc.) —
they're isolated at the top of `app.py` specifically to make this swap easy.

## 10. Reports

**Reports** has two tabs:

- **Full Management Report (PDF)** — the rich, board-ready deliverable:
  cover page, Executive Summary (AI Insights), a complete **KPI Dashboard
  with plain-English definitions** for every ratio, a detailed **Variance
  Analysis** (budget vs actual vs forecast by Line of Business, plus a
  driver breakdown chart), and the **Production Forecast** (chart + table,
  using the same Holt-Winters/linear-trend model as the Forecasting page).
- **Standard Reports** — quicker single-purpose exports: PDF (portfolio
  summary + AI Executive Insights), Excel, PPTX, or DOCX.

Both PDF report types always embed the current AI Executive Insights text,
whether that's Gemini-generated or the rule-based fallback.

## 11. AI Insights and view exports (CEO / COO / CFO)

Each of the three role views now ends with its own **AI Insights** section
(cached independently per role, with a **Regenerate** button) plus
**Generate PDF** / **Generate PPTX** buttons that export that exact view —
its KPIs, its charts, and the current AI Insights text — as a fully
formatted, downloadable file. This is separate from the Reports page: it's
a one-click export of the view the user is already looking at.

## 12. Extending the app

Key functions to extend, all named to match the original specification:

`load_hierarchy · save_hierarchy · validate_hierarchy · load_template
(→ make_template) · validate_upload · calculate_kpis · detect_outliers ·
forecast_metrics · variance_analysis · generate_ai_insights ·
authenticate_user · send_email_support · generate_reports`
