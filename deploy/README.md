# Streamlit Community Cloud dashboard

This directory contains the lightweight, read-only deployment of the research dashboard.
It does not need the 2.6 GB research database, model files, FinMind, or Shioaji.

## Refresh the snapshot

From the repository root, run:

```powershell
python tools/export_dashboard_snapshot.py
```

The exporter reads `database/quant.db` and atomically rebuilds
`deploy/dashboard_snapshot.sqlite`. Commit the refreshed snapshot whenever new research
runs should appear in the cloud dashboard.

## Streamlit Community Cloud settings

- Repository: `povmaru888/AI_Quant_Research`
- Branch: `main`
- Entrypoint: `deploy/streamlit_app.py`
- Python: `3.12`
- Secrets: none

The deployment is presentation-only. Research jobs, data updates, training, and live
trading remain local workflows.
