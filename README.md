# Ljubljana Marathon 2026 — Race Explorer

Interactive 10 km, half marathon (21.0975 km) and marathon (42.195 km) result visualizations.
Built with Streamlit and Plotly. Two-runner comparison, gender histograms, net/gross
percentiles, cumulative checkpoint trajectories, and start-delay / overperformance plots.

## Data included

- `2026_ljubljana_21k_results.xlsx`: historical half-marathon results (8,962 finishers).
- **Not included:** full 10 km and 42 km results. To publish those races, copy your
  *completed* local scraper outputs into this repository before deployment:
  - `2026_ljubljana_10k_results.xlsx` (or `.csv`)
  - `2026_ljubljana_42k_results.xlsx` (or `.csv`)

The app automatically discovers those filenames beside `marathon_dashboard.py`.
Do not publish partial files. Check the scraper's `2026_ljubljana_XXk_results.status.json`
for `overall_complete: true`. Marathon splits may still be incomplete if category enrichment fails.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m streamlit run marathon_dashboard.py
```

## Public deployment (Streamlit Community Cloud)

1. Create a new GitHub repository, public for the simplest setup.
2. Add all files **from this directory to the repository root**. Add your final
   10k and 42k result exports before or after deployment. Only upload data you
   have permission to publicly redistribute.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app** -> **Yup, I have an app**.
5. Select the repo, branch `main`, and entrypoint `marathon_dashboard.py`.
6. Choose a subdomain and deploy. It will have a public `*.streamlit.app` URL.

Cloud builds install only the five packages in requirements.txt. There is no need
for Playwright/Chromium when the app reads static exported results. The scraper
runs locally, not on the public server.

If you prefer keeping the GitHub repository private, grant Streamlit access to
that repository and change app sharing to **public** under app settings. This
makes the app public while keeping its source files inaccessible via GitHub.

To update results later, commit and push the new CSV/XLSX. Streamlit Cloud will
pick up the changes. For licensing/privacy, independently verify permission
before republishing identifiable participant data.

## Source

Official results: https://www.timingljubljana.si/

This is an independent visualization, not an official marathon organizer website.
