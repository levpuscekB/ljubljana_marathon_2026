"""Interactive historical Ljubljana Marathon race results explorer.

Run with: python -m streamlit run marathon_dashboard.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_core import (
    BASELINE, FEMALE, MALE, SELECTED, COMPARE, distances_for,
    apply_delay_models, clock, detect_checkpoint_basis, get_profile, local_percentile,
    mean_profile, normalize_text, pack_profiles, percentile_summary,
    read_results, athlete_name_key, same_distance_year_matches, faster_than_percentile,
)
from race_config import RACES, ALL_SPLITS, results_stem, EVENT_YEARS, race_date

st.set_page_config(page_title="Ljubljana Marathon | Historical results", page_icon="🏃", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
  .block-container {padding-top: 1.4rem; padding-bottom: 2.3rem; max-width: 1550px;}
  h1 {letter-spacing: -.04em; font-weight: 760 !important;}
  h2, h3 {letter-spacing: -.02em;}
  div[data-testid="stMetric"] {background: rgba(125,140,160,.055); border-radius: 11px; padding: 11px 13px;}
  .intro {font-size: 1.03rem; color: #64748b; margin-top: -.65rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
USE_WEBGL = True

@st.cache_data(show_spinner="Reading race results…")
def load_data(blob: bytes, filename: str):
    return read_results(io.BytesIO(blob), name=filename)

@st.cache_data(show_spinner="Estimating finishing-time baselines…")
def prepare_models(frame):
    return apply_delay_models(frame)


def source_file(race, year):
    root = Path(__file__).resolve().parent
    names = [results_stem(race, year) + ext for ext in ('.xlsx', '.csv')]
    if race == '21' and int(year) == 2026:
        names += ['2026_ljubljana_halfmarathon_results.xlsx',
                  '2026_ljubljana_halfmarathon_results.csv']
    for folder in dict.fromkeys([root, root.parent, Path.cwd()]):
        for filename in names:
            path = folder / filename
            if path.is_file():
                return path
    return None


@st.cache_data(show_spinner="Reading other year's finishers…")
def load_other_year(blob, filename, year):
    result = read_results(io.BytesIO(blob), name=filename)
    # Reject accidentally misnamed files that contain a different year.
    if 'year' in result and result['year'].notna().any():
        if set(result['year'].dropna().astype(int)) != {int(year)}:
            raise ValueError(f'{filename} contains a different year than {year}.')
    return result


def runner_name(rec):
    if rec is None:
        return ""
    return f"{rec['name']} · #{rec['bib']}"


def axis_clock(figure, column=None, start=None, end=None, is_time=True):
    low, high = (float(start), float(end)) if start is not None else (60, 240)
    if not np.isfinite(low+high):
        return
    stride = 15 if (high-low) < 160 else 30
    ticks = np.arange(np.floor(low/stride)*stride, np.ceil(high/stride)*stride+1, stride)
    if is_time:
        labels = [f"{int(t//60)}:{int(t%60):02d}" for t in ticks]
    else:
        labels = [str(t) for t in ticks]
    figure.update_xaxes(tickmode="array", tickvals=ticks.tolist(), ticktext=labels,
                        range=[max(0.0, low), high])


def base_chart(figure, height=400):
    figure.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=36, r=22, b=45, t=87),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font_size=12),
        plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(showline=True, linecolor="#C6CCD6", gridcolor="#EEF1F4", zeroline=False)
    figure.update_yaxes(showline=False, gridcolor="#EEF1F4", zeroline=False, rangemode="tozero")
    return figure


def draw_reference_lines(fig, summary, selections=None):
    marks = [
        ("p05", "5%", "#7A8391", "dot", 1.045),
        ("p25", "25%", "#7A8391", "dot", 1.115),
        ("p50", "50%", "#4A5568", "dash", 1.045),
        ("p75", "75%", "#7A8391", "dot", 1.115),
        ("p95", "95%", "#7A8391", "dot", 1.045),
        ("mean", "Mean", "#141E2B", "solid", 1.185),
    ]
    for key, tag, color, pattern, label_y in marks:
        x = summary[key] / 60
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color=color, width=1.65 if key != "mean" else 2.2, dash=pattern))
        fig.add_annotation(x=x, y=label_y, yref="paper", text=tag, showarrow=False,
                           font=dict(size=10, color=color), bgcolor="rgba(255,255,255,.86)",
                           borderpad=1)
    # Chart shapes/annotations draw above the histogram bars. Stagger labels
    # so both runners have separate, consistently colored markers.
    for index, (label, color, time_sec, _name) in enumerate(selections or []):
        x = time_sec / 60
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color=color, width=3.5, dash="solid" if index == 0 else "dash"))
        fig.add_annotation(x=x, y=1.255 + .083 * index, yref="paper", text=f"Runner {label}",
                           showarrow=False, font=dict(size=11, color=color),
                           bgcolor="rgba(255,255,255,.97)", borderpad=2)


def distribution_figure(df, gender_view, metric, selections, bin_minutes):
    col = "net_time_sec" if metric == "net" else "gross_time_sec"
    sub = df if gender_view == "Both" else df[df.gender == gender_view]
    title = "Net / chip time" if metric == "net" else "Gross / gun time"
    summary = percentile_summary(sub, col)
    fig = go.Figure()
    if summary is None:
        return base_chart(fig), summary
    times = sub[col].to_numpy(dtype=float)
    lo = np.floor(times.min() / 60 / bin_minutes) * bin_minutes
    hi = np.ceil(times.max() / 60 / bin_minutes) * bin_minutes
    edges = np.arange(lo, hi + bin_minutes * 1.001, bin_minutes, dtype=float)
    if len(edges) < 2:
        edges = np.array([lo, lo + bin_minutes])
    x = (edges[:-1] + edges[1:]) / 2
    for sex, label, color in [("M", "Men", MALE), ("F", "Women", FEMALE)]:
        if gender_view not in ("Both", sex):
            continue
        arr = sub.loc[sub.gender == sex, col].to_numpy(dtype=float) / 60
        counts, _ = np.histogram(arr, bins=edges)
        fig.add_trace(go.Bar(
            x=x, y=counts, width=bin_minutes * .91, name=label,
            marker=dict(color=color, line=dict(width=0)), opacity=.9,
            customdata=np.array([clock(t * 60) for t in x]),
            hovertemplate="Time bin midpoint: %{customdata}<br>Finishers: %{y:,}<extra>" + label + "</extra>"
        ))
    if gender_view == "Both":
        fig.update_layout(barmode="stack")
    selected_times = [(tag, color, float(rec[col]), runner_name(rec))
                      for rec, tag, color in selections
                      if gender_view == "Both" or rec.gender == gender_view]
    draw_reference_lines(fig, summary, selected_times)
    base_chart(fig, 430)
    fig.update_layout(title=dict(text=title, x=0.02, xanchor="left", y=.98,
                                 font=dict(size=16, color="#202938")),
                      margin=dict(l=34, r=15, b=48, t=128),
                      bargap=.03, hovermode="x")
    axis_clock(fig, start=lo - 2, end=hi + bin_minutes + 2)
    fig.update_xaxes(title_text="Finish time (h:mm)")
    fig.update_yaxes(title_text="Number of finishers")
    return fig, summary


def stats_row(summary):
    if summary is None:
        return ""
    return "  ·  ".join([
        f"**N:** {summary['n']:,}",
        f"**Mean:** {clock(summary['mean'])}",
        f"**5%:** {clock(summary['p05'])}",
        f"**25%:** {clock(summary['p25'])}",
        f"**Median:** {clock(summary['p50'])}",
        f"**75%:** {clock(summary['p75'])}",
        f"**95%:** {clock(summary['p95'])}",
    ])


def split_figure(data, selections, checkpoint_basis, max_background, line_opacity, race):
    xx, yy, n = pack_profiles(data, checkpoint_basis, sample_limit=max_background, race=race)
    fig = go.Figure()
    trace_class = go.Scattergl if USE_WEBGL else go.Scatter
    if n:
        # One trace per runner would slow down a large field. None separates
        # individual trajectories within this single transparent trace.
        fig.add_trace(trace_class(
            x=xx, y=yy, mode="lines", name=f"Runners ({n:,} paths)",
            line=dict(color=f"rgba(102,111,127,{line_opacity:.4f})", width=1.0),
            hoverinfo="skip", showlegend=True))
    dx, mean_y, mean_n, complete = mean_profile(data, checkpoint_basis, race=race)
    if mean_n:
        fig.add_trace(trace_class(
            x=dx, y=mean_y, mode="lines+markers", name=f"Mean ({mean_n:,} runners)",
            line=dict(color=BASELINE, width=3.4),
            marker=dict(size=6, color=BASELINE),
            customdata=[clock(y * 60) for y in mean_y],
            hovertemplate="%{x:.1f} km: %{customdata}<extra>Mean</extra>"))

    # IMPORTANT: same trace renderer as grey paths + added last. SVG overlays
    # can end up BEHIND a WebGL canvas even when later in fig.data.
    # Put pale halos first and then both colored trajectories on top.
    profiles = []
    for rec, tag, color in selections:
        if not ((data.runner_id == rec.runner_id).any()):
            continue
        pts = get_profile(rec, checkpoint_basis, race=race)
        a, b = zip(*pts)
        profiles.append((rec, tag, color, list(a), list(b)))
    for rec, tag, color, a, b in profiles:
        fig.add_trace(trace_class(
            x=a, y=[v / 60 for v in b], mode="lines", showlegend=False,
            line=dict(color="rgba(255,255,255,.97)", width=7.2), hoverinfo="skip"))
    for rec, tag, color, a, b in profiles:
        fig.add_trace(trace_class(
            x=a, y=[v / 60 for v in b], mode="lines+markers",
            name=f"Runner {tag}: {runner_name(rec)}", legendgroup=f"runner_{tag}",
            line=dict(color=color, width=4.5, dash="solid" if tag == "A" else "dash"),
            marker=dict(color=color, size=10 if tag == "A" else 11,
                        symbol="circle" if tag == "A" else "diamond",
                        line=dict(width=1.8, color="white")),
            customdata=[clock(v) for v in b],
            hovertemplate=f"Runner {tag}: {runner_name(rec)}<br>" +
                          "%{x:.1f} km: %{customdata}<extra></extra>"))
    base_chart(fig, 550)
    fig.update_layout(margin=dict(l=54, r=35, b=55, t=70), hovermode="closest",
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    ds = distances_for(race)
    fig.update_xaxes(title_text="Distance (km)", range=[-.25, RACES[race]['km']*1.025],
                     tickmode="array", tickvals=ds,
                     ticktext=[f"{d:g}" if d != RACES[race]['km'] else f"{d:.2f}" for d in ds])
    fig.update_yaxes(title_text="Cumulative net time (minutes)")
    return fig, n, mean_n, complete


def scatter_background(fig, df, xcol, ycol, sex, hover_label, opacity):
    sub = df[(df.gender == sex) & df[ycol].notna()]
    if sub.empty:
        return
    color = MALE if sex == "M" else FEMALE
    label = "Men" if sex == "M" else "Women"
    display = [f"{a} · #{b}" for a, b in zip(sub.name, sub.bib)]
    trace_class = go.Scattergl if USE_WEBGL else go.Scatter
    fig.add_trace(trace_class(
        x=sub[xcol] / 60, y=sub[ycol] / 60,
        mode="markers", name=label, legendgroup=sex,
        marker=dict(color=color, size=4.5, opacity=opacity),
        customdata=np.array(display),
        hovertemplate="%{customdata}<br>Start delay: %{x:.1f} min<br>" + hover_label +
                      ": %{y:.1f} min<extra>" + label + "</extra>",
    ))


def model_figure(df, models, selections, view, point_opacity):
    fig = go.Figure()
    trace_class = go.Scattergl if USE_WEBGL else go.Scatter
    sexes = ["M", "F"] if view == "Both" else [view]
    for sex in sexes:
        scatter_background(fig, df, "start_delay_sec", "net_time_sec", sex,
                           "Net time", point_opacity)
    for sex in sexes:
        model = models.get(sex)
        if model is None:
            continue
        color = MALE if sex == "M" else FEMALE
        x = model.delay.to_numpy(dtype=float) / 60
        q1 = model.q25.to_numpy(dtype=float) / 60
        q3 = model.q75.to_numpy(dtype=float) / 60
        fig.add_trace(trace_class(
            x=np.r_[x, x[::-1]], y=np.r_[q3, q1[::-1]],
            fill="toself", fillcolor="rgba(110,110,110,.08)",
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(trace_class(
            x=x, y=model.expected / 60, mode="lines",
            line=dict(color=color, width=3.6),
            name=("Men" if sex == "M" else "Women") + " expected median",
            hovertemplate="Delay: %{x:.1f} min<br>Expected: %{y:.1f} min<extra></extra>"))

    # Connect each selected point to its gender-specific model first, then
    # draw both LARGE markers as the final traces (no WebGL/SVG layering mix).
    visible = [(rec, tag, color) for rec, tag, color in selections if rec.gender in sexes]
    for rec, tag, color in visible:
        if np.isfinite(rec.expected_sec):
            x = float(rec.start_delay_sec) / 60
            y = float(rec.net_time_sec) / 60
            expected = float(rec.expected_sec) / 60
            fig.add_trace(trace_class(
                x=[x, x], y=[y, expected], mode="lines", showlegend=False,
                line=dict(color=color, width=3, dash="dash"), hoverinfo="skip"))
    for rec, tag, color in visible:
        x = float(rec.start_delay_sec) / 60
        y = float(rec.net_time_sec) / 60
        fig.add_trace(trace_class(
            x=[x], y=[y], mode="markers", name=f"Runner {tag}: {runner_name(rec)}",
            marker=dict(color=color, size=20,
                        symbol="diamond" if tag == "B" else "circle",
                        line=dict(color="white", width=3)),
            hovertemplate=f"Runner {tag}: {runner_name(rec)}<br>" +
                          "Delay: %{x:.1f} min<br>Net: %{y:.1f} min<extra></extra>"))
    base_chart(fig, 510)
    fig.update_layout(margin=dict(l=42, r=15, b=50, t=85),
                      legend=dict(orientation="h", y=1.03, x=0), hovermode="closest")
    fig.update_xaxes(title_text="Start delay = gross - net (minutes)")
    fig.update_yaxes(title_text="Net finishing time (minutes)")
    return fig


def residual_figure(df, models, selections, view, point_opacity):
    fig = go.Figure()
    trace_class = go.Scattergl if USE_WEBGL else go.Scatter
    sexes = ["M", "F"] if view == "Both" else [view]
    for sex in sexes:
        scatter_background(fig, df, "start_delay_sec", "overperformance_sec", sex,
                           "Relative performance", point_opacity)
    fig.add_hline(y=0, line_width=2, line_color="#202938", line_dash="dash",
                  layer="below", annotation_text="Expected performance (0)",
                  annotation_position="top left")
    visible = [(rec, tag, color) for rec, tag, color in selections
               if rec.gender in sexes and np.isfinite(rec.overperformance_sec)]
    for rec, tag, color in visible:
        x = float(rec.start_delay_sec) / 60
        y = float(rec.overperformance_sec) / 60
        fig.add_trace(trace_class(
            x=[x, x], y=[0, y], mode="lines", showlegend=False,
            line=dict(color=color, width=3, dash="dot"), hoverinfo="skip"))
    for rec, tag, color in visible:
        x = float(rec.start_delay_sec) / 60
        y = float(rec.overperformance_sec) / 60
        fig.add_trace(trace_class(
            x=[x], y=[y], mode="markers", name=f"Runner {tag}: {runner_name(rec)}",
            marker=dict(color=color, size=20,
                        symbol="diamond" if tag == "B" else "circle",
                        line=dict(color="white", width=3)),
            hovertemplate=f"Runner {tag}: {runner_name(rec)}<br>" +
                          "Delay: %{x:.1f} min<br>Ahead of expected: %{y:+.1f} min<extra></extra>"))
    base_chart(fig, 510)
    fig.update_layout(margin=dict(l=42, r=15, b=50, t=85),
                      legend=dict(orientation="h", y=1.03, x=0), hovermode="closest")
    fig.update_xaxes(title_text="Start delay (minutes)")
    fig.update_yaxes(title_text="Faster (+) / slower (-) than expected (minutes)",
                     zeroline=False)
    return fig



def runner_picker(frame, tag, color, key_scope=""):
    """Independent name/bib search so runners A and B can be compared."""
    st.markdown(f'<span style="color:{color};font-size:19px">&#9679;</span> '
                f'**Runner {tag}**', unsafe_allow_html=True)
    search = st.text_input(f"Name or bib for runner {tag}", key=f"runner_search_{tag}_{key_scope}",
                           placeholder="Type a name or bib number")
    if not search.strip():
        return None
    query = normalize_text(search.strip())
    hits = frame.loc[frame.search.str.contains(query, regex=False)].copy()
    if hits.empty:
        st.warning(f"No match for runner {tag}. Try another name or bib.")
        return None
    hits = hits.sort_values(["net_time_sec", "name"]).head(100)
    options = hits.runner_id.tolist()
    labels = {rec.runner_id: f"{rec['name']}  |  #{rec.bib}  |  {rec.gender}  |  {clock(rec.net_time_sec)}"
              for _, rec in hits.iterrows()}
    choice = st.selectbox(f"Select runner {tag}", options, key=f"runner_choice_{tag}_{key_scope}",
                          format_func=lambda value: labels.get(value, value))
    if len(frame.loc[frame.search.str.contains(query, regex=False)]) > 100:
        st.caption("Showing the fastest 100 matches. Refine the name or bib.")
    return frame.loc[frame.runner_id == choice].iloc[0]


def runner_stats(rec, tag, color):
    st.markdown(f'<h4 style="color:{color};margin-bottom:.2rem">Runner {tag}: '
                f'{runner_name(rec)}</h4>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Net / chip", clock(rec.net_time_sec))
    b.metric("Gross / gun", clock(rec.gross_time_sec))
    c.metric("Start delay", clock(rec.start_delay_sec, compact=True))
    if pd.notna(rec.overperformance_sec):
        d.metric("Ahead of expectation", clock(rec.overperformance_sec, signed=True, compact=True))
    else:
        d.metric("Ahead of expectation", "Not enough data")


# ------------------------------- Main app ----------------------------------
st.title("Ljubljana Marathon · Historical results")
st.markdown('<p class="intro">Finish-time distributions, split trajectories and comparisons across runners and years · 10 / 21.1 / 42.2 km</p>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Race & data")
    year = st.selectbox("Choose race year", options=sorted(EVENT_YEARS,reverse=True),
                        index=0, key="race_year")
    race = st.selectbox("Choose race distance", options=list(RACES), index=1,
                        format_func=lambda value: RACES[value]['label'], key='race_distance')
    date_str = race_date(year, race)
    date_label = f"Official date: {date_str}" if date_str else str(year)
    st.caption(f"{date_label}  ·  {RACES[race]['km']:g} km")
    if race == "10" and year < 2019:
        st.caption("Historical recreational races were sometimes shorter than 10 km; the scraper validates the distance before saving.")
    uploaded = st.file_uploader("Upload CSV or Excel for this distance", type=["csv", "xlsx"],
                                key=f'upload_race_{year}_{race}',
                                help="Upload only results from the selected race distance.")
    if uploaded is not None:
        file_data, fname = uploaded.getvalue(), uploaded.name
        source_desc = f"Uploaded: {fname}"
    else:
        found = source_file(race, year)
        if found is None:
            st.info(f"No {year} {race} km file here yet. Run the scraper, or upload the results.")
            st.code(f"python scrape_ljubljana.py --year {year} --distance {race} --show-browser")
            st.stop()
        file_data, fname = found.read_bytes(), found.name
        source_desc = f"Local file: {fname}"
    st.caption(source_desc)
    if uploaded is None:
        status_file = found.with_suffix('.status.json')
        if status_file.exists():
            try:
                status = json.loads(status_file.read_text(encoding='utf-8'))
                if not status.get('overall_complete', False):
                    st.warning('This file is a PARTIAL scrape. Full-population statistics are not yet valid.')
                if status.get('checkpoint_category_failures'):
                    st.warning('Some age-category checkpoint pages failed: ' +
                               ', '.join(status['checkpoint_category_failures']))
            except (OSError, ValueError):
                pass

try:
    frame = load_data(file_data, fname)
    if 'year' in frame.columns and frame['year'].notna().any():
        if set(frame['year'].dropna().astype(int)) != {int(year)}:
            raise ValueError(f"File does not match selected year {year}. Please upload the correct results.")
except Exception as exc:
    st.error(f"Cannot read this results file: {exc}")
    st.stop()
if frame.empty:
    st.error("No valid finishers found in the supplied file.")
    st.stop()
frame, models = prepare_models(frame)

with st.sidebar:
    st.divider()
    st.subheader("Compare runners")
    selected_a = runner_picker(frame, "A", SELECTED, f"{year}_{race}")
    selected_b = runner_picker(frame, "B", COMPARE, f"{year}_{race}")
    if selected_a is not None and selected_b is not None and selected_a.runner_id == selected_b.runner_id:
        st.warning("Runner B is the same person as A. Pick someone else to compare.")
        selected_b = None
    if selected_a is None and selected_b is None:
        st.caption("Search by name or bib for either runner. Both can be highlighted at once.")
    st.divider()
    bins = st.slider("Histogram bin width (minutes)", min_value=1, max_value=15, value=5)
    inferred_basis = detect_checkpoint_basis(frame, race=race)
    split_basis = st.radio("Checkpoint clock basis", ["gross", "net"],
                           index=0 if inferred_basis == "gross" else 1,
                           format_func=lambda basis: ("Gun elapsed (subtract start delay)" if basis == "gross"
                                                      else "Chip elapsed (use as-is)"),
                           help="21 km can be inferred from 20 km data. For 10/42 km this is an assumption; override if needed.")
    if race == '21' and inferred_basis == 'gross':
        st.caption("21 km: detected gun-based splits; subtracting start delay.")
    elif race in ('10', '42'):
        st.caption("10/42 km: clock basis cannot be inferred reliably; gun elapsed is the default assumption. You can change it.")
    max_lines = st.select_slider("Max grey background runners",
                                 options=[1500, 3000, 6000, 10000, 20000],
                                 value=20000,
                                 help="Statistics always include every finisher.")
    line_opacity = st.slider("Split trajectory opacity", .005, .100, .020, .005,
                             format="%.3f", help="Lower alpha creates smoother density gradients from overlapping runners.")
    point_opacity = st.slider("Scatter background opacity", .02, .30, .09, .01,
                              format="%.2f", help="Transparency of unselected runners in overperformance plots.")
    USE_WEBGL = st.checkbox("Fast rendering (WebGL)", value=True,
                            help="Turn OFF if your browser cannot display WebGL.")

selections = []
if selected_a is not None:
    selections.append((selected_a, "A", SELECTED))
if selected_b is not None:
    selections.append((selected_b, "B", COMPARE))

n_m = int((frame.gender == "M").sum())
n_f = int((frame.gender == "F").sum())
split_cols = [f"{d}km_net_sec" for d in RACES[race]["checkpoints"]]
n_split = int(frame[split_cols].notna().any(axis=1).sum())
c1,c2,c3,c4 = st.columns(4)
c1.metric("Finishers", f"{len(frame):,}")
c2.metric("Men", f"{n_m:,}")
c3.metric("Women", f"{n_f:,}")
c4.metric("Runners with checkpoints", f"{n_split:,}")

if len(frame) < 100 or n_m == 0 or n_f == 0:
    st.warning("This may be a partial scrape or sample (fewer than 100 finishers or a missing gender). Plots use only the loaded records, not necessarily the full race.")
if n_split == 0:
    st.warning("This race file has no intermediate checkpoint times. Finishing-time distributions and overperformance still work; split trajectories require measured checkpoint data, which may be unavailable for the 10 km race.")

if selections:
    st.subheader("Runner comparison" if len(selections) == 2 else "Selected runner")
    if len(selections) == 2:
        # Two columns keep the comparison compact and clearly color-coded.
        left, right = st.columns(2)
        for col, (rec, tag, color) in zip([left, right], selections):
            with col:
                st.markdown(f'<h4 style="color:{color}">Runner {tag}: {runner_name(rec)}</h4>',
                            unsafe_allow_html=True)
                a, b = st.columns(2)
                a.metric("Net time", clock(rec.net_time_sec))
                b.metric("Gross time", clock(rec.gross_time_sec))
                a.metric("Start delay", clock(rec.start_delay_sec, compact=True))
                b.metric("Relative performance", clock(rec.overperformance_sec, signed=True, compact=True))
    else:
        runner_stats(*selections[0])

section_distributions, section_splits, section_overperformance, section_years, section_data = st.tabs(
    ["Finish distributions", "Split trajectories", "Overperformance", "Compare years", "Data quality"])

with section_distributions:
    st.subheader("Finishing time distributions")
    st.caption("Men in blue, women in pink; combined histograms stack both groups. Vertical markers: 5%, 25%, 50%, 75%, 95%, mean (black), and runner A (gold) / runner B (teal).")
    sex_options = st.tabs(["Combined", "Men", "Women"])
    for t, sex in zip(sex_options, ["Both", "M", "F"]):
        with t:
            col_net, col_gross = st.columns(2, gap="medium")
            with col_net:
                fig, summary = distribution_figure(frame, sex, "net", selections, bins)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
                st.markdown(stats_row(summary))
            with col_gross:
                fig, summary = distribution_figure(frame, sex, "gross", selections, bins)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
                st.markdown(stats_row(summary))
    st.caption("Percentile marker 5% is the time at which 5% of finishers were faster; 95% is the time at which 95% were faster. Gross = net + start delay.")

with section_splits:
    st.subheader("All runners: cumulative race time by distance")
    view = st.segmented_control("Show runners", options=["Both", "M", "F"], default="Both",
                                format_func=lambda a: {"Both":"All", "M":"Men", "F":"Women"}[a],
                                key="split_sex")
    sub = frame if view == "Both" else frame[frame.gender == view]
    split_selections = [(rec, tag, color) for rec, tag, color in selections
                        if view == "Both" or rec.gender == view]
    fig, n, mean_n, complete = split_figure(sub, split_selections, split_basis, max_lines, line_opacity, race)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
    if not n:
        st.info("No usable checkpoint records in this view. Load the full CSV with split columns populated.")
    else:
        method = "runners with all recorded checkpoints" if complete else "runners with available checkpoints (per-distance means)"
        st.caption(f"{n:,} grey background trajectories displayed. Black line = mean of {mean_n:,} {method}. Gold = runner A; teal = runner B (if selected). Grey paths may be subsampled for speed.")
    for rec, tag, color in split_selections:
        if len(get_profile(rec, split_basis, race)) == 2:
            st.info(f"Runner {tag} has no usable intermediate checkpoints; only start and finish can be plotted.")
    st.caption(f"Y-axis = cumulative chip time, not pace. X-axis ends at the official {RACES[race]['km']:g} km finish. Checkpoint clock correction follows your sidebar setting. Missing checkpoints are never estimated.")

with section_overperformance:
    st.subheader("Start position and relative performance")
    st.write("**Start delay** is `gross − net`: how long after the gun you crossed the start. The fitted reference shows typical net time among runners with similar delays, estimated separately for men and women.")
    ov_view = st.segmented_control("Show runners", options=["Both", "M", "F"], default="Both",
                                   format_func=lambda a: {"Both":"All", "M":"Men", "F":"Women"}[a],
                                   key="ov_sex")
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Net time versus starting delay**")
        st.plotly_chart(model_figure(frame, models, selections, ov_view, point_opacity), use_container_width=True,
                        config={"displaylogo":False})
    with right:
        st.markdown("**Overperformance relative to similar starters**")
        st.plotly_chart(residual_figure(frame, models, selections, ov_view, point_opacity), use_container_width=True,
                        config={"displaylogo":False})
    for rec, tag, color in selections:
        if pd.notna(rec.overperformance_sec):
            local = local_percentile(frame, rec)
            st.markdown(f'<span style="color:{color};font-size:16px">&#9679;</span> '
                        f'**Runner {tag} | {runner_name(rec)}:** '
                        f'started **{clock(rec.start_delay_sec, compact=True)}** after the gun; '
                        f'net **{clock(rec.net_time_sec)}** vs expected **{clock(rec.expected_sec)}**; '
                        f'**{clock(rec.overperformance_sec, signed=True, compact=True)}** relative to baseline.',
                        unsafe_allow_html=True)
            if local:
                st.caption(f"Runner {tag} was faster than ~{local['percentile']:.0f}% of "
                           f"{local['n']} same-sex runners with nearby start delays.")
        else:
            st.info(f"Runner {tag}: Not enough same-sex finishers to fit a baseline.")
    st.caption("Interpretation: positive residual = faster than expected for that start delay; negative = slower. This is descriptive, NOT a causal estimate of congestion or a fair performance ranking. Corral allocation, predicted pace and fitness strongly influence both starting position and net result.")
    st.caption("Method: equal-count start-delay bins → median chip times → weighted monotonic (isotonic) fit. Nearby-start percentile is based on the nearest same-sex start delays. Sparse samples can be unstable.")


with section_years:
    st.subheader("Compare results across editions")
    st.caption("Choose any two downloaded years for the same distance. Each edition uses its own "
               "gender-specific reference population. Confirm runner identity; bib numbers can change.")
    available_years = sorted(
        {int(y) for y in EVENT_YEARS if y == year or source_file(race, y) is not None}
    )
    if len(available_years) < 2:
        st.info("Download at least one additional year for this race distance to compare results.")
        st.code(f"python scrape_ljubljana.py --year 2025 2026 --distance {race} --show-browser")
    else:
        default_second = available_years[-1]
        default_first = available_years[-2]
        col_first, col_second = st.columns(2)
        with col_first:
            y1 = st.selectbox("Earlier / reference year", options=available_years,
                              index=available_years.index(default_first), key=f"cross_y1_{race}")
        with col_second:
            other_options = [v for v in available_years if v != y1]
            desired = default_second if default_second in other_options else other_options[-1]
            y2 = st.selectbox("Comparison year", options=other_options,
                              index=other_options.index(desired), key=f"cross_y2_{race}")
        yearly = {}
        for y in (y1, y2):
            if y == year:
                yearly[y] = frame
                continue
            other_path = source_file(race, y)
            if other_path is not None:
                try:
                    other_df = load_other_year(other_path.read_bytes(), other_path.name, y)
                    yearly[y], _ = prepare_models(other_df)
                    status_path = other_path.with_suffix('.status.json')
                    if status_path.exists():
                        status = json.loads(status_path.read_text(encoding='utf-8'))
                        if not status.get('overall_complete'):
                            st.warning(f"{y} file is a partial scrape; percentile comparisons may be misleading.")
                except Exception as exc:
                    st.warning(f"Cannot load {y} {race} km results: {exc}")
        if y1 not in yearly or y2 not in yearly:
            st.info("Could not load both years. Add the matching Excel/CSV files and reload.")
        else:
            left_df, right_df = yearly[y1], yearly[y2]
            st.caption(f"{y1}: {len(left_df):,} finishers · {y2}: {len(right_df):,} finishers")
            cross_search = st.text_input(f"Find your {y1} result (name or bib)",
                                         key=f"cross_search_{race}_{y1}_{y2}",
                                         placeholder=f"Runner name or {y1} bib")
            selected_first = None
            selected_second = None
            if cross_search.strip():
                hits = left_df.loc[left_df.search.str.contains(normalize_text(cross_search.strip()), regex=False)]
                hits = hits.sort_values(['net_time_sec', 'name']).head(150)
                if hits.empty:
                    st.info(f"No runner in {y1} matched that search.")
                else:
                    lookup_first = {r.runner_id: r for _,r in hits.iterrows()}
                    key_first = st.selectbox(f"Choose the {y1} result", options=list(lookup_first),
                        format_func=lambda k: f"{lookup_first[k]['name']} · #{lookup_first[k]['bib']} · {lookup_first[k]['gender']} · {clock(lookup_first[k]['net_time_sec'])}",
                        key=f"cross_pick_first_{race}_{y1}_{y2}")
                    selected_first = lookup_first[key_first]
                    suggestions = same_distance_year_matches(selected_first, right_df)
                    if len(suggestions):
                        st.success(f"Found {len(suggestions)} possible name match(es) in {y2}. Confirm identity below.")
                    else:
                        st.info(f"No identical name in {y2}. Try manually searching for alternative spellings.")
                    manual_second = st.text_input(f"Optional: search {y2} by name/bib",
                                                  key=f"cross_search_second_{race}_{y1}_{y2}",
                                                  placeholder="Alternative name spelling")
                    candidates = suggestions.copy()
                    if manual_second.strip():
                        extra = right_df.loc[right_df.search.str.contains(
                            normalize_text(manual_second.strip()), regex=False)]
                        candidates = pd.concat([candidates, extra]).drop_duplicates('runner_id')
                    if len(candidates):
                        lookup_second = {r.runner_id: r for _,r in candidates.sort_values('net_time_sec').head(150).iterrows()}
                        key_second = st.selectbox(f"Confirm the {y2} result (not automatically linked)",
                            options=[None]+list(lookup_second), index=0,
                            format_func=lambda k: "Select a result to compare" if k is None else
                               f"{lookup_second[k]['name']} · #{lookup_second[k]['bib']} · {lookup_second[k]['gender']} · {clock(lookup_second[k]['net_time_sec'])}",
                            key=f"cross_pick_second_{race}_{y1}_{y2}")
                        if key_second is not None:
                            selected_second = lookup_second[key_second]
                    if selected_second is not None and selected_first.gender != selected_second.gender:
                        st.error("The results have different gender classifications. Check athlete identity.")
                        selected_second = None
                    if selected_second is not None and athlete_name_key(selected_first['name']) != athlete_name_key(selected_second['name']):
                        st.warning("The names differ. Verify that these are really the same athlete.")

            if selected_first is not None and selected_second is not None:
                percent_first = faster_than_percentile(left_df, selected_first)
                percent_second = faster_than_percentile(right_df, selected_second)
                delta = selected_first.net_time_sec - selected_second.net_time_sec
                c1, c2, change_col = st.columns(3)
                with c1:
                    st.markdown(f"**{y1} · gold**")
                    st.metric("Net time", clock(selected_first.net_time_sec))
                    st.metric("Within-gender percentile", f"{percent_first:.1f}%")
                    st.caption(f"{selected_first['name']} · bib #{selected_first['bib']} · gross {clock(selected_first.gross_time_sec)}")
                    st.metric("Start delay", clock(selected_first.start_delay_sec, compact=True))
                    st.metric("Above expected", clock(selected_first.overperformance_sec, signed=True, compact=True))
                with c2:
                    st.markdown(f"**{y2} · teal**")
                    st.metric("Net time", clock(selected_second.net_time_sec))
                    st.metric("Within-gender percentile", f"{percent_second:.1f}%")
                    st.caption(f"{selected_second['name']} · bib #{selected_second['bib']} · gross {clock(selected_second.gross_time_sec)}")
                    st.metric("Start delay", clock(selected_second.start_delay_sec, compact=True))
                    st.metric("Above expected", clock(selected_second.overperformance_sec, signed=True, compact=True))
                with change_col:
                    st.markdown(f"**Change from {y1} to {y2}**")
                    st.metric("Net time improvement", clock(delta, signed=True),
                              help=f"Positive means {y2} was faster than {y1}")
                    st.metric("Percentile point change", f"{percent_second-percent_first:+.1f} pp")
                    st.metric("Net pace change", f"{(selected_second.net_time_sec-selected_first.net_time_sec)/RACES[race]['km']:+.1f} s/km",
                              help="Negative means faster pace in comparison year")
                st.caption("Percentile: proportion of same-gender finishers slower than the runner in each edition. "
                           "Percentile changes also depend on changes in the field.")

                basis1col, basis2col = st.columns(2)
                with basis1col:
                    basis1 = st.radio(f"{y1} checkpoint clock", ["gross", "net"],
                                      index=0 if detect_checkpoint_basis(left_df,race)=="gross" else 1,
                                      format_func=lambda v: "Gun elapsed" if v=="gross" else "Chip elapsed",
                                      key=f"cross_basis_first_{race}_{y1}_{y2}", horizontal=True)
                with basis2col:
                    basis2 = st.radio(f"{y2} checkpoint clock", ["gross", "net"],
                                      index=0 if detect_checkpoint_basis(right_df,race)=="gross" else 1,
                                      format_func=lambda v: "Gun elapsed" if v=="gross" else "Chip elapsed",
                                      key=f"cross_basis_second_{race}_{y1}_{y2}", horizontal=True)
                fig = go.Figure()
                for y, rec, color, basis in ((y1, selected_first, SELECTED, basis1),
                                             (y2, selected_second, COMPARE, basis2)):
                    prof = get_profile(rec, checkpoint_basis=basis, race=race)
                    fig.add_trace(go.Scatter(x=[d for d,_ in prof], y=[t/60 for _,t in prof],
                        mode="lines+markers", name=f"{y} · {clock(rec.net_time_sec)}",
                        line=dict(color=color, width=4.5, dash="solid" if y==y1 else "dash"),
                        marker=dict(color=color, size=10, symbol="circle" if y==y1 else "diamond"),
                        customdata=[clock(t) for _,t in prof],
                        hovertemplate=f"{y}: %{{x:.1f}} km · %{{customdata}}<extra></extra>"))
                base_chart(fig, 480)
                fig.update_layout(title=f"Cumulative net time · {y1} vs {y2}",
                                  legend=dict(orientation="h", y=1.05, x=0),
                                  margin=dict(t=85,b=52,l=53,r=22))
                fig.update_xaxes(title="Distance (km)", range=[0, RACES[race]['km']*1.02])
                fig.update_yaxes(title="Elapsed chip time (minutes)")
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
                st.caption("Intermediate checkpoints are observed values only. Choose each year's checkpoint "
                           "clock convention. Historical checkpoints may be missing.")
                with st.expander("Compare checkpoint times"):
                    a=dict(get_profile(selected_first, basis1, race))
                    b=dict(get_profile(selected_second, basis2, race))
                    distances=sorted(set(a)&set(b))
                    table=pd.DataFrame([{
                        "Distance (km)":d,
                        f"{y1} chip elapsed":clock(a[d]),
                        f"{y2} chip elapsed":clock(b[d]),
                        f"Improvement in {y2}":clock(a[d]-b[d],signed=True),
                    } for d in distances if d>0])
                    st.dataframe(table, use_container_width=True, hide_index=True)
            elif selected_first is not None:
                st.caption(f"Select and confirm the corresponding {y2} result to see the comparison.")

with section_data:
    st.subheader("Data completeness")
    q = st.columns(3)
    q[0].metric("Valid finishers", f"{len(frame):,}")
    q[1].metric("With all race splits", f"{int(frame[split_cols].notna().all(axis=1).sum()):,}")
    q[2].metric("With any split", f"{n_split:,}")
    split_table = pd.DataFrame({
        "Checkpoint": [f"{km} km" for km in RACES[race]["checkpoints"]] + ["Net finish", "Gross finish"],
        "Available": [int(frame[c].notna().sum()) for c in split_cols + ["net_time_sec", "gross_time_sec"]],
    })
    split_table["Coverage"] = (split_table.Available / len(frame) * 100).map(lambda x: f"{x:.1f}%")
    st.dataframe(split_table, use_container_width=True, hide_index=True)
    if len(frame) > 0:
        st.download_button("Download cleaned data (CSV)", frame.drop(columns=["search", "color"]).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"ljubljana_{year}_{race}k_cleaned.csv", mime="text/csv")
    with st.expander("Notes about input and interpretation"):
        st.write("Reads CSV/XLSX exports of official overall classifications: both sexes, all ages. "
                 "Checkpoint values are retained from earlier exports only when the runner identity agrees, or taken from any displayed checkpoints. "
                 "Rows without valid net and gross finish are excluded; missing intermediate times are not interpolated. "
                 "For 10 km, the 5 km column may be completely blank at the official source. "
                 "The check-point clock basis for 10 km and 42 km cannot be independently verified using only this data: "
                 "the sidebar setting is an assumption. If a scrape is incomplete, the statistics are not population statistics.")

st.caption(f"Official {year} Ljubljana Marathon organizer archive · {RACES[race]['long_name']}. Men and women, all ages.")
