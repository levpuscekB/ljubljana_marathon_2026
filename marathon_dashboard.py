"""Interactive 2026 Ljubljana 10 km, half-marathon and marathon explorer.

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
    read_results,
)
from race_config import RACES, ALL_SPLITS, results_stem

st.set_page_config(page_title="Ljubljana 2026 | All distances", page_icon="🏃", layout="wide",
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


def source_file(race):
    root = Path(__file__).resolve().parent
    names = [results_stem(race) + ext for ext in ('.xlsx', '.csv')]
    if race == '21':
        names += ['2026_ljubljana_halfmarathon_results.xlsx',
                  '2026_ljubljana_halfmarathon_results.csv']
    for folder in dict.fromkeys([root, root.parent, Path.cwd()]):
        for filename in names:
            path = folder / filename
            if path.is_file():
                return path
    return None


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



def runner_picker(frame, tag, color):
    """Independent name/bib search so runners A and B can be compared."""
    st.markdown(f'<span style="color:{color};font-size:19px">&#9679;</span> '
                f'**Runner {tag}**', unsafe_allow_html=True)
    search = st.text_input(f"Name or bib for runner {tag}", key=f"runner_search_{tag}",
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
    choice = st.selectbox(f"Select runner {tag}", options, key=f"runner_choice_{tag}",
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
st.title("Ljubljana Marathon 2026")
st.markdown('<p class="intro">Finish-time distributions, split trajectories and two-runner comparisons · 10 / 21.1 / 42.2 km</p>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Race & data")
    race = st.selectbox("Choose race distance", options=list(RACES), index=1,
                        format_func=lambda value: RACES[value]['label'], key='race_distance')
    st.caption(f"Official date: {RACES[race]['date']}  ·  {RACES[race]['km']:g} km")
    uploaded = st.file_uploader("Upload CSV or Excel for this distance", type=["csv", "xlsx"],
                                key=f'upload_race_{race}',
                                help="Upload only results from the selected race distance.")
    if uploaded is not None:
        file_data, fname = uploaded.getvalue(), uploaded.name
        source_desc = f"Uploaded: {fname}"
    else:
        found = source_file(race)
        if found is None:
            st.info(f"No {race} km file here yet. Run the scraper for {race} km, or upload the results.")
            st.code(f"python scrape_ljubljana_2026.py --distance {race} --show-browser")
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
    selected_a = runner_picker(frame, "A", SELECTED)
    selected_b = runner_picker(frame, "B", COMPARE)
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

section_distributions, section_splits, section_overperformance, section_data = st.tabs(
    ["Finish distributions", "Split trajectories", "Overperformance", "Data quality"])

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
                           file_name=f"ljubljana_{race}k_cleaned.csv", mime="text/csv")
    with st.expander("Notes about input and interpretation"):
        st.write("Reads CSV/XLSX exports of official overall classifications: both sexes, all ages. "
                 "For 42 km and 21 km, optional supplemental checkpoints are merged from age-category tables by bib. "
                 "Rows without valid net and gross finish are excluded; missing intermediate times are not interpolated. "
                 "For 10 km, the 5 km column may be completely blank at the official source. "
                 "The check-point clock basis for 10 km and 42 km cannot be independently verified using only this data: "
                 "the sidebar setting is an assumption. If a scrape is incomplete, the statistics are not population statistics.")

st.caption(f"Official 2026 TIMING Ljubljana {RACES[race]['long_name']} results. Men and women, all ages.")
