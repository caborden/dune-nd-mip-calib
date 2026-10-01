"""Plots with explicit pixel/interval denominators and no implicit charge cut."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter


def _annotate_empty(ax, count):
    if not count:
        ax.text(0.5, 0.5, "No eligible entries", transform=ax.transAxes, ha="center")


def make_plots(tables, output=None, *, sample="event", tick_us=0.1):
    """Six figures, plus returned per-channel/chip denominators.

    Totals use sum_quality_ok; ordered comparisons require order_quality_ok.
    Track plots additionally require all event-pixel hits to be associated with
    that track (association_complete). All excluded entries remain in CSVs.
    Gap histograms count adjacent intervals, unlike the pixel histograms.
    """
    if sample not in {"event", "track"}:
        raise ValueError("sample must be event or track")
    if not np.isfinite(tick_us) or tick_us <= 0:
        raise ValueError("tick_us must be positive")
    prefix = "track_" if sample == "track" else ""
    pixels = tables[prefix + "pixels"].copy()
    gaps = tables[prefix + "gaps"].copy()
    eligible = pixels["sum_quality_ok"].eq(True)
    if sample == "track" and len(pixels):
        eligible &= pixels["association_complete"].eq(True)
    good = pixels.loc[eligible]
    pairs = good.loc[(good["multiplicity"] == 2) & good["order_quality_ok"].eq(True)]
    figs = {}
    label = "event–pixel candidates" if sample == "event" else "track–pixel candidates (complete associations)"

    fig, ax = plt.subplots()
    counts = good["multiplicity"].value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set(xlabel="Recorded hits per pixel encounter", ylabel="Pixel encounters", title=label)
    _annotate_empty(ax, len(good)); figs["multiplicity"] = fig

    from .charge_splitting import PIXEL_KEY
    keys = PIXEL_KEY + (["track_id"] if sample == "track" else [])
    selected_gaps = gaps.merge(good[keys], on=keys, how="inner") if len(good) else gaps.iloc[:0]
    dt = selected_gaps.loc[selected_gaps["order_quality_ok"].eq(True), "delta_t_us"].to_numpy(dtype=float)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fig, (ax, tail) = plt.subplots(1, 2, figsize=(11, 4))
    if len(dt):
        # Native tick bins in a labeled zoom; tails remain visible in a full-range ECDF.
        zoom = dt[dt <= 20]
        edges = np.arange(-0.5 * tick_us, 20 + 1.5 * tick_us, tick_us)
        ax.hist(zoom, bins=edges)
        ordered = np.sort(dt)
        tail.step(ordered, np.arange(1, len(dt) + 1) / len(dt), where="post")
        tail.set_xscale("log")
        tail.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        tail.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        tail.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="Adjacent-hit separation [µs]", ylabel="Adjacent intervals",
           title=f"0–20 µs zoom; {int((dt > 20).sum())} intervals beyond zoom", xlim=(0, 20))
    tail.set(xlabel="Adjacent-hit separation [µs]", ylabel="Cumulative fraction", title="Full positive finite range", ylim=(0, 1.02))
    _annotate_empty(ax, len(dt)); figs["time_gaps"] = fig

    fig, ax = plt.subplots()
    values = good["Q_total"].to_numpy(dtype=float)
    edges = np.histogram_bin_edges(values, bins=50) if len(values) else np.linspace(0, 1, 51)
    for name, mask in [("1", good["multiplicity"] == 1), ("2", good["multiplicity"] == 2),
                       ("≥3", good["multiplicity"] >= 3)]:
        ax.hist(good.loc[mask, "Q_total"], bins=edges, histtype="step", label=f"N={name}, entries={mask.sum()}")
    ax.set(xlabel="Recorded Q sum [ke⁻]", ylabel="Pixel encounters", title=label)
    ax.legend(); _annotate_empty(ax, len(good)); figs["charge_totals"] = fig

    fig, ax = plt.subplots()
    ax.scatter(pairs["Q1"], pairs["Q2"], s=12, alpha=0.5)
    ax.set(xlabel="First hit Q₁ [ke⁻]", ylabel="Second hit Q₂ [ke⁻]", title=f"Exactly two hits, resolved ordering: {len(pairs)} pixels")
    _annotate_empty(ax, len(pairs)); figs["q2_vs_q1"] = fig

    fig, ax = plt.subplots()
    positive = pairs.loc[pairs["Q_total"] > 0]
    ax.scatter(positive["duration_us"], positive["Q2"] / positive["Q_total"], s=12, alpha=0.5)
    ax.set(xlabel="Second − first hit time [µs]", ylabel="Q₂ / Q sum",
           title=f"Two hits; positive sum: {len(positive)} pixels ({len(pairs)-len(positive)} excluded)")
    # Do not restrict fractions to [0,1]: constituent negative charges are retained.
    _annotate_empty(ax, len(positive)); figs["fraction_vs_time"] = fig

    channel_keys = ["file_id", "io_group", "pixel_io_channel", "chip_id", "channel_id"]
    channels = good.assign(repeated=good["multiplicity"] >= 2).groupby(channel_keys).agg(
        n_encounters=("multiplicity", "size"), n_repeated=("repeated", "sum")).reset_index()
    channels["repeat_fraction"] = channels["n_repeated"] / channels["n_encounters"]
    chips = channels.groupby(channel_keys[:-1])[["n_encounters", "n_repeated"]].sum().reset_index()
    chips["repeat_fraction"] = chips["n_repeated"] / chips["n_encounters"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(np.arange(len(chips)), chips["repeat_fraction"], s=20)
    if 0 < len(chips) <= 30:
        ax.set_xticks(np.arange(len(chips)), chips[channel_keys[:-1]].astype(str).agg("/".join, axis=1), rotation=90)
    ax.set(xlabel="Chip group (file / IO group / pixel IO key / chip)", ylabel="Repeated / observed pixel encounters",
           title="Observed-hit denominator; excludes zero-hit pixels")
    _annotate_empty(ax, len(chips)); figs["repeat_fraction"] = fig
    for fig in figs.values():
        fig.tight_layout()
    if output is not None:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(directory / f"{sample}_{name}.png", dpi=150)
        channels.to_csv(directory / f"{sample}_channel_fractions.csv", index=False)
        chips.to_csv(directory / f"{sample}_chip_fractions.csv", index=False)
    return figs, channels, chips


def plot_event(hits, *, title="Event", projection="yz"):
    """One marker per recorded hit; repeated pixels overlap in the yz projection."""
    if projection not in {"yz", "xz", "xy"}:
        raise ValueError("projection must be yz, xz, or xy")
    vertical, horizontal = projection
    from .charge_splitting import PIXEL_KEY
    repeated = hits.groupby(PIXEL_KEY)["hit_row"].transform("size") >= 2
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(hits[horizontal], hits[vertical], c=hits["t_drift_us"], s=20)
    ax.scatter(hits.loc[repeated, horizontal], hits.loc[repeated, vertical],
               s=80, facecolors="none", edgecolors="red", label="Repeated pixel")
    if "association_role" in hits:
        recovered = hits["association_role"] == "recovered"
        ax.scatter(hits.loc[recovered, horizontal], hits.loc[recovered, vertical],
                   marker="x", c="black", label="Recovered hit")
    ax.set(xlabel=f"{horizontal} [cm]", ylabel=f"{vertical} [cm]", title=title)
    ax.set_aspect("equal"); ax.legend()
    fig.colorbar(sc, ax=ax, label="Event-relative reconstructed drift time [µs]")
    fig.tight_layout()
    return fig


def main():
    from .scan import load_results
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results")
    p.add_argument("--output", required=True)
    p.add_argument("--sample", choices=["event", "track"], default="event")
    args = p.parse_args()
    tables, manifest = load_results(args.results)
    ticks = {float(f["attrs"]["run_info"]["crs_ticks"]) for f in manifest["files"]}
    if len(ticks) != 1:
        raise ValueError("Plot separate scans for files with different charge clocks")
    make_plots(tables, args.output, sample=args.sample, tick_us=ticks.pop())
    plt.close("all")


if __name__ == "__main__":
    main()
