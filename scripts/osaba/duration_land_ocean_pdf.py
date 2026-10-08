#!/usr/bin/env python3
"""Plot four duration PDFs from completed statistics and existing Amax classes.

No mosaic, label or 3D QC volume is read. Geographic classes are reused from
amax_land_ocean_pdf.py: the centroid at the first occurrence of Amax, not a
new class at birth. One component contributes one duration, in hours.

The default is the stored elapsed duration (last minus first observation).
Zero durations are counted explicitly and excluded from the positive-duration
PDF. --duration-definition covered adds one measured sampling interval, making
single-observation components visible without pretending their stored duration
was nonzero. Both axes are logarithmic by default.

Run from the repository root:
    .venv/bin/python test_simple/duration_land_ocean_pdf.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date

from amax_land_ocean_pdf import DEFAULT_OUTPUT as DEFAULT_CLASSIFICATIONS
from amax_land_ocean_pdf import DEFAULT_STATS, REGIONS, floats, integers
from reconcile.io.netcdf import _fsync_directory
from reconcile.io.runtime import destination_lease, unique_atomic_path
from tb_land_ocean_pdf import identity, signature, write_json

LOGGER = logging.getLogger("duration_land_ocean_pdf")
SCHEMA = "object_duration_geographic_pdf_v1"
DEFAULT_OUTPUT = Path("test_simple/outputs_duration_land_ocean_pdf_satellite_20160810_20160910")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-root", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--classification-dir", type=Path, default=DEFAULT_CLASSIFICATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds", type=float, nargs=2, default=[210.0, 220.0])
    parser.add_argument("--duration-definition", choices=("elapsed", "covered"), default="elapsed")
    parser.add_argument("--scale", choices=("loglog", "linear"), default="loglog")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--width", type=float, default=6.0)
    parser.add_argument("--height", type=float, default=4.0)
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (*args.thresholds, args.width, args.height)):
        parser.error("thresholds and figure dimensions must be finite and positive")
    if args.thresholds[0] == args.thresholds[1] or not 2 <= args.bins <= 1000:
        parser.error("require two distinct thresholds and 2..1000 bins")
    return args


def read_duration(
    path: Path,
    threshold: float,
    cache_path: Path,
    classification_key: str,
    definition: str,
) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as cache:
        if str(cache["signature"].item()) != classification_key:
            raise ValueError("Geographic classification cache has a different signature")
        ids = cache["component_id"].copy()
        region = cache["region"].copy()
        boundary = cache["window_boundary"].copy()
    if (
        region.shape != ids.shape
        or boundary.shape != ids.shape
        or np.any(~np.isin(region, (0, 1, 2)))
    ):
        raise ValueError("Invalid geographic component cache")
    with Dataset(path) as nc:
        if any(
            int(getattr(nc, flag, 0)) != 1 for flag in ("statistics_complete", "reconcile_complete")
        ):
            raise ValueError(f"Statistics are incomplete: {path}")
        if not np.isclose(float(getattr(nc, "Tb_seed_K", np.nan)), threshold):
            raise ValueError(f"Wrong threshold in {path}")
        for name in (
            "INT_CCnumber",
            "INT_duration",
            "INT_obs_count",
            "INT_UTC_timeInit",
            "INT_UTC_timeEnd",
        ):
            if nc[name].dimensions != ("CC",):
                raise ValueError(f"Wrong dimensions for {name}")
        if not np.array_equal(integers(nc["INT_CCnumber"]), ids):
            raise ValueError("Cached geographical IDs do not match statistics")
        if nc["INT_duration"].units != "h":
            raise ValueError("Stored durations must declare hours")
        elapsed = floats(nc["INT_duration"])
        if np.any(~np.isfinite(elapsed) | (elapsed < 0)):
            raise ValueError("Missing/nonfinite/negative duration in complete statistics")
        times = integers(nc["time"])
        steps = np.diff(times)
        if len(steps) == 0 or np.any(steps <= 0) or not np.all(steps == steps[0]):
            raise ValueError("Need at least two regularly sampled time coordinates")
        time_var = nc["time"]
        dates = num2date(
            times[:2], time_var.units, calendar=getattr(time_var, "calendar", "standard")
        )
        cadence = (dates[1] - dates[0]).total_seconds() / 3600.0
        if not math.isfinite(cadence) or cadence <= 0:
            raise ValueError("Invalid measured sampling cadence")
        for name in ("INT_UTC_timeInit", "INT_UTC_timeEnd"):
            if nc[name].units != time_var.units:
                raise ValueError("Object and coordinate time units differ")
        expected = (
            (integers(nc["INT_UTC_timeEnd"]) - integers(nc["INT_UTC_timeInit"]))
            * cadence
            / steps[0]
        )
        if not np.allclose(elapsed, expected, rtol=1e-6, atol=1e-6):
            raise ValueError("Stored duration differs from last minus first observation")
        observations = integers(nc["INT_obs_count"])
        if np.any(observations < 1) or not np.array_equal(elapsed == 0, observations == 1):
            raise ValueError("Zero duration and single-observation counts disagree")
        if not np.allclose(elapsed / cadence, np.rint(elapsed / cadence), rtol=1e-6, atol=1e-6):
            raise ValueError("Durations are not on the measured sampling lattice")
        return {
            "threshold_K": threshold,
            "duration_h": elapsed + cadence if definition == "covered" else elapsed,
            "elapsed_h": elapsed,
            "region": region,
            "window_boundary": boundary,
            "cadence_h": cadence,
            "components": len(ids),
            "time_start": str(nc.time_coverage_start),
            "time_end": str(nc.time_coverage_end),
            "source_netcdf": str(getattr(nc, "source_netcdf", "")),
            "source_variable": str(getattr(nc, "source_variable", "")),
            "processing": str(getattr(nc, "processing", "")),
            "connectivity": int(getattr(nc, "connectivity", 0)),
        }


def histogram_edges(records: list[dict[str, Any]], bins: int, scale: str) -> np.ndarray:
    cadence = records[0]["cadence_h"]
    if not all(np.isclose(r["cadence_h"], cadence) for r in records):
        raise ValueError("The two runs have different sampling cadences")
    positive = [r["duration_h"][(r["duration_h"] > 0) & (r["region"] > 0)] for r in records]
    nonempty = [values for values in positive if len(values)]
    if not nonempty:
        raise ValueError("No classified components have a positive plotted duration")
    minimum, maximum = min(v.min() for v in nonempty), max(v.max() for v in nonempty)
    lower = max(cadence / 2, minimum - cadence / 2)
    upper = cadence * 2 ** math.ceil(math.log2(maximum / cadence)) + cadence / 2
    if scale == "linear":
        raw = np.linspace(lower, upper, bins + 1)
    else:
        raw = np.geomspace(lower, upper, bins + 1)
    # Place edges between native durations and merge redundant narrow bins.
    # Otherwise log bins narrower than 30 min manufacture empty gaps between
    # 0.5 h, 1 h, 1.5 h, etc. No duration is shifted or rounded for counting.
    return np.unique((np.rint(raw / cadence - 0.5) + 0.5) * cadence)


def histograms(
    records: list[dict[str, Any]], edges: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves, summaries = [], []
    for record in records:
        info = {
            key: record[key]
            for key in ("threshold_K", "components", "cadence_h", "time_start", "time_end")
        }
        info["groups"] = {}
        for code, name in enumerate(REGIONS):
            belongs = record["region"] == code
            values, elapsed = record["duration_h"][belongs], record["elapsed_h"][belongs]
            count = np.histogram(values[values > 0], bins=edges)[0]
            total = int(count.sum())
            density = count / (total * np.diff(edges)) if total else np.zeros(len(count))
            info["groups"][name] = {
                "components": len(values),
                "elapsed_zero_components": int(np.count_nonzero(elapsed == 0)),
                "elapsed_zero_fraction": float(np.mean(elapsed == 0)) if len(elapsed) else None,
                "positive_plotted_components": total,
                "plotted_zero_components": int(np.count_nonzero(values == 0)),
                "positive_median_h": float(np.median(values[values > 0])) if total else None,
                "maximum_plotted_duration_h": float(values.max()) if len(values) else None,
                "window_boundary_components": int(
                    np.count_nonzero(record["window_boundary"][belongs])
                ),
                "pdf_integral": float(np.sum(density * np.diff(edges))),
            }
            if code:
                if not total:
                    raise ValueError(
                        f"No positive durations for {name} {record['threshold_K']:g} K"
                    )
                curves.append(
                    {
                        "threshold_K": record["threshold_K"],
                        "region": name,
                        "counts": count,
                        "pdf": density,
                    }
                )
        summaries.append(info)
    return curves, summaries


def plot_pdf(
    path: Path, curves: list[dict[str, Any]], edges: np.ndarray, args: argparse.Namespace
) -> None:
    centres = (edges[:-1] + edges[1:]) / 2
    colours = dict(zip(sorted(args.thresholds), ("#c63e82", "#969000")))
    fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")
    temporary = unique_atomic_path(path, suffix=".png")
    try:
        for curve in curves:
            density = curve["pdf"]
            displayed = (
                np.where(density > 0, density, np.nan) if args.scale == "loglog" else density
            )
            ax.plot(
                centres,
                displayed,
                color=colours[curve["threshold_K"]],
                lw=2.0,
                ls="-" if curve["region"] == "Land" else "--",
                marker="o",
                markersize=2.3,
                label=f"{curve['region']} {curve['threshold_K']:g} K",
            )
        if args.scale == "loglog":
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set(xlim=(edges[0], edges[-1]), xlabel="Duration [h]", ylabel=r"PDF [h$^{-1}$]")
        title = (
            "Cloud-duration distributions"
            if args.duration_definition == "covered"
            else "Cloud-duration distributions (duration > 0 h)"
        )
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(alpha=0.18, which="major")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
        fig.savefig(temporary, format="png", dpi=160, facecolor="white")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def write_histograms(path: Path, curves: list[dict[str, Any]], edges: np.ndarray) -> None:
    temporary = unique_atomic_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "threshold_K",
                    "region",
                    "duration_lower_h",
                    "duration_upper_h",
                    "components",
                    "pdf_per_h",
                ]
            )
            for curve in curves:
                for i, count in enumerate(curve["counts"]):
                    writer.writerow(
                        [
                            curve["threshold_K"],
                            curve["region"],
                            edges[i],
                            edges[i + 1],
                            int(count),
                            float(curve["pdf"][i]),
                        ]
                    )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    class_summary_path = args.classification_dir / "summary.json"
    class_summary_id = identity(class_summary_path)
    classes = json.loads(class_summary_path.read_text(encoding="utf-8"))
    if not classes.get("complete") or classes.get("classification") != "peak_centroid":
        raise ValueError(
            "Need completed Amax-centroid classifications; run amax_land_ocean_pdf.py first"
        )
    paths = [
        args.stats_root / f"Tb_seed_{seed:g}K" / "CC_stats_indexed.nc" for seed in args.thresholds
    ]
    sources = [identity(path) for path in paths]
    for source in sources:
        if source not in classes["statistics_sources"]:
            raise ValueError("Geographic classes were made from different/modified statistics")
    cache_paths = [
        args.classification_dir / f"classified_components_{seed:g}K.npz" for seed in args.thresholds
    ]
    cache_ids = [identity(path) for path in cache_paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with destination_lease(args.output_dir / "duration-pdf-diagnostic"):
        records = [
            read_duration(
                path, seed, cache, classes["classification_signature"], args.duration_definition
            )
            for seed, path, cache in zip(args.thresholds, paths, cache_paths)
        ]
        for field in (
            "time_start",
            "time_end",
            "source_netcdf",
            "source_variable",
            "processing",
            "connectivity",
        ):
            if records[0][field] != records[1][field]:
                raise ValueError(f"The runs differ in {field}; cannot compare like-for-like")
        if (
            any(
                identity(p) != original
                for p, original in zip(paths + cache_paths, sources + cache_ids)
            )
            or identity(class_summary_path) != class_summary_id
        ):
            raise RuntimeError("Statistics/classifications changed; figure not published")
        edges = histogram_edges(records, args.bins, args.scale)
        curves, groups = histograms(records, edges)
        provenance = {
            "schema": SCHEMA,
            "statistics_sources": sources,
            "classification_sources": cache_ids,
            "classification": "peak_centroid",
            "classification_signature": classes["classification_signature"],
            "geography_rule": classes["geography_rule"],
            "land": classes["land"],
            "basins": classes["basins"],
            "thresholds_K": args.thresholds,
            "duration_definition": args.duration_definition,
            "duration_rule": (
                "INT_duration = last observation - first observation"
                if args.duration_definition == "elapsed"
                else "INT_duration + one measured sampling interval"
            ),
            "normalization": "one object one count; conditional on positive plotted duration, separately per region/threshold",
            "zero_duration_rule": (
                "stored zeros excluded from log PDF and counted in summary"
                if args.duration_definition == "elapsed"
                else "stored zeros retained as one sampling interval; recorded NetCDF durations not changed"
            ),
            "duration_units": "h",
            "density_units": "h-1",
            "plot_scale": args.scale,
            "bin_rule": "common edges snapped between native sampled durations; redundant narrow bins merged",
            "requested_bins": args.bins,
            "actual_bins": len(edges) - 1,
            "bin_edges_h": edges.tolist(),
            "window_rule": "observed-window duration; boundary-touching components retained and counted",
        }
        plot_pdf(args.output_dir / "duration_land_ocean_pdf.png", curves, edges, args)
        write_histograms(args.output_dir / "duration_histograms.csv", curves, edges)
        summary = {
            **provenance,
            "experiment_signature": signature(provenance),
            "groups": groups,
            "complete": True,
            "full_volume_reads": 0,
            "figure_size_inches": [args.width, args.height],
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(args.output_dir / "summary.json", summary)
        LOGGER.info(
            "Complete in %.1fs: %s",
            summary["elapsed_seconds"],
            args.output_dir / "duration_land_ocean_pdf.png",
        )
        return summary


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
