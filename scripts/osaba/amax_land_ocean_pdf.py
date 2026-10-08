#!/usr/bin/env python3
"""Plot four object-size PDFs from completed single-threshold statistics.

One object contributes one Amax, in km2, over the observed time window. Land is
solid and Ocean dashed; 210 K is pink and 220 K olive. Classification defaults
to the centroid at the first observation attaining the stored maximum area.
Both axes and the common histogram bins are logarithmic by default.
No labels, Tb mosaics, QC volumes, tracking or statistics are recomputed.

From the repository root:
    .venv/bin/python test_simple/amax_land_ocean_pdf.py

Inputs/classifications are cached independently of the histogram/plot settings.
All outputs belong to this standalone diagnostic, not the production software.
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
import shapely
from netCDF4 import Dataset

from reconcile.io.netcdf import _fsync_directory
from reconcile.io.runtime import destination_lease, unique_atomic_path
from tb_land_ocean_pdf import (
    identity,
    read_geometries,
    shape_provenance,
    signature,
    write_json,
    write_npz,
)

LOGGER = logging.getLogger("amax_land_ocean_pdf")
SCHEMA = "object_amax_geographic_pdf_v1"
REGIONS = ("Unclassified", "Land", "Ocean")
DEFAULT_STATS = Path("test_simple/outputs_mosaic_seed_labeling_satellite_1_month")
DEFAULT_GEOGRAPHY = Path("test_simple/outputs_tb_land_ocean_pdf_satellite_20160810_20160910")
DEFAULT_OUTPUT = Path("test_simple/outputs_amax_land_ocean_pdf_satellite_20160810_20160910")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-root", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--thresholds", type=float, nargs=2, default=[210.0, 220.0])
    parser.add_argument("--geography-dir", type=Path, default=DEFAULT_GEOGRAPHY)
    parser.add_argument("--land-shapefile", type=Path)
    parser.add_argument("--basin-shapefile", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--classification",
        choices=("peak_centroid", "initial_centroid"),
        default="peak_centroid",
    )
    parser.add_argument(
        "--bins", type=int, default=30, help="Common histogram bins for all curves."
    )
    parser.add_argument("--area-min", type=float, help="Optional lower displayed bound, km2.")
    parser.add_argument("--area-max", type=float, help="Optional upper displayed bound, km2.")
    parser.add_argument(
        "--scale",
        choices=("loglog", "linear"),
        default="loglog",
        help="Log-spaced bins/log axes, or equally spaced bins/linear axes.",
    )
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--width", type=float, default=6.0)
    parser.add_argument("--height", type=float, default=4.0)
    args = parser.parse_args(argv)
    if any(not math.isfinite(t) or t <= 0 for t in args.thresholds):
        parser.error("thresholds must be finite positive Kelvin values")
    if args.thresholds[0] == args.thresholds[1]:
        parser.error("thresholds must differ")
    if not 2 <= args.bins <= 1000 or args.batch_size < 1:
        parser.error("require 2..1000 bins and a positive batch-size")
    for key in ("width", "height", "area_min", "area_max"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.area_min is not None and args.area_max is not None and args.area_max <= args.area_min:
        parser.error("area-max must exceed area-min")
    return args


def floats(variable: Any, selection: Any = slice(None)) -> np.ndarray:
    return np.asarray(np.ma.filled(variable[selection].astype(np.float64), np.nan))


def integers(variable: Any, selection: Any = slice(None)) -> np.ndarray:
    values = variable[selection]
    if variable.dtype.kind not in "iu" or np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"Expected unmasked integer values in {variable.name}")
    return np.asarray(values, dtype=np.int64)


def read_components(path: Path, threshold: float, rule: str, batch_size: int) -> dict[str, Any]:
    """Read only object/ragged vectors; never index the 3D QC variable."""
    with Dataset(path) as nc:
        if not all(
            int(getattr(nc, name, 0)) == 1 for name in ("statistics_complete", "reconcile_complete")
        ):
            raise ValueError(f"Statistics are incomplete: {path}")
        if not np.isclose(float(getattr(nc, "Tb_seed_K", np.nan)), threshold):
            raise ValueError(f"Wrong seed threshold in {path}")
        required = {
            "INT_CCnumber": ("CC",),
            "INT_surfmaxkm2_235K": ("CC",),
            "time": ("time",),
            "INT_UTC_timeInit": ("CC",),
            "INT_UTC_timeEnd": ("CC",),
        }
        if rule == "peak_centroid":
            required.update(
                {
                    name: ("obs",)
                    for name in (
                        "obs_parent_index",
                        "obs_time_index",
                        "LC_surfkm2_235K",
                        "LC_lat",
                        "LC_lon",
                    )
                }
            )
        else:
            required.update({"INT_latInit": ("CC",), "INT_lonInit": ("CC",)})
        for name, dimensions in required.items():
            if name not in nc.variables or nc[name].dimensions != dimensions:
                raise ValueError(f"Missing/misdimensioned statistics variable {name} in {path}")
        area_var = nc["INT_surfmaxkm2_235K"]
        if getattr(area_var, "units", "") not in ("km2", "km^2", "km²"):
            raise ValueError("Amax must explicitly declare square kilometres")
        # The 235K suffix is a legacy name, not the threshold of these diagnostics.
        if not np.isclose(float(getattr(area_var, "threshold_tb_K", np.nan)), threshold):
            raise ValueError("Amax variable threshold does not match the requested run")
        lat_name, lon_name = (
            ("LC_lat", "LC_lon") if rule == "peak_centroid" else ("INT_latInit", "INT_lonInit")
        )
        if nc[lat_name].units != "degrees_north" or nc[lon_name].units != "degrees_east":
            raise ValueError("Centroid coordinates must declare geographical degree units")
        if rule == "peak_centroid":
            snapshot_area = nc["LC_surfkm2_235K"]
            if snapshot_area.units != area_var.units or not np.isclose(
                float(getattr(snapshot_area, "threshold_tb_K", np.nan)), threshold
            ):
                raise ValueError("Snapshot area units/threshold differ from the stored Amax")
        ids, area = integers(nc["INT_CCnumber"]), floats(area_var)
        n = len(ids)
        if len(np.unique(ids)) != n or np.any(ids <= 0):
            raise ValueError("Component IDs must be unique and positive")
        valid_area = np.isfinite(area) & (area > 0)
        times = integers(nc["time"])
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError("Statistics require a strictly increasing time coordinate")
        init, end = integers(nc["INT_UTC_timeInit"]), integers(nc["INT_UTC_timeEnd"])
        for name in ("INT_UTC_timeInit", "INT_UTC_timeEnd"):
            if nc[name].units != nc["time"].units:
                raise ValueError("Component/time coordinate units differ")
        boundary = (init == times[0]) | (end == times[-1])
        if rule == "initial_centroid":
            lat, lon = floats(nc["INT_latInit"]), floats(nc["INT_lonInit"])
            ref_time = np.searchsorted(times, init)
            if np.any(ref_time >= len(times)) or not np.array_equal(times[ref_time], init):
                raise ValueError("Initial component times are absent from the time coordinate")
        else:
            sentinel = np.iinfo(np.int64).max
            ref_time = np.full(n, sentinel, dtype=np.int64)
            chosen_obs = np.full(n, sentinel, dtype=np.int64)
            lat, lon = np.full(n, np.nan), np.full(n, np.nan)
            n_obs = len(nc.dimensions["obs"])
            parent_base = int(getattr(nc["obs_parent_index"], "start_index", 0))
            time_base = int(getattr(nc["obs_time_index"], "start_index", 0))
            for start in range(0, n_obs, batch_size):
                stop = min(start + batch_size, n_obs)
                selected = slice(start, stop)
                parents = integers(nc["obs_parent_index"], selected) - parent_base
                obs_times = integers(nc["obs_time_index"], selected) - time_base
                obs_area = floats(nc["LC_surfkm2_235K"], selected)
                if np.any((parents < 0) | (parents >= n)) or np.any(
                    (obs_times < 0) | (obs_times >= len(times))
                ):
                    raise ValueError("Invalid ragged parent/time index")
                matches = np.flatnonzero(valid_area[parents] & (obs_area == area[parents]))
                if matches.size:
                    p, t = parents[matches], obs_times[matches]
                    unique = np.unique(p)
                    old_time = ref_time[unique].copy()
                    np.minimum.at(ref_time, p, t)
                    chosen_obs[unique[ref_time[unique] < old_time]] = sentinel
                    candidates = matches[t == ref_time[p]]
                    np.minimum.at(chosen_obs, parents[candidates], start + candidates)
                    winners = candidates[start + candidates == chosen_obs[parents[candidates]]]
                    lat[parents[winners]] = floats(nc["LC_lat"], selected)[winners]
                    lon[parents[winners]] = floats(nc["LC_lon"], selected)[winners]
                LOGGER.info("%g K peak observations %d/%d", threshold, stop, n_obs)
            if np.any(valid_area & (chosen_obs == sentinel)):
                raise ValueError("Stored Amax does not match any ragged observation")
            ref_time[~valid_area] = -1
        return {
            "component_id": ids,
            "amax_km2": area,
            "latitude": lat,
            "longitude": lon,
            "reference_frame": ref_time,
            "window_boundary": boundary,
            "metadata": {
                "components": n,
                "valid_area_components": int(valid_area.sum()),
                "time_start": str(nc.time_coverage_start),
                "time_end": str(nc.time_coverage_end),
                "frames": len(times),
                "source_netcdf": str(getattr(nc, "source_netcdf", "")),
                "source_variable": str(getattr(nc, "source_variable", "")),
                "processing": str(getattr(nc, "processing", "")),
                "threshold_comparison": str(getattr(nc, "threshold_comparison", "")),
                "connectivity": int(getattr(nc, "connectivity", 0)),
                "diagnostic_signature": str(getattr(nc, "diagnostic_signature", "")),
                "area_variable": area_var.name,
                "invalid_centroid_components": int(
                    np.count_nonzero(
                        ~np.isfinite(lat)
                        | ~np.isfinite(lon)
                        | (np.abs(lat) > 90)
                        | (np.abs(lon) > 360)
                    )
                ),
            },
        }


def classify_centroids(
    lat: np.ndarray,
    lon: np.ndarray,
    land: Any,
    basins: dict[int, list[Any]],
    batch_size: int,
    *,
    validate_geometry: bool = True,
) -> np.ndarray:
    """Same polygon/priority rules as Tb PDF; exact continuous centroid points."""
    regions = np.zeros(len(lat), dtype=np.uint8)
    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
    x = (lon + 180) % 360 - 180
    for geometry in (land, *(g for items in basins.values() for g in items)):
        if validate_geometry and not shapely.is_valid(geometry):
            raise ValueError("Invalid geographical polygon; repair the source explicitly")
        shapely.prepare(geometry)
    for start in range(0, len(lat), batch_size):
        stop = min(start + batch_size, len(lat))
        block = regions[start:stop]
        y, xx, finite = lat[start:stop], x[start:stop], valid[start:stop]
        block[finite & shapely.intersects_xy(land, xx, y)] = 1
        for group, geometries in basins.items():
            for geometry in geometries:
                inside = finite & shapely.contains_xy(geometry, xx, y) & (block != 1)
                if np.any(inside & (block >= 2) & (block != group)):
                    raise ValueError("Ocean-basin interiors overlap at a centroid")
                block[inside] = group
        LOGGER.info("Geographic centroids %d/%d", stop, len(lat))
    regions[regions >= 2] = 2  # All four basins/other seas together are Ocean.
    return regions


def resolve_geography(args: argparse.Namespace) -> tuple[Path, Path]:
    summary_path = args.geography_dir / "summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    land = args.land_shapefile or Path(existing.get("land", {}).get("path", ""))
    basins = args.basin_shapefile or Path(existing.get("basins", {}).get("path", ""))
    if not land.is_file() or not basins.is_file():
        raise FileNotFoundError(
            "Reuse the completed Tb PDF geography directory or provide both shapefiles; "
            "this script never downloads geographical data."
        )
    return land, basins


def histogram_edges(records: list[dict[str, Any]], args: argparse.Namespace) -> np.ndarray:
    extents = []
    for record in records:
        area = record["amax_km2"]
        selected = area[np.isfinite(area) & (area > 0) & (record["region"] > 0)]
        if selected.size:
            extents.append((float(selected.min()), float(selected.max())))
    if not extents:
        raise ValueError("No positive-area classified components")
    lo, hi = min(e[0] for e in extents), max(e[1] for e in extents)
    if args.scale == "loglog":
        lower = args.area_min if args.area_min is not None else 10.0 ** math.floor(math.log10(lo))
        upper = args.area_max if args.area_max is not None else 10.0 ** math.ceil(math.log10(hi))
        if upper <= lower:
            upper = lower * 10 if args.area_max is None else upper
        if upper <= lower:
            raise ValueError("The displayed upper area bound must exceed the lower bound")
        return np.geomspace(lower, upper, args.bins + 1)
    lower = args.area_min if args.area_min is not None else 0.0
    upper = args.area_max if args.area_max is not None else hi * 1.05
    if upper <= lower:
        raise ValueError("The displayed upper area bound must exceed the lower bound")
    return np.linspace(lower, upper, args.bins + 1)


def object_histograms(
    records: list[dict[str, Any]], edges: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves, summaries = [], []
    for record in records:
        area = record["amax_km2"]
        finite = np.isfinite(area) & (area > 0)
        info = {"threshold_K": record["threshold_K"], **record["metadata"], "groups": {}}
        info["invalid_area_components"] = int((~finite).sum())
        info["window_boundary_components"] = int(record["window_boundary"].sum())
        for code, name in enumerate(REGIONS):
            values = area[finite & (record["region"] == code)]
            counts = np.histogram(values, bins=edges)[0]
            total = int(counts.sum())
            density = counts / (total * np.diff(edges)) if total else np.zeros(len(counts))
            info["groups"][name] = {
                "components": len(values),
                "displayed_components": total,
                "below_range": int(np.count_nonzero(values < edges[0])),
                "above_range": int(np.count_nonzero(values > edges[-1])),
                "pdf_integral": float(np.sum(density * np.diff(edges))),
                "minimum_km2": float(values.min()) if len(values) else None,
                "median_km2": float(np.median(values)) if len(values) else None,
                "maximum_km2": float(values.max()) if len(values) else None,
            }
            if code:
                curves.append(
                    {
                        "threshold_K": record["threshold_K"],
                        "region": name,
                        "counts": counts,
                        "pdf": density,
                    }
                )
        summaries.append(info)
    return curves, summaries


def publish_png(
    path: Path, curves: list[dict[str, Any]], edges: np.ndarray, args: argparse.Namespace
) -> None:
    centres = (
        np.sqrt(edges[:-1] * edges[1:]) if args.scale == "loglog" else (edges[:-1] + edges[1:]) / 2
    )
    colours = dict(zip(sorted(args.thresholds), ("#c63e82", "#969000")))
    fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")
    try:
        for curve in curves:
            density = curve["pdf"]
            # A zero histogram bin is undefined on a log y-axis, not a small
            # positive density. Keep those zeros in CSV and leave plotted gaps.
            displayed = (
                np.where(density > 0, density, np.nan) if args.scale == "loglog" else density
            )
            ax.plot(
                centres,
                displayed,
                lw=2.0,
                color=colours[curve["threshold_K"]],
                ls="-" if curve["region"] == "Land" else "--",
                label=f"{curve['region']} {curve['threshold_K']:g} K",
            )
        if args.scale == "loglog":
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set(xlim=(edges[0], edges[-1]), xlabel=r"$A_{\max}$ [km$^2$]", ylabel=r"PDF [km$^{-2}$]")
        ax.set_title("Tropical cloud maximum-area distributions", fontsize=12, pad=10)
        ax.grid(alpha=0.18, which="major")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
        temporary = unique_atomic_path(path, suffix=".png")
        try:
            fig.savefig(temporary, format="png", dpi=160, facecolor="white")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        plt.close(fig)


def write_histograms(path: Path, curves: list[dict[str, Any]], edges: np.ndarray) -> None:
    temporary = unique_atomic_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "threshold_K",
                    "region",
                    "area_lower_km2",
                    "area_upper_km2",
                    "components",
                    "pdf_per_km2",
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
    land_path, basin_path = resolve_geography(args)
    stats_paths = [
        args.stats_root / f"Tb_seed_{threshold:g}K" / "CC_stats_indexed.nc"
        for threshold in args.thresholds
    ]
    provenance = {
        "schema": SCHEMA,
        "statistics_sources": [identity(p) for p in stats_paths],
        "thresholds_K": args.thresholds,
        "classification": args.classification,
        "land": shape_provenance(land_path),
        "basins": shape_provenance(basin_path),
        "geography_reference_directory": str(args.geography_dir.resolve()),
        "geography_rule": "continuous centroid; land boundary included; ocean interior only; land priority",
        "peak_rule": "first time index at stored Amax; earliest ragged row breaks remaining ties; stored float32 precision",
        "sample_rule": "one component contributes one observed-window Amax; no lifetime/area filter",
        "normalization": "object-count PDF, separately normalized per threshold/region within displayed range; no area weighting",
        "area_units": "km2",
        "density_units": "km-2",
    }
    key = signature(provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with destination_lease(args.output_dir / "amax-pdf-diagnostic"):
        geometries = None
        geometries_validated = False
        for threshold, stats_path in zip(args.thresholds, stats_paths):
            cache_path = args.output_dir / f"classified_components_{threshold:g}K.npz"
            reused = False
            if cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as cached:
                    if str(cached["signature"].item()) == key:
                        record = {
                            name: cached[name].copy()
                            for name in (
                                "component_id",
                                "amax_km2",
                                "latitude",
                                "longitude",
                                "reference_frame",
                                "window_boundary",
                                "region",
                            )
                        }
                        record["metadata"] = json.loads(str(cached["metadata"].item()))
                        reused = True
                        LOGGER.info("Reusing classified components at %g K", threshold)
            if not reused:
                record = read_components(
                    stats_path, threshold, args.classification, args.batch_size
                )
                if geometries is None:
                    geometries = read_geometries(land_path, basin_path)
                record["region"] = classify_centroids(
                    record["latitude"],
                    record["longitude"],
                    *geometries,
                    args.batch_size,
                    validate_geometry=not geometries_validated,
                )
                geometries_validated = True
                write_npz(
                    cache_path,
                    signature=key,
                    metadata=json.dumps(record["metadata"], sort_keys=True),
                    **{name: value for name, value in record.items() if name != "metadata"},
                )
            record["threshold_K"] = threshold
            record["classification_reused"] = reused
            record["metadata"]["invalid_centroid_components"] = int(
                np.count_nonzero(
                    ~np.isfinite(record["latitude"])
                    | ~np.isfinite(record["longitude"])
                    | (np.abs(record["latitude"]) > 90)
                    | (np.abs(record["longitude"]) > 360)
                )
            )
            records.append(record)
        if any(
            identity(p) != expected
            for p, expected in zip(stats_paths, provenance["statistics_sources"])
        ):
            raise RuntimeError("Statistics changed during analysis; figures not published")
        if (
            shape_provenance(land_path) != provenance["land"]
            or shape_provenance(basin_path) != provenance["basins"]
        ):
            raise RuntimeError(
                "Geographical sources changed during analysis; figures not published"
            )
        reference = records[0]["metadata"]
        for record in records[1:]:
            for field in (
                "time_start",
                "time_end",
                "frames",
                "source_netcdf",
                "source_variable",
                "processing",
                "threshold_comparison",
                "connectivity",
            ):
                if record["metadata"][field] != reference[field]:
                    raise ValueError(
                        f"The two runs differ in {field}; cannot compare like-for-like"
                    )
        edges = histogram_edges(records, args)
        curves, groups = object_histograms(records, edges)
        for curve in curves:
            if curve["counts"].sum() == 0:
                raise ValueError(
                    f"No objects in the displayed range for {curve['region']} {curve['threshold_K']:g} K"
                )
        publish_png(args.output_dir / "amax_land_ocean_pdf.png", curves, edges, args)
        write_histograms(args.output_dir / "amax_histograms.csv", curves, edges)
        summary = {
            **provenance,
            "classification_signature": key,
            "bin_edges_km2": edges.tolist(),
            "plot_scale": args.scale,
            "figure_size_inches": [args.width, args.height],
            "groups": groups,
            "classification_reused": [r["classification_reused"] for r in records],
            "complete": True,
            "elapsed_seconds": time.perf_counter() - started,
            "zero_bin_display": (
                "gaps on log y-axis; all zero counts retained in CSV"
                if args.scale == "loglog"
                else "zero-density bins plotted at zero; all zero counts retained in CSV"
            ),
            "full_volume_reads": 0,
        }
        write_json(args.output_dir / "summary.json", summary)
        LOGGER.info(
            "Complete in %.1fs: %s",
            summary["elapsed_seconds"],
            args.output_dir / "amax_land_ocean_pdf.png",
        )
        return summary


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
