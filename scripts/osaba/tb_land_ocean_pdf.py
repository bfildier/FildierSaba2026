#!/usr/bin/env python3
"""Stream a Kelvin mosaic into an area-weighted land/ocean PDF.

No tracking, interpolation, temperature resampling, smoothing or KDE is applied.
Geographical masks classify pixel centres; monthly data are read once, one frame
at a time. Run --download-geodata once if the required local polygons are absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from cartopy import config as cartopy_config
from cartopy.io.shapereader import Reader
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from netCDF4 import Dataset, num2date

from reconcile.io.runtime import destination_lease, unique_atomic_path
from reconcile.io.netcdf import _fsync_directory
from reconcile.tracking.statistics import area_per_row_km2

LOGGER = logging.getLogger("tb_land_ocean_pdf")
SCHEMA = "tb_geographic_conditional_pdf_v1"
GROUPS = ("Unclassified", "Land", "Pacific", "Atlantic", "Indian", "Other seas")
COLORS = ("#eeeeee", "#ad6536", "#2878b5", "#45a778", "#9865ad", "#8496a5")
OCEAN_NAMES = {
    "North Pacific Ocean": 2,
    "South Pacific Ocean": 2,
    "North Atlantic Ocean": 3,
    "South Atlantic Ocean": 3,
    "Indian Ocean": 4,
    "South China and Easter Archipelagic Seas": 5,  # Exact WFS name (sic).
    "South China and Eastern Archipelagic Seas": 5,
    "Mediterranean Region": 5,
    "Baltic Sea": 5,
    "Arctic Ocean": 5,
    "Southern Ocean": 5,
}
BASIN_URL = (
    "https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0"
    "&request=GetFeature&typeName=MarineRegions%3Agoas&outputFormat=shape-zip"
    "&srsName=EPSG%3A4326"
)
LAND_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-netcdf", type=Path, required=True)
    parser.add_argument("--variable", default="Harmonized_irBT_mosaic")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("test_simple/outputs_tb_land_ocean_pdf")
    )
    parser.add_argument("--land-shapefile", type=Path)
    parser.add_argument("--basin-shapefile", type=Path)
    parser.add_argument("--download-geodata", action="store_true")
    parser.add_argument("--tb-min", type=float, default=170.0)
    parser.add_argument("--tb-max", type=float, default=245.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, help="Leading subset for a smoke test only.")
    parser.add_argument("--row-block", type=int, default=128, help="Histogram/mask working rows.")
    parser.add_argument(
        "--figure-note", default="", help="Optional scientific note saved in summary.json only."
    )
    parser.add_argument(
        "--source-tb-max",
        type=float,
        help="Known upstream upper Tb bound saved in metadata only; does not limit the plot.",
    )
    args = parser.parse_args(argv)
    if not np.isfinite([args.tb_min, args.tb_max, args.bin_width]).all():
        parser.error("temperature bounds and bin width must be finite")
    if args.tb_min <= 0 or args.tb_max <= args.tb_min or args.bin_width <= 0:
        parser.error("require 0 < tb-min < tb-max and bin-width > 0")
    n_bins = (args.tb_max - args.tb_min) / args.bin_width
    if not np.isclose(n_bins, round(n_bins)) or not 1 <= n_bins <= 10000:
        parser.error("the temperature interval must contain 1..10000 whole bins")
    if args.row_block < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("row-block and max-frames must be positive")
    if args.source_tb_max is not None and (
        not np.isfinite(args.source_tb_max) or args.source_tb_max <= args.tb_min
    ):
        parser.error("source-tb-max must be finite and greater than tb-min")
    return args


def identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = unique_atomic_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz(path: Path, **arrays: Any) -> None:
    temporary = unique_atomic_path(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def download_shapes(url: str, directory: Path) -> Path:
    """Download public vector data into this diagnostic's own output directory."""
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "source.zip"
    if not archive.exists():
        temporary = unique_atomic_path(archive)
        try:
            LOGGER.info("Downloading geographical polygons: %s", url)
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            with zipfile.ZipFile(temporary) as bundle:
                if sum(info.file_size for info in bundle.infolist()) > 2 * 1024**3:
                    raise ValueError("Unexpectedly large geographical archive")
                if not any(info.filename.lower().endswith(".shp") for info in bundle.infolist()):
                    raise ValueError("Downloaded archive has no shapefile")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = Path(info.filename)
            if name.suffix.lower() not in {".shp", ".shx", ".dbf", ".prj", ".cpg"}:
                continue
            target = directory / name.name  # Never trust paths in remote archives.
            if not target.exists():
                with bundle.open(info) as src, target.open("xb") as out:
                    shutil.copyfileobj(src, out)
    candidates = list(directory.glob("*.shp"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one shapefile in {directory}: {candidates}")
    return candidates[0]


def shape_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "files_sha256": {
            part.name: sha256_file(part)
            for suffix in (".shp", ".shx", ".dbf", ".prj")
            if (part := path.with_suffix(suffix)).exists()
        },
    }


def resolve_shapes(args: argparse.Namespace) -> tuple[Path, Path]:
    geodata = args.output_dir / "geodata"
    land = args.land_shapefile
    if land is None:
        cached = (
            Path(cartopy_config["data_dir"]) / "shapefiles/natural_earth/physical/ne_10m_land.shp"
        )
        land = cached if cached.exists() else geodata / "land/ne_10m_land.shp"
    basins = args.basin_shapefile
    if basins is None:
        existing = list((geodata / "basins").glob("*.shp"))
        basins = existing[0] if len(existing) == 1 else geodata / "basins/goas.shp"
    for kind, path, url in (("land", land, LAND_URL), ("basins", basins, BASIN_URL)):
        if not path.exists():
            if not args.download_geodata:
                raise FileNotFoundError(f"Missing {kind} polygons: {path}; use --download-geodata")
            downloaded = download_shapes(url, geodata / kind)
            if kind == "land":
                land = downloaded
            else:
                basins = downloaded
    return land, basins


def read_geometries(land_path: Path, basin_path: Path) -> tuple[Any, dict[int, list[Any]]]:
    for path in (land_path, basin_path):
        crs = path.with_suffix(".prj").read_text(encoding="utf-8").upper()
        if "GEOGCS" not in crs or not ("WGS_1984" in crs or "WGS 84" in crs):
            raise ValueError(f"Expected WGS84 geographical shapefile: {path}")
    reader = Reader(str(land_path))
    try:
        land = shapely.union_all(list(reader.geometries()))
    finally:
        reader.close()
    if land.is_empty:
        raise ValueError("Land polygon dataset is empty")
    basins: dict[int, list[Any]] = {group: [] for group in range(2, 6)}
    names = set()
    reader = Reader(str(basin_path))
    try:
        for record in reader.records():
            name = str(record.attributes.get("name", ""))
            if name not in OCEAN_NAMES:
                raise ValueError(f"Unrecognized Marine Regions polygon: {name!r}")
            names.add(name)
            basins[OCEAN_NAMES[name]].append(record.geometry)
    finally:
        reader.close()
    if not all(basins[group] for group in (2, 3, 4)):
        raise ValueError("Polygons must include Pacific, Atlantic and Indian oceans")
    LOGGER.info("Loaded Marine Regions areas: %s", sorted(names))
    return land, basins


def rasterize_regions(
    lat: np.ndarray, lon: np.ndarray, land: Any, basins: dict[int, list[Any]], row_block: int
) -> np.ndarray:
    """Assign centres, with land priority and strict detection of basin overlaps."""
    # Use the polygons' native -180..180 longitudes; do not cut basins at fixed meridians.
    normalized_lon = (np.asarray(lon, dtype=np.float64) + 180) % 360 - 180
    regions: np.ndarray = np.zeros((len(lat), len(lon)), dtype=np.uint8)
    # Rasterize the constituent polygons directly: dissolving millions of coastal
    # vertices is needlessly expensive and does not change centre membership.
    geometries = {
        group: [g for g in items if g.bounds[1] <= np.max(lat) and g.bounds[3] >= np.min(lat)]
        for group, items in basins.items()
    }
    for geometry in (land, *(g for items in geometries.values() for g in items)):
        if not shapely.is_valid(geometry):
            raise ValueError("Invalid geographical polygon; repair the source explicitly")
        shapely.prepare(geometry)
    for start in range(0, len(lat), row_block):
        block = regions[start : start + row_block]
        y = np.asarray(lat[start : start + row_block], dtype=np.float64)[:, None]
        x = normalized_lon[None, :]
        block[shapely.intersects_xy(land, x, y)] = 1
        for group, items in geometries.items():
            for geometry in items:
                inside = shapely.contains_xy(geometry, x, y) & (block != 1)
                if np.any(inside & (block >= 2) & (block != group)):
                    raise ValueError("Overlapping ocean-basin polygon interiors at pixel centres")
                block[inside] = group
        LOGGER.info("Geographic mask rows %d/%d", min(start + row_block, len(lat)), len(lat))
    return regions


def new_accumulator(n_bins: int) -> dict[str, np.ndarray]:
    return {
        "counts": np.zeros((len(GROUPS), n_bins), dtype=np.int64),
        "area": np.zeros((len(GROUPS), n_bins), dtype=np.float64),
        "finite_counts": np.zeros(len(GROUPS), dtype=np.int64),
        "finite_area": np.zeros(len(GROUPS), dtype=np.float64),
    }


def accumulate_frame(
    frame: np.ndarray,
    regions: np.ndarray,
    row_area: np.ndarray,
    edges: np.ndarray,
    accumulator: dict[str, np.ndarray],
    row_block: int,
) -> None:
    """Accumulate exact histogram bins; the upper endpoint belongs to the last bin."""
    n_bins = len(edges) - 1
    for start in range(0, frame.shape[0], row_block):
        values = frame[start : start + row_block]
        group = regions[start : start + row_block]
        weights = np.broadcast_to(row_area[start : start + row_block, None], values.shape)
        finite = np.isfinite(values)
        accumulator["finite_counts"] += np.bincount(group[finite], minlength=len(GROUPS))
        accumulator["finite_area"] += np.bincount(
            group[finite], weights=weights[finite], minlength=len(GROUPS)
        )
        selected = finite & (values >= edges[0]) & (values <= edges[-1])
        indices = np.searchsorted(edges, values[selected], side="right") - 1
        indices = np.minimum(indices, n_bins - 1)
        joint = group[selected].astype(np.int64) * n_bins + indices
        accumulator["counts"] += np.bincount(joint, minlength=len(GROUPS) * n_bins).reshape(
            len(GROUPS), n_bins
        )
        accumulator["area"] += np.bincount(
            joint, weights=weights[selected], minlength=len(GROUPS) * n_bins
        ).reshape(len(GROUPS), n_bins)


def conditional_pdf(histogram: np.ndarray, edges: np.ndarray) -> np.ndarray:
    total = histogram.sum(axis=-1, keepdims=True)
    return np.divide(
        histogram,
        total * np.diff(edges),
        out=np.zeros_like(histogram, dtype=np.float64),
        where=total > 0,
    )


def combined_groups(accumulator: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    result = {
        name: {key: value[group] for key, value in accumulator.items()}
        for group, name in enumerate(GROUPS)
    }
    result["Ocean"] = {key: value[2:].sum(axis=0) for key, value in accumulator.items()}
    return result


def plot_pdf(
    path: Path,
    groups: dict[str, dict[str, Any]],
    edges: np.ndarray,
    source_tb_max: float | None = None,
) -> None:
    """One clean land/ocean panel; scientific details stay in the sidecars."""
    centres = (edges[:-1] + edges[1:]) / 2
    # Display the requested bin interval, including bins with no observations.
    # Upstream thermal retention is metadata, not an axis limit.
    display_max = float(edges[-1])
    if display_max <= edges[0]:
        raise ValueError("Displayed upper Tb bound must be greater than the lower bound")
    fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    peak = 0.0
    for name in ("Land", "Ocean"):
        histogram = groups[name]["area"]
        if histogram.sum() == 0:
            continue
        pdf = conditional_pdf(histogram, edges)
        peak = max(peak, float(pdf.max()))
        ax.plot(
            centres,
            pdf,
            color="#243f5b" if name == "Ocean" else COLORS[1],
            lw=2.4,
            ls="--" if name == "Ocean" else "-",
            label=name,
        )
    ax.set(
        xlim=(edges[0], display_max),
        ylim=(0, peak * 1.1 if peak else 1),
        xlabel=r"$T_b$ [K]",
        ylabel=r"PDF [K$^{-1}$]",
    )
    ticks = ax.get_xticks()
    ax.set_xticks(
        np.unique(np.r_[ticks[(ticks >= edges[0]) & (ticks <= display_max)], edges[0], display_max])
    )
    ax.set_title("Tropical brightness-temperature distributions", fontsize=14, pad=12)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_mask(path: Path, regions: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
    order_y, order_x = np.argsort(lat), np.argsort((lon + 180) % 360 - 180)
    # Display a decimated map only; the numerical analysis uses every original pixel.
    sy = order_y[:: max(1, len(lat) // 600)]
    sx = order_x[:: max(1, len(lon) // 2400)]
    fig, ax = plt.subplots(figsize=(15, 4.2), layout="constrained")
    ax.pcolormesh(
        (lon[sx] + 180) % 360 - 180,
        lat[sy],
        regions[np.ix_(sy, sx)],
        cmap=ListedColormap(COLORS),
        norm=BoundaryNorm(np.arange(-0.5, len(GROUPS) + 0.5), len(GROUPS)),
        shading="nearest",
        rasterized=True,
    )
    ax.set(xlabel="Longitude [degrees east]", ylabel="Latitude [degrees north]")
    ax.set_title("Geographic masks used for the Tb distributions (pixel-centre classification)")
    ax.legend(
        handles=[Patch(color=color, label=name) for name, color in zip(GROUPS, COLORS)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=6,
        frameon=False,
    )
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def validate_coordinates(lat: np.ndarray, lon: np.ndarray, times: np.ndarray) -> None:
    for name, coordinate, limit in (("latitude", lat, 90), ("longitude", lon, 360)):
        differences = np.diff(coordinate.astype(np.float64))
        if coordinate.ndim != 1 or len(coordinate) < 2 or not np.isfinite(coordinate).all():
            raise ValueError(f"Expected finite 1D {name} coordinates with at least two points")
        if np.any(np.abs(coordinate) > limit):
            raise ValueError(f"Invalid {name} range")
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(f"{name} must be strictly monotonic")
        if not np.allclose(differences, np.median(differences), rtol=2e-3, atol=2e-5):
            raise ValueError(f"This diagnostic requires regular {name} spacing")
    steps = np.diff(times.astype(np.float64))
    if len(np.unique((lon + 180) % 360)) != len(lon):
        raise ValueError("Duplicate longitudes after wrapping (e.g. both -180 and +180)")
    if not len(times) or not np.isfinite(times).all():
        raise ValueError("Expected finite time coordinates")
    if len(steps) and (np.any(steps <= 0) or not np.allclose(steps, steps[0])):
        raise ValueError("Expected a strictly increasing, regularly sampled time axis")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with destination_lease(args.output_dir / "tb-pdf-diagnostic"):
        land_path, basin_path = resolve_shapes(args)
        source = identity(args.input_netcdf)
        with Dataset(args.input_netcdf) as nc:
            bt = nc.variables[args.variable]
            if bt.dimensions != ("time", "latitude", "longitude"):
                raise ValueError(f"Expected time,latitude,longitude, got {bt.dimensions}")
            if str(getattr(bt, "units", "")).strip().lower() not in {"k", "kelvin", "kelvins"}:
                raise ValueError("Input temperatures must explicitly declare Kelvin units")
            lat = np.asarray(nc.variables["latitude"][:], dtype=np.float64)
            lon = np.asarray(nc.variables["longitude"][:], dtype=np.float64)
            time_var = nc.variables["time"]
            times = np.asarray(time_var[:], dtype=np.float64)
            validate_coordinates(lat, lon, times)
            n_frames = min(len(times), args.max_frames or len(times))
            calendar = getattr(time_var, "calendar", "standard")
            dates: Any = num2date(times[[0, n_frames - 1]], time_var.units, calendar=calendar)
            edges = np.linspace(
                args.tb_min,
                args.tb_max,
                round((args.tb_max - args.tb_min) / args.bin_width) + 1,
            )
            row_area = area_per_row_km2(lat, lon)
            provenance = {
                "schema": SCHEMA,
                "source": source,
                "variable": args.variable,
                "shape": list(bt.shape),
                "frames": n_frames,
                "time_start": dates[0].isoformat(),
                "time_end": dates[-1].isoformat(),
                "time_units": time_var.units,
                "time_calendar": calendar,
                "latitude_range": [float(lat.min()), float(lat.max())],
                "bin_edges_K": edges.tolist(),
                "land": shape_provenance(land_path),
                "basins": shape_provenance(basin_path),
                "basin_source": "https://doi.org/10.14284/542",
                "land_source": "https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/",
                "ocean_name_mapping": {name: GROUPS[group] for name, group in OCEAN_NAMES.items()},
                "mask_rule": "pixel centre; land boundary included; ocean interior only; land priority",
                "sample_rule": "all finite mosaic values, including preprocessing/interpolation",
                "normalization": "area weighted, conditional on closed displayed Tb interval, per region",
                "weight_units": "km2_observations (regular time spacing; not unique geographical area)",
                "source_scientific_signature": str(getattr(bt, "cache_scientific_signature", "")),
            }
            experiment = signature(provenance)
            mask_key = signature(
                [provenance["land"], provenance["basins"], lat.tolist(), lon.tolist(), SCHEMA]
            )
            mask_path = args.output_dir / "geographic_mask.npz"
            if mask_path.exists():
                with np.load(mask_path, allow_pickle=False) as cached:
                    if str(cached["signature"].item()) != mask_key:
                        raise ValueError(
                            "Different geographical grid/sources: use a new output directory"
                        )
                    regions = cached["regions"].copy()
            else:
                land, basins = read_geometries(land_path, basin_path)
                regions = rasterize_regions(lat, lon, land, basins, args.row_block)
                write_npz(
                    mask_path, signature=mask_key, regions=regions, latitude=lat, longitude=lon
                )
                del land, basins
            if regions.shape != (len(lat), len(lon)) or np.any(regions >= len(GROUPS)):
                raise ValueError("Invalid cached geographical mask")
            checkpoint = args.output_dir / "histogram_checkpoint.npz"
            accumulator = new_accumulator(len(edges) - 1)
            next_frame = 0
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as cached:
                    if str(cached["signature"].item()) != experiment:
                        raise ValueError("Different experiment/input: use a new output directory")
                    next_frame = int(cached["next_frame"].item())
                    accumulator = {key: cached[key].copy() for key in accumulator}
                if not 0 <= next_frame <= n_frames:
                    raise ValueError("Invalid histogram checkpoint frame")
                LOGGER.info("Reusing %d/%d accumulated frames", next_frame, n_frames)
            bt.set_var_chunk_cache(size=64 * 1024**2, nelems=1009, preemption=0.75)
            for frame_index in range(next_frame, n_frames):
                sample = bt[frame_index]
                if sample.dtype.kind not in "f":
                    sample = sample.astype(np.float64)
                frame = np.asarray(np.ma.filled(sample, np.nan))
                accumulate_frame(frame, regions, row_area, edges, accumulator, args.row_block)
                if (frame_index + 1) % 24 == 0 or frame_index + 1 == n_frames:
                    if identity(args.input_netcdf) != source:
                        raise RuntimeError(
                            "Input mosaic changed during analysis; results not published"
                        )
                    write_npz(
                        checkpoint,
                        signature=experiment,
                        next_frame=frame_index + 1,
                        edges=edges,
                        **accumulator,
                    )
                    elapsed = time.perf_counter() - start
                    LOGGER.info("Frames %d/%d; elapsed %.1fs", frame_index + 1, n_frames, elapsed)
        if identity(args.input_netcdf) != source:
            raise RuntimeError("Input mosaic changed; results not published")
        groups = combined_groups(accumulator)
        summaries = {}
        for name, data in groups.items():
            finite_area = float(data["finite_area"])
            selected_area = float(data["area"].sum())
            summaries[name] = {
                "finite_pixel_observations": int(data["finite_counts"]),
                "selected_pixel_observations": int(data["counts"].sum()),
                "finite_weight_km2_observations": finite_area,
                "selected_weight_km2_observations": selected_area,
                "selected_fraction_of_finite_area": (
                    selected_area / finite_area if finite_area else None
                ),
                "conditional_pdf_integral": float(
                    np.sum(conditional_pdf(data["area"], edges) * np.diff(edges))
                ),
            }
        summary = {
            **provenance,
            "experiment_signature": experiment,
            "groups": summaries,
            "geographic_pixel_counts": {
                name: int(np.count_nonzero(regions == i)) for i, name in enumerate(GROUPS)
            },
            "elapsed_seconds": time.perf_counter() - start,
            "complete": True,
            "figure_note": args.figure_note,
            "source_retained_tb_max_K_annotation": args.source_tb_max,
            "cached_frames_reused": next_frame,
            "frames_read_this_invocation": n_frames - next_frame,
        }
        plot_pdf(
            args.output_dir / "tb_land_ocean_pdf.png",
            groups,
            edges,
            args.source_tb_max,
        )
        plot_mask(args.output_dir / "geographic_mask.png", regions, lat, lon)
        csv_path = args.output_dir / "tb_histograms.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "region",
                    "tb_lower_K",
                    "tb_upper_K",
                    "pixel_observations",
                    "area_weight_km2_observations",
                    "conditional_area_pdf_K_inverse",
                ]
            )
            for name, data in groups.items():
                pdf = conditional_pdf(data["area"], edges)
                for i in range(len(edges) - 1):
                    writer.writerow(
                        [
                            name,
                            edges[i],
                            edges[i + 1],
                            int(data["counts"][i]),
                            float(data["area"][i]),
                            float(pdf[i]),
                        ]
                    )
        write_json(args.output_dir / "summary.json", summary)
        LOGGER.info("Complete: %s", args.output_dir / "tb_land_ocean_pdf.png")
        return summary


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
