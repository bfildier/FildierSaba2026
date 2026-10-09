#!/usr/bin/env python3
"""Three-panel land/ocean figure: (a) Tb PDF, (b) Amax PDF, (c) duration PDF.

Merge of tb_land_ocean_pdf.py, amax_land_ocean_pdf.py and
duration_land_ocean_pdf.py. Scientific rules are unchanged:

(a) Area-weighted conditional Tb PDF from the Kelvin mosaic, streamed one frame
    at a time; pixel centres are classified Land / Ocean (land priority).
(b) One object, one Amax [km2]; classified by the centroid at the first
    observation attaining the stored Amax (or the initial centroid).
(c) One object, one duration [h] (elapsed, or covered = elapsed + 1 sampling
    interval); classified with the same Amax centroid class as (b).

Land is solid, Ocean dashed; 210 K is pink and 220 K olive in (b) and (c).
Geographic classification of components is done once and shared by (b) and (c).
Caches (geographic_mask.npz, histogram_checkpoint.npz, classified_components_*K.npz)
use the same names/signatures as the original scripts and are reused if present.

Example:
    python tb_amax_duration_pdf.py --input-netcdf mosaic.nc

Call as

python scripts/Fig_land_ocean_contrasts.py --input-netcdf /data/bfildier/CCRAFT/data/GEOgrid_coldcloud_interpolated/20160810T0000_20160910T2330/bt_mosaic.nc --area-min 100 --area-max 100000 --duration-min 5 --width 9.0 --height 4.0

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os, sys
import shutil
import time
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from cartopy import config as cartopy_config
from cartopy.io.shapereader import Reader
from netCDF4 import Dataset, num2date

currentdir = os.getcwd()
print(currentdir)
sys.path.append(currentdir)

rootdir = Path(__file__).resolve().parents[1]
print(rootdir)
sys.path.insert(0, str(rootdir))
sys.path.insert(0, str(rootdir / "util" / "CCRAFT" / "python" / "src"))

from util.CCRAFT.python.src.reconcile.io.runtime import destination_lease, unique_atomic_path
from util.CCRAFT.python.src.reconcile.io.netcdf import _fsync_directory
from util.CCRAFT.python.src.reconcile.tracking.statistics import area_per_row_km2


LOGGER = logging.getLogger("Fig_land_ocean_contrasts")
SCHEMA_TB = "tb_geographic_conditional_pdf_v1"
SCHEMA_AMAX = "object_amax_geographic_pdf_v1"
SCHEMA_DURATION = "object_duration_geographic_pdf_v1"
GROUPS = ("Unclassified", "Land", "Pacific", "Atlantic", "Indian", "Other seas")
REGIONS = ("Unclassified", "Land", "Ocean")
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
LAND_COLOR, OCEAN_COLOR = "#ad6536", "#243f5b"
THRESHOLD_COLORS = ("#c63e82", "#969000")
DEFAULT_STATS = Path(
    "/data/bfildier/CCRAFT/data/GEOgrid_coldcloud_interpolated/20160810T0000_20160910T2330"
)


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    io = parser.add_argument_group("inputs/outputs")
    io.add_argument("--input-netcdf", type=Path, required=True, help="Kelvin Tb mosaic (panel a).")
    io.add_argument("--variable", default="Harmonized_irBT_mosaic")
    io.add_argument("--stats-root", type=Path, default=DEFAULT_STATS)
    io.add_argument("--output-dir", type=Path, default=Path("output/land_ocean"))
    io.add_argument("--figure-dir", type=Path, default=Path("figures/"))
    io.add_argument("--figure-name", default="Fig_land_ocean_contrasts")
    io.add_argument("--formats", nargs="+", default=["png","pdf"], choices=("png", "pdf", "svg"))
    io.add_argument("--land-shapefile", type=Path)
    io.add_argument("--basin-shapefile", type=Path)
    io.add_argument("--download-geodata", action="store_true")

    tb = parser.add_argument_group("panel (a): Tb")
    tb.add_argument("--tb-min", type=float, default=180.0)
    tb.add_argument("--tb-max", type=float, default=235.0)
    tb.add_argument("--bin-width", type=float, default=1.0)
    tb.add_argument("--max-frames", type=int, help="Leading subset for a smoke test only.")
    tb.add_argument("--row-block", type=int, default=128, help="Histogram/mask working rows.")

    ob = parser.add_argument_group("panels (b), (c): objects")
    ob.add_argument("--thresholds", type=float, nargs=2, default=[210.0, 220.0])
    ob.add_argument(
        "--classification", choices=("peak_centroid", "initial_centroid"), default="peak_centroid"
    )
    ob.add_argument("--bins", type=int, default=30, help="Common bins for (b) and (c).")
    ob.add_argument("--area-min", type=float, help="Optional lower displayed Amax bound, km2.")
    ob.add_argument("--area-max", type=float, help="Optional upper displayed Amax bound, km2.")
    ob.add_argument("--duration-min", type=float, help="Optional lower displayed duration bound, hours.")
    ob.add_argument("--scale", choices=("loglog", "linear"), default="loglog")
    ob.add_argument("--duration-definition", choices=("elapsed", "covered"), default="elapsed")
    ob.add_argument("--batch-size", type=int, default=200_000)

    fig = parser.add_argument_group("figure")
    fig.add_argument("--width", type=float, default=16.0)
    fig.add_argument("--height", type=float, default=4.4)

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


# --------------------------------------------------------------------------- #
# Shared helpers: provenance, atomic writes, geography
# --------------------------------------------------------------------------- #
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


def write_csv(path: Path, header: list[str], rows: Any) -> None:
    temporary = unique_atomic_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
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


@lru_cache(maxsize=1)
def read_geometries(land_path: Path, basin_path: Path) -> tuple[Any, dict[int, list[Any]]]:
    """Read polygons once; shared by the Tb mask and the component classification."""
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


def floats(variable: Any, selection: Any = slice(None)) -> np.ndarray:
    return np.asarray(np.ma.filled(variable[selection].astype(np.float64), np.nan))


def integers(variable: Any, selection: Any = slice(None)) -> np.ndarray:
    values = variable[selection]
    if variable.dtype.kind not in "iu" or np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"Expected unmasked integer values in {variable.name}")
    return np.asarray(values, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Panel (a): Tb distributions
# --------------------------------------------------------------------------- #
def rasterize_regions(
    lat: np.ndarray, lon: np.ndarray, land: Any, basins: dict[int, list[Any]], row_block: int
) -> np.ndarray:
    """Assign centres, with land priority and strict detection of basin overlaps."""
    normalized_lon = (np.asarray(lon, dtype=np.float64) + 180) % 360 - 180
    regions: np.ndarray = np.zeros((len(lat), len(lon)), dtype=np.uint8)
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


def tb_distributions(
    args: argparse.Namespace, land_path: Path, basin_path: Path
) -> tuple[dict[str, dict[str, Any]], np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
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
            args.tb_min, args.tb_max, round((args.tb_max - args.tb_min) / args.bin_width) + 1
        )
        row_area = area_per_row_km2(lat, lon)
        provenance = {
            "schema": SCHEMA_TB,
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
            [provenance["land"], provenance["basins"], lat.tolist(), lon.tolist(), SCHEMA_TB]
        )
        mask_path = args.output_dir / "geographic_mask.npz"
        if mask_path.exists():
            with np.load(mask_path, allow_pickle=False) as cached:
                if str(cached["signature"].item()) != mask_key:
                    raise ValueError("Different geographical grid/sources: use a new output directory")
                regions = cached["regions"].copy()
        else:
            land, basins = read_geometries(land_path, basin_path)
            regions = rasterize_regions(lat, lon, land, basins, args.row_block)
            write_npz(mask_path, signature=mask_key, regions=regions, latitude=lat, longitude=lon)
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
                    raise RuntimeError("Input mosaic changed during analysis; results not published")
                write_npz(
                    checkpoint,
                    signature=experiment,
                    next_frame=frame_index + 1,
                    edges=edges,
                    **accumulator,
                )
                LOGGER.info(
                    "Frames %d/%d; elapsed %.1fs",
                    frame_index + 1,
                    n_frames,
                    time.perf_counter() - start,
                )
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
            "selected_fraction_of_finite_area": selected_area / finite_area if finite_area else None,
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
        "cached_frames_reused": next_frame,
        "frames_read_this_invocation": n_frames - next_frame,
    }
    return groups, edges, summary


# --------------------------------------------------------------------------- #
# Panels (b), (c): object statistics
# --------------------------------------------------------------------------- #
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
            },
        }


def classify_centroids(
    lat: np.ndarray,
    lon: np.ndarray,
    land: Any,
    basins: dict[int, list[Any]],
    batch_size: int,
) -> np.ndarray:
    """Same polygon/priority rules as the Tb mask; exact continuous centroid points."""
    regions = np.zeros(len(lat), dtype=np.uint8)
    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
    x = (lon + 180) % 360 - 180
    for geometry in (land, *(g for items in basins.values() for g in items)):
        if not shapely.is_valid(geometry):
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
    regions[regions >= 2] = 2  # All basins/other seas together are Ocean.
    return regions


def read_duration(path: Path, threshold: float, ids: np.ndarray, definition: str) -> dict[str, Any]:
    """Durations of the same components (same order) as the Amax classification."""
    with Dataset(path) as nc:
        if any(
            int(getattr(nc, flag, 0)) != 1 for flag in ("statistics_complete", "reconcile_complete")
        ):
            raise ValueError(f"Statistics are incomplete: {path}")
        if not np.isclose(float(getattr(nc, "Tb_seed_K", np.nan)), threshold):
            raise ValueError(f"Wrong threshold in {path}")
        for name in ("INT_CCnumber", "INT_duration", "INT_obs_count"):
            if nc[name].dimensions != ("CC",):
                raise ValueError(f"Wrong dimensions for {name}")
        if not np.array_equal(integers(nc["INT_CCnumber"]), ids):
            raise ValueError("Classified component IDs do not match statistics")
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
        "duration_h": elapsed + cadence if definition == "covered" else elapsed,
        "elapsed_h": elapsed,
        "cadence_h": cadence,
    }


def load_components(
    args: argparse.Namespace,
    threshold: float,
    stats_path: Path,
    key: str,
    land_path: Path,
    basin_path: Path,
) -> dict[str, Any]:
    """Amax record with geographic class (cached), plus durations of the same objects."""
    cache_path = args.output_dir / f"classified_components_{threshold:g}K.npz"
    record = None
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
                LOGGER.info("Reusing classified components at %g K", threshold)
    reused = record is not None
    if record is None:
        record = read_components(stats_path, threshold, args.classification, args.batch_size)
        land, basins = read_geometries(land_path, basin_path)
        record["region"] = classify_centroids(
            record["latitude"], record["longitude"], land, basins, args.batch_size
        )
        write_npz(
            cache_path,
            signature=key,
            metadata=json.dumps(record["metadata"], sort_keys=True),
            **{name: value for name, value in record.items() if name != "metadata"},
        )
    record["threshold_K"] = threshold
    record["classification_reused"] = reused
    lat, lon = record["latitude"], record["longitude"]
    record["metadata"]["invalid_centroid_components"] = int(
        np.count_nonzero(
            ~np.isfinite(lat) | ~np.isfinite(lon) | (np.abs(lat) > 90) | (np.abs(lon) > 360)
        )
    )
    record.update(
        read_duration(stats_path, threshold, record["component_id"], args.duration_definition)
    )
    return record


def amax_edges(records: list[dict[str, Any]], args: argparse.Namespace) -> np.ndarray:
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


def duration_edges(records: list[dict[str, Any]], args: argparse.Namespace) -> np.ndarray:
    bins = args.bins
    scale = args.scale
    cadence = records[0]["cadence_h"]
    if not all(np.isclose(r["cadence_h"], cadence) for r in records):
        raise ValueError("The two runs have different sampling cadences")
    positive = [r["duration_h"][(r["duration_h"] > 0) & (r["region"] > 0)] for r in records]
    nonempty = [values for values in positive if len(values)]
    if not nonempty:
        raise ValueError("No classified components have a positive plotted duration")
    minimum, maximum = min(v.min() for v in nonempty), max(v.max() for v in nonempty)
    lower = args.duration_min if args.duration_min is not None else max(cadence / 2, minimum - cadence / 2)
    upper = cadence * 2 ** math.ceil(math.log2(maximum / cadence)) + cadence / 2
    raw = np.linspace(lower, upper, bins + 1) if scale == "linear" else np.geomspace(lower, upper, bins + 1)
    # Place edges between native durations and merge redundant narrow bins,
    # otherwise log bins narrower than one sampling interval create empty gaps.
    return np.unique((np.rint(raw / cadence - 0.5) + 0.5) * cadence)


def object_histograms(
    records: list[dict[str, Any]], edges: np.ndarray, quantity: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """quantity: 'amax_km2' or 'duration_h'. One object, one count, per threshold/region."""
    curves, summaries = [], []
    for record in records:
        value = record[quantity]
        finite = np.isfinite(value) & (value > 0)
        info: dict[str, Any] = {
            "threshold_K": record["threshold_K"],
            "components": int(len(value)),
            "invalid_or_zero_components": int((~finite).sum()),
            "window_boundary_components": int(record["window_boundary"].sum()),
            "groups": {},
        }
        for code, name in enumerate(REGIONS):
            belongs = record["region"] == code
            values = value[finite & belongs]
            counts = np.histogram(values, bins=edges)[0]
            total = int(counts.sum())
            density = counts / (total * np.diff(edges)) if total else np.zeros(len(counts))
            group = {
                "components": int(belongs.sum()),
                "plotted_components": len(values),
                "displayed_components": total,
                "below_range": int(np.count_nonzero(values < edges[0])),
                "above_range": int(np.count_nonzero(values > edges[-1])),
                "pdf_integral": float(np.sum(density * np.diff(edges))),
                "minimum": float(values.min()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "maximum": float(values.max()) if len(values) else None,
            }
            if quantity == "duration_h":
                elapsed = record["elapsed_h"][belongs]
                group["elapsed_zero_components"] = int(np.count_nonzero(elapsed == 0))
                group["elapsed_zero_fraction"] = (
                    float(np.mean(elapsed == 0)) if len(elapsed) else None
                )
            info["groups"][name] = group
            if code:
                if not total:
                    raise ValueError(
                        f"No objects in the displayed range for {name} "
                        f"{record['threshold_K']:g} K ({quantity})"
                    )
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


# --------------------------------------------------------------------------- #
# Figure and tables
# --------------------------------------------------------------------------- #
def panel_label(ax: Any, letter: str) -> None:
    ax.text(
        0.025,
        0.97,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.7)
    )


def plot_object_panel(
    ax: Any,
    curves: list[dict[str, Any]],
    edges: np.ndarray,
    args: argparse.Namespace,
    geometric_centres: bool,
    marker: str | None,
) -> None:
    # colors = dict(zip(sorted(args.thresholds), THRESHOLD_COLORS))
    colors = dict(zip(["Land","Ocean"], [LAND_COLOR,OCEAN_COLOR]))
    loglog = args.scale == "loglog"
    centres = np.sqrt(edges[:-1] * edges[1:]) if (loglog and geometric_centres) else (edges[:-1] + edges[1:]) / 2
    for curve in curves:
        density = curve["pdf"]
        # Zero bins are undefined on a log axis: leave gaps (zeros stay in the CSV).
        displayed = np.where(density > 0, density, np.nan) if loglog else density
        ax.plot(
            centres,
            displayed,
            lw=2.0,
            # color=colors[curve["threshold_K"]],
            color=colors[curve["region"]],
            ls="-" if curve["threshold_K"] == 210.0 else "--",
            marker=marker,
            markersize=2.3,
            label=f"{curve['region']} {curve['threshold_K']:g} K",
        )
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(alpha=0.18, which="major")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper right")


def plot_tb_panel(ax: Any, groups: dict[str, dict[str, Any]], edges: np.ndarray) -> None:
    centres = (edges[:-1] + edges[1:]) / 2
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
            color=OCEAN_COLOR if name == "Ocean" else LAND_COLOR,
            lw=2,
            # ls="--" if name == "Ocean" else "-",
            label=name,
        )
    ax.set(
        xlim=(edges[0], edges[-1]),
        ylim=(0, peak * 1.1 if peak else 1),
        xlabel=r"$T_b$ (K)",
        ylabel=r"PDF (K$^{-1}$)",
    )
    ticks = ax.get_xticks()
    ax.set_xticks(np.unique(np.r_[ticks[(ticks >= edges[0]) & (ticks <= edges[-1])], edges[0], edges[-1]]))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    # Offset below the (a) label so the two never overlap.
    # ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0.0, 0.89))
    ax.legend(frameon=False, fontsize=11, loc="lower right")


def publish_figure(
    base: Path,
    tb_groups: dict[str, dict[str, Any]],
    tb_edges: np.ndarray,
    amax_curves: list[dict[str, Any]],
    a_edges: np.ndarray,
    dur_curves: list[dict[str, Any]],
    d_edges: np.ndarray,
    args: argparse.Namespace,
) -> list[Path]:
    # fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height), layout="constrained")
    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.height), layout="constrained")
    written = []
    try:
        plot_tb_panel(axes[0], tb_groups, tb_edges)
        plot_object_panel(axes[1], amax_curves, a_edges, args, geometric_centres=True, marker=None)
        axes[1].set(xlabel=r"$A_{max}$ (km$^2$)", ylabel=r"PDF (km$^{-2}$)")
        # plot_object_panel(axes[2], dur_curves, d_edges, args, geometric_centres=False, marker="o")
        # axes[2].set(xlabel="Duration [h]", ylabel=r"PDF [h$^{-1}$]")
        for ax, letter in zip(axes, "abc"):
            panel_label(ax, letter)
        for fmt in args.formats:
            path = base.with_suffix(f".{fmt}")
            temporary = unique_atomic_path(path, suffix=f".{fmt}")
            try:
                fig.savefig(temporary, format=fmt, dpi=200, facecolor="white")
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                temporary.replace(path)
                _fsync_directory(path.parent)
            finally:
                temporary.unlink(missing_ok=True)
            written.append(path)
    finally:
        plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    with destination_lease(args.output_dir / "tb-amax-duration-diagnostic"):
        land_path, basin_path = resolve_shapes(args)

        # (a) Tb
        tb_groups, tb_edges, tb_summary = tb_distributions(args, land_path, basin_path)

        # (b), (c) objects
        stats_paths = [args.stats_root / f"CC_stats_indexed_{t:g}K.nc" for t in args.thresholds]
        sources = [identity(p) for p in stats_paths]
        # Same provenance (hence same cache signature) as amax_land_ocean_pdf.py.
        amax_provenance = {
            "schema": SCHEMA_AMAX,
            "statistics_sources": sources,
            "thresholds_K": args.thresholds,
            "classification": args.classification,
            "land": shape_provenance(land_path),
            "basins": shape_provenance(basin_path),
            "geography_reference_directory": str(args.output_dir.resolve()),
            "geography_rule": "continuous centroid; land boundary included; ocean interior only; land priority",
            "peak_rule": "first time index at stored Amax; earliest ragged row breaks remaining ties; stored float32 precision",
            "sample_rule": "one component contributes one observed-window Amax; no lifetime/area filter",
            "normalization": "object-count PDF, separately normalized per threshold/region within displayed range; no area weighting",
            "area_units": "km2",
            "density_units": "km-2",
        }
        key = signature(amax_provenance)
        records = [
            load_components(args, t, p, key, land_path, basin_path)
            for t, p in zip(args.thresholds, stats_paths)
        ]
        if any(identity(p) != s for p, s in zip(stats_paths, sources)):
            raise RuntimeError("Statistics changed during analysis; figure not published")
        if (
            shape_provenance(land_path) != amax_provenance["land"]
            or shape_provenance(basin_path) != amax_provenance["basins"]
        ):
            raise RuntimeError("Geographical sources changed during analysis; figure not published")
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
                if record["metadata"][field] != records[0]["metadata"][field]:
                    raise ValueError(f"The two runs differ in {field}; cannot compare like-for-like")

        a_edges = amax_edges(records, args)
        d_edges = duration_edges(records, args)
        amax_curves, amax_groups = object_histograms(records, a_edges, "amax_km2")
        dur_curves, dur_groups = object_histograms(records, d_edges, "duration_h")

        figures = publish_figure(
            args.figure_dir / args.figure_name,
            tb_groups,
            tb_edges,
            amax_curves,
            a_edges,
            dur_curves,
            d_edges,
            args,
        )

        # Tables
        rows = []
        for name, data in tb_groups.items():
            pdf = conditional_pdf(data["area"], tb_edges)
            for i in range(len(tb_edges) - 1):
                rows.append(
                    [name, tb_edges[i], tb_edges[i + 1], int(data["counts"][i]),
                     float(data["area"][i]), float(pdf[i])]
                )
        write_csv(
            args.output_dir / "tb_histograms.csv",
            ["region", "tb_lower_K", "tb_upper_K", "pixel_observations",
             "area_weight_km2_observations", "conditional_area_pdf_K_inverse"],
            rows,
        )
        for fname, curves, edges, lo, hi, dens in (
            ("amax_histograms.csv", amax_curves, a_edges, "area_lower_km2", "area_upper_km2", "pdf_per_km2"),
            ("duration_histograms.csv", dur_curves, d_edges, "duration_lower_h", "duration_upper_h", "pdf_per_h"),
        ):
            write_csv(
                args.output_dir / fname,
                ["threshold_K", "region", lo, hi, "components", dens],
                (
                    [c["threshold_K"], c["region"], edges[i], edges[i + 1], int(n), float(c["pdf"][i])]
                    for c in curves
                    for i, n in enumerate(c["counts"])
                ),
            )

        summary = {
            "schemas": [SCHEMA_TB, SCHEMA_AMAX, SCHEMA_DURATION],
            "tb": tb_summary,
            "amax": {
                **amax_provenance,
                "classification_signature": key,
                "bin_edges_km2": a_edges.tolist(),
                "groups": amax_groups,
                "classification_reused": [r["classification_reused"] for r in records],
                "invalid_centroid_components": [
                    r["metadata"]["invalid_centroid_components"] for r in records
                ],
            },
            "duration": {
                "duration_definition": args.duration_definition,
                "duration_rule": (
                    "INT_duration = last observation - first observation"
                    if args.duration_definition == "elapsed"
                    else "INT_duration + one measured sampling interval"
                ),
                "zero_duration_rule": (
                    "stored zeros excluded from log PDF and counted in summary"
                    if args.duration_definition == "elapsed"
                    else "stored zeros retained as one sampling interval; recorded NetCDF durations not changed"
                ),
                "classification": "same Amax-centroid class as panel (b)",
                "cadence_h": records[0]["cadence_h"],
                "requested_bins": args.bins,
                "actual_bins": len(d_edges) - 1,
                "bin_edges_h": d_edges.tolist(),
                "groups": dur_groups,
                "window_rule": "observed-window duration; boundary-touching components retained and counted",
            },
            "plot_scale": args.scale,
            "figures": [str(p) for p in figures],
            "figure_size_inches": [args.width, args.height],
            "complete": True,
            "full_volume_reads": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(args.output_dir / "summary_three_panel.json", summary)
        LOGGER.info("Complete in %.1fs: %s", summary["elapsed_seconds"], figures[0])
        return summary


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()