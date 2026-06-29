#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure unique 5x5 : champ Tb (masqué sous un niveau) + isoligne rouge,
un panneau par cas, chacun lu depuis SON PROPRE fichier NetCDF (un fichier
par cas, retrouvé via gather_files_for_case), au pas de temps indiqué dans
show_coords, centré sur (lat_c, lon_c) avec une fenêtre carrée
delta_lat x delta_lon.

Entrées :

Sortie :
- Une image (PNG) avec une grille 5x5 de sous-figures, un panneau par cas.
"""

import os, glob, re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime, timedelta
import copy

DIR_DATA = '/bdd/GEOgrid_coldcloud'

show_coords = {
    "C1": ("2016-06-14T20:00", 7.5, -150), # Westward propagating disturbance at 20º/24h≈2200km/d 
    "C2": ("2016-06-18T19:30", 7.5, -150), # Mix of quasi-stationary aggregate and westward propagating disturbance at 23º/24h≈2500km/d 
    "C3": ("2016-06-24T19:30", 7.5, -145), # Westward propagating disturbance at 15º/24h≈1600km/d 
    "C5": ("2016-09-04T23:00", 10, -160), # Quasi-stationary aggregate
    "C6": ("2016-09-10T16:00", 12.5, -160), # Quasi-stationary aggregate
    "C7": ("2016-06-06T21:00", 9, -125), # Isolated, long-lasting, stationary MCC
    "C8": ("2016-06-28T02:30", 10, -120), # Westward propagating disturbance at 20º/36h≈1300km/d
    "C9": ("2016-06-04T00:30", 5, 20), # Upscale circular merging (several aggregates)
    "C10": ("2016-07-01T18:30", 10, 20), # Partial upscale merging + coupling to AEW
    "C11": ("2016-07-23T03:30", 7.5, 20), # Upscale circular merging (several aggregates)
    "C12": ("2016-07-05T12:00", 5.0, 20), # Multi-day upscale merging (here, not Tbmin but decaying phase, showing remianing fronline, appearance of new individual convective towers)
    "C13": ("2016-07-17T16:00", 2.5, 22), # Disjoint aggregate non merging (except southern part)
    "C14": ("2016-06-09T21:00", 10, 0), # AEW
    "C15": ("2016-08-06T20:30", 5.0, 25.0), # Disjoint aggregate non merging (except southern part)
    "C16": ("2016-08-24T04:30", 5.0, 20.0), # Upscale circular merging (several aggregates) with larger-scale 2-day oscillations
    "C17": ("2016-07-06T07:00", 20, 130), # tropical cyclone
    "C18": ("2016-08-05T23:00", 20, 150), # tropical cyclone
    "C19": ("2016-08-14T00:30", 20, 145), # tropical cyclone
    "C21": ("2016-01-11T12:30", -5.0, -65), # upscale circular merging
    "C22": ("2016-01-14T14:45", -15.0, -65), # long-lasting cluster merging diurnal cycles
    "C23": ("2016-01-19T08:15", -12.0, -70), # upscale circular merging
    "C24": ("2016-02-29T08:00", -5.0, -70), # large continuous cluster with several "touching" DCS 
    "C25": ("2016-04-01T21:15", -5.0, -60), # two large alignments
    "C26": ("2016-04-18T21:00", 0, -60), # squall lines
    "C27": ("2016-04-20T01:00", 0, -65), # alignment of DCSs
}

# mapping satellite -> (sous-dossier, préfixe fichier)
SAT_MAP = {
    "HIMAWARI": ("HIMAWARI+1407", "GEO_L1C-HIMA08", "?_IR???_004_*V1.1"),
    "GOES-W":   ("GOES-W-1350",   "GEO_L1C-GOES15", "?_IR???_004_*V1.1"),
    "GOES-E":   ("GOES-E-0750",   "GEO_L1C-GOES13", "?_IR???_004_*V1.1"),
    "MSG":      ("MSG+0000",      "GEO_L1C-MSG3",   "?_IR???_004_*V1.1"),
    "IODC":     ("IODC_MFG+0570", "GEO_L1C-MET7",   "?_IR???_004_*V1.1"),
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _require_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cartopy est requis. Installe avec: pip install cartopy"
        ) from exc
    return ccrs, cfeature


def _clean_case_id(value) -> str:
    case_id = str(value).strip()
    if case_id.endswith(".0"):
        case_id = case_id[:-2]
    return case_id


def _align_lon_bounds(lon_vals, lon_bounds):
    lo, hi = lon_bounds
    lon_vals = np.asarray(lon_vals, dtype=float)
    data_360 = lon_vals.min() >= 0 and lon_vals.max() > 180
    user_360 = lo >= 0 and hi >= 0
    if data_360 and not user_360:
        lo, hi = lo % 360, hi % 360
    if not data_360 and user_360:
        lo = (lo + 180) % 360 - 180
        hi = (hi + 180) % 360 - 180
    return float(lo), float(hi)


def _sel_lon_wrap(da, lo, hi):
    lon = da.longitude
    lo, hi = _align_lon_bounds(lon.values, (lo, hi))
    lon_min, lon_max = float(lon.min()), float(lon.max())

    if lo <= hi:
        return da.sel(longitude=slice(lo, hi))

    da1 = da.sel(longitude=slice(lo, lon_max))
    da2 = da.sel(longitude=slice(lon_min, hi))
    lon2 = xr.where(da2.longitude < lo, da2.longitude + 360, da2.longitude)
    da2 = da2.assign_coords(longitude=lon2)
    return xr.concat([da1, da2], dim="longitude").sortby("longitude")


def _to_pd_index(arr):
    try:
        return pd.to_datetime(arr)
    except Exception:
        return pd.DatetimeIndex([pd.Timestamp(str(t)) for t in arr])

# Regex pour extraire l'horodatage "YYYY-MM-DDTHH-MM-SS" depuis un nom de fichier
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")
 
def gather_files_for_case(data_dir, satellite, time_str, tolerance_minutes=15):
    """
    Variante tolérante de gather_files_for_case : au lieu d'exiger une
    correspondance exacte sur l'horodatage dans le nom de fichier, liste
    tous les fichiers du/des jour(s) concernés (et du jour précédent/suivant
    si la tolérance déborde de minuit), extrait leur horodatage, et renvoie
    le chemin du fichier le plus proche de time_str, à condition que l'écart
    soit <= tolerance_minutes.
 
    day_iter et SAT_MAP sont injectés (les mêmes que dans ton environnement)
    pour ne pas dépendre d'un import global.
 
    Lève FileNotFoundError si aucun fichier n'est trouvé dans la tolérance.
    """
    if satellite not in SAT_MAP:
        raise ValueError(f"Satellite inconnu: {satellite} (attendus: {list(SAT_MAP.keys())})")
 
    subdir, prefix, suffix = SAT_MAP[satellite]
    target = datetime.fromisoformat(time_str)
 
    win_start = target - timedelta(minutes=tolerance_minutes)
    win_end = target + timedelta(minutes=tolerance_minutes)
 
    if day_iter is not None:
        days = list(day_iter(win_start, win_end))
    else:
        # fallback simple si day_iter n'est pas fourni : jours distincts couverts par la fenêtre
        days = sorted({(d.year, d.month, d.day) for d in (win_start, win_end, target)})
 
    candidates = []
    for y, m, d in days:
        pat = os.path.join(
            data_dir,
            subdir,
            f"{y}",
            f"{y}_{m:02d}_{d:02d}*",
            f"{prefix}_*_{suffix}.nc",
        )
        candidates.extend(glob.glob(pat))
 
    best_path, best_dt, best_diff = None, None, None
    for path in candidates:
        m = _TS_RE.search(os.path.basename(path))
        if not m:
            continue
        file_dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S")
        diff = abs((file_dt - target).total_seconds())
        if best_diff is None or diff < best_diff:
            best_path, best_dt, best_diff = path, file_dt, diff
 
    if best_path is None or best_diff > tolerance_minutes * 60:
        raise FileNotFoundError(
            f"Aucun fichier pour satellite={satellite} dans +/-{tolerance_minutes} min "
            f"autour de {time_str} (meilleur écart trouvé: "
            f"{best_diff/60 if best_diff is not None else 'aucun'} min)."
        )
 
    return best_path

def day_iter(start_dt, end_dt):
    cur = start_dt.date()
    endd = end_dt.date()
    one = timedelta(days=1)
    while cur <= endd:
        yield cur.year, cur.month, cur.day
        cur += one


# --------------------------------------------------------------------------
# Coeur : pour un cas donné, retrouver le fichier, ouvrir, extraire le panneau
# --------------------------------------------------------------------------
def _extract_case_frame(
    dir_data,
    sat,
    time_str,
    lat_c,
    lon_c,
    delta_lat,
    delta_lon,
    var_raw
):
    """
    Retrouve le(s) fichier(s) NetCDF du cas, ouvre le premier, sélectionne
    la boîte lat/lon centrée sur (lat_c, lon_c) et le pas de temps le plus
    proche de time_str.

    Retourne (lon2d, lat2d, frame2d, extent, t_used, file_used).
    """
    target_time = pd.to_datetime(time_str)

    latmin, latmax = lat_c - delta_lat / 2, lat_c + delta_lat / 2
    lonmin, lonmax = lon_c - delta_lon / 2, lon_c + delta_lon / 2

    file_used = gather_files_for_case(dir_data, sat, time_str)
    if not file_used:
        raise FileNotFoundError(f"Aucun fichier trouvé pour sat={sat}, t={time_str}")

    ds = xr.open_dataset(file_used)
    try:
        da = ds[var_raw]

        if da.latitude[0] > da.latitude[-1]:
            da = da.sortby("latitude")

        da = _sel_lon_wrap(da, lonmin, lonmax)
        da = da.sel(latitude=slice(latmin, latmax))

        da.coords["time"] = _to_pd_index(da.time.values)

        t_idx = int(
            np.abs(
                (da.time.values - np.datetime64(target_time))
                .astype("timedelta64[s]")
                .astype(float)
            ).argmin()
        )
        frame = da.isel(time=t_idx)

        lat = frame.latitude.values
        lon = frame.longitude.values
        arr = frame.values

        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
        Lon2d, Lat2d = np.meshgrid(lon, lat)
        t_used = pd.to_datetime(frame.time.values).strftime("%Y-%m-%d %H:%M")

        return Lon2d, Lat2d, arr, extent, t_used, file_used
    finally:
        ds.close()


# --------------------------------------------------------------------------
# Figure 5x5
# --------------------------------------------------------------------------
def plot_cases_grid(
    cases_df,
    show_coords: dict,
    dir_data: str,
    var_raw: str,
    delta_lat: float = 30,
    delta_lon: float = 30,
    output_path: str = "grid_cases.png",
    contour_level_k: float = 210.0,
    mask_above_k: float = 235.0,
    cmap_raw: str = "jet_r",
    vmin: float = 170.0,
    vmax: float = 245.0,
    nrows: int = 5,
    ncols: int = 5,
    panel_size: float = 3,
    dpi: int = 300,
    coast_lw: float = 0.3,
    contour_lw: float = 0.4,
    grid_alpha: float = 0.8,
) -> str:
    """
    Génère une figure unique en grille nrows x ncols, un panneau par cas
    (clés de show_coords, dans l'ordre du dict), chaque panneau étant lu
    depuis son propre fichier NetCDF (via gather_files_for_case), centré
    sur (lat_c, lon_c) avec une fenêtre delta_lat x delta_lon, au temps
    indiqué dans show_coords.
    """
    ccrs, cfeature = _require_cartopy()

    case_ids = list(show_coords.keys())
    n_panels = nrows * ncols
    if len(case_ids) > n_panels:
        raise ValueError(f"{len(case_ids)} cas mais grille {nrows}x{ncols} = {n_panels} places.")

    # table case_id -> satellite / New ID, à partir de cases_df["Old ID"]
    cases_df = cases_df.copy()
    cases_df["ID_clean"] = cases_df["Old ID"].map(_clean_case_id)
    sat_by_case = (
        cases_df.drop_duplicates("ID_clean")
        .set_index("ID_clean")["Satellite"]
        .to_dict()
    )
    newid_by_case = (
        cases_df.drop_duplicates("ID_clean")
        .set_index("ID_clean")["New ID"]
        .to_dict()
    )

    norm = Normalize(vmin, vmax, clip=True)
    cmap = copy.copy(plt.get_cmap(cmap_raw))
    try:
        cmap = cmap.with_extremes(bad="white", under="white")
    except AttributeError:
        cmap.set_bad("white")

    fig = plt.figure(figsize=(panel_size * ncols, panel_size * nrows), dpi=dpi, facecolor="white")

    last_im = None

    # lettres de panneau (a), (b), (c), ...
    import string
    panel_letters = list(string.ascii_lowercase)

    for k, case_id in enumerate(case_ids):
        print("--------------------------------------------------")
        print("|     Case %s                                    |" % case_id)
        print("--------------------------------------------------")

        ax = fig.add_subplot(nrows, ncols, k + 1, projection=ccrs.PlateCarree())

        time_str, lat_c, lon_c = show_coords[case_id]
        sat = sat_by_case.get(case_id)
        new_id = newid_by_case.get(case_id, case_id)
        panel_label = f"({panel_letters[k]})" if k < len(panel_letters) else f"({k + 1})"

        latmin, latmax = lat_c - delta_lat / 2, lat_c + delta_lat / 2
        lonmin, lonmax = lon_c - delta_lon / 2, lon_c + delta_lon / 2
        print("time: %s" % time_str)
        print("lat range: %s - %s" % (latmin, latmax))
        print("lon range: %s - %s" % (lonmin, lonmax))

        if sat is None or (isinstance(sat, float) and np.isnan(sat)):
            ax.set_axis_off()
            ax.set_title(f"{new_id} (satellite inconnu)", fontsize=12, color="gray")
            ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="top", ha="left", zorder=10)
            continue

        sat = str(sat).strip()

        try:
            Lon2d, Lat2d, arr, extent, t_used, file_used = _extract_case_frame(
                dir_data, sat, time_str, lat_c, lon_c, delta_lat, delta_lon,
                var_raw,
            )
            print("file: %s" % file_used)
        except Exception as e:
            ax.set_axis_off()
            ax.set_title(f"{new_id} (erreur)", fontsize=12, color="red")
            ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="top", ha="left", zorder=10)
            print(f"!! {case_id}: {e}")
            continue

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines("110m", linewidth=coast_lw)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=coast_lw * 0.8)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linewidth=coast_lw*0.5, linestyle="-", color="gray", alpha=grid_alpha)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = True
        gl.bottom_labels = True
        arr_masked = np.ma.array(arr, mask=~np.isfinite(arr) | (arr >= mask_above_k))

        im = ax.pcolormesh(
            Lon2d, Lat2d, arr_masked,
            transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, shading="auto"
        )
        last_im = im

        ax.contour(
            Lon2d, Lat2d, arr,
            levels=[float(contour_level_k)],
            colors="red", linewidths=contour_lw,
            transform=ccrs.PlateCarree()
        )

        ax.set_title(f"{new_id} - {t_used}", fontsize=12, pad=2)

        # numérotation du panneau, position identique en haut à gauche
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top", ha="left", zorder=10,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))

    # cases vides restantes si moins de cas que de places
    for k in range(len(case_ids), n_panels):
        ax = fig.add_subplot(nrows, ncols, k + 1)
        ax.set_axis_off()

    if last_im is not None:
        cax = fig.add_axes([0.30, 0.02, 0.4, 0.012])
        cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cb.set_label('Infrared brightness temperature (K)', fontsize=16)
        cb.ax.tick_params(labelsize=14)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.07, wspace=0.05, hspace=0.25)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved grid figure to {output_path}")
    return output_path


# --------------------------------------------------------------------------
# Exemple d'utilisation
# --------------------------------------------------------------------------
if __name__ == "__main__":

    # Load cases info
    cases_list_file = '/home/bfildier/analyses/FildierSaba2026/input/cases.csv'
    cases_df = pd.read_csv(cases_list_file,sep=';')

    # Plot
    plot_cases_grid(
        cases_df=cases_df,                       
        show_coords=show_coords,
        dir_data=DIR_DATA,                       
        var_raw="Harmonized_irBT",               
        delta_lat=30,
        delta_lon=30,
        output_path="../figures/fig_cases.png",
        contour_level_k=210.0,
    )