#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure pour UN SEUL cas (choisi par l'utilisateur) : à partir de DEUX seuils
de température de brillance (Tb), calcule les composantes connexes associées
à chaque seuil et affiche leurs labels (en couleurs) superposés au champ Tb,
pour 4 pas de temps choisis dans une plage [time_start, time_end].

Contrairement à la version "lags" (un panneau par décalage temporel autour
d'un temps de référence, chaque panneau lu dans son propre fichier), cette
version :
  - charge TOUS les fichiers NetCDF du cas compris entre deux dates/heures
    (gather_files_for_case_between), les concatène en une série temporelle
    continue (recadrée sur la boîte lat/lon du cas) ;
  - choisit 4 pas de temps dans cette série (par défaut, répartis
    régulièrement entre time_start et time_end) ;
  - pour CHAQUE seuil (threshold1_k, threshold2_k), calcule les composantes
    connexes (scipy.ndimage.label, connectivité 8) du masque Tb <= seuil ;
  - trace une figure à 2 lignes (une par seuil) x 4 colonnes (une par pas de
    temps choisi), avec le champ Tb en fond (niveaux de gris) et les labels
    des composantes connexes superposés en couleurs.

Sortie :
- Une image (PNG), 2 lignes x 4 colonnes.
"""

import os, glob, re, string, copy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from datetime import datetime, timedelta
from scipy import ndimage

DIR_DATA = '/bdd/GEOgrid_coldcloud'

show_coords = {
    "RC1": ("2016-06-14T20:00", 7.5, -150), # Westward propagating disturbance at 20º/24h≈2200km/d
    "RC2": ("2016-06-18T19:30", 7.5, -150), # Mix of quasi-stationary aggregate and westward propagating disturbance at 23º/24h≈2500km/d
    "RC3": ("2016-06-24T19:30", 7.5, -145), # Westward propagating disturbance at 15º/24h≈1600km/d
    "RC4": ("2016-09-04T23:00", 10, -160), # Quasi-stationary aggregate
    "RC5": ("2016-09-10T16:00", 12.5, -160), # Quasi-stationary aggregate
    "RC6": ("2016-06-06T21:00", 9, -125), # Isolated, long-lasting, stationary MCC
    "RC7": ("2016-06-28T02:30", 10, -120), # Westward propagating disturbance at 20º/36h≈1300km/d
    "RC8": ("2016-06-04T00:30", 5, 20), # Upscale circular merging (several aggregates)
    "RC9": ("2016-07-01T18:30", 10, 20), # Partial upscale merging + coupling to AEW
    "RC10": ("2016-07-23T03:30", 7.5, 20), # Upscale circular merging (several aggregates)
    "RC11": ("2016-07-05T12:00", 5.0, 20), # Multi-day upscale merging
    "RC12": ("2016-07-17T16:00", 2.5, 22), # Disjoint aggregate non merging (except southern part)
    "RC13": ("2016-06-09T21:00", 10, 0), # AEW
    "RC14": ("2016-08-06T20:30", 5.0, 25.0), # Disjoint aggregate non merging (except southern part)
    "RC15": ("2016-08-24T06:00", 5.0, 20.0), # Upscale circular merging with larger-scale 2-day oscillations
    "RC16": ("2016-07-06T07:00", 20, 130), # tropical cyclone
    "RC17": ("2016-08-05T23:00", 20, 150), # tropical cyclone
    "RC18": ("2016-08-14T00:30", 20, 145), # tropical cyclone
    "RC19": ("2016-01-11T12:30", -5.0, -65), # upscale circular merging
    "RC20": ("2016-01-14T14:45", -15.0, -65), # long-lasting cluster merging diurnal cycles
    "RC21": ("2016-01-19T08:15", -12.0, -70), # upscale circular merging
    "RC22": ("2016-02-29T08:00", -5.0, -70), # large continuous cluster with several "touching" DCS
    "RC23": ("2016-04-01T21:15", -5.0, -60), # two large alignments
    "RC24": ("2016-04-18T21:00", 0, -60), # squall lines
    "RC25": ("2016-04-20T01:00", 0, -65), # alignment of DCSs
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
# Helpers (inchangés par rapport au script "lags")
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


def day_iter(start_dt, end_dt):
    cur = start_dt.date()
    endd = end_dt.date()
    one = timedelta(days=1)
    while cur <= endd:
        yield cur.year, cur.month, cur.day
        cur += one


def gather_files_for_case(data_dir, satellite, time_str, tolerance_minutes=15):
    """
    Renvoie le chemin du fichier le plus proche de time_str (à +/- tolerance
    minutes), pour un seul instant. Conservé pour compatibilité mais non
    utilisé par plot_case_labels_range (voir gather_files_for_case_between).
    """
    if satellite not in SAT_MAP:
        raise ValueError(f"Satellite inconnu: {satellite} (attendus: {list(SAT_MAP.keys())})")

    subdir, prefix, suffix = SAT_MAP[satellite]
    target = datetime.fromisoformat(time_str)

    win_start = target - timedelta(minutes=tolerance_minutes)
    win_end = target + timedelta(minutes=tolerance_minutes)

    days = list(day_iter(win_start, win_end))

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


def gather_files_for_case_between(data_dir, satellite, time_start, time_end):
    """
    Liste TOUS les fichiers NetCDF disponibles pour un satellite donné, dont
    l'horodatage (extrait du nom de fichier) tombe dans l'intervalle fermé
    [time_start, time_end].

    Renvoie une liste de tuples (datetime, chemin), triée par temps croissant.
    Lève FileNotFoundError si aucun fichier n'est trouvé dans l'intervalle.
    """
    if satellite not in SAT_MAP:
        raise ValueError(f"Satellite inconnu: {satellite} (attendus: {list(SAT_MAP.keys())})")

    subdir, prefix, suffix = SAT_MAP[satellite]
    start = pd.to_datetime(time_start).to_pydatetime()
    end = pd.to_datetime(time_end).to_pydatetime()

    if start > end:
        start, end = end, start

    candidates = []
    for y, m, d in day_iter(start, end):
        pat = os.path.join(
            data_dir,
            subdir,
            f"{y}",
            f"{y}_{m:02d}_{d:02d}*",
            f"{prefix}_*_{suffix}.nc",
        )
        candidates.extend(glob.glob(pat))

    files_times = []
    for path in candidates:
        m = _TS_RE.search(os.path.basename(path))
        if not m:
            continue
        file_dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S")
        if start <= file_dt <= end:
            files_times.append((file_dt, path))

    if not files_times:
        raise FileNotFoundError(
            f"Aucun fichier pour satellite={satellite} entre {start} et {end}."
        )

    files_times.sort(key=lambda x: x[0])
    return files_times


# --------------------------------------------------------------------------
# Chargement d'une plage temporelle complète pour un cas
# --------------------------------------------------------------------------
def _load_case_range(dir_data, sat, time_start, time_end, lat_c, lon_c,
                      delta_lat, delta_lon, var_raw):
    """
    Charge et concatène en une seule série temporelle (dim "time") tous les
    fichiers NetCDF du satellite `sat` compris entre time_start et time_end,
    recadrés sur la boîte lat/lon centrée sur (lat_c, lon_c).

    Retourne un DataArray xarray de dimensions (time, latitude, longitude).
    """
    latmin, latmax = lat_c - delta_lat / 2, lat_c + delta_lat / 2
    lonmin, lonmax = lon_c - delta_lon / 2, lon_c + delta_lon / 2

    files_times = gather_files_for_case_between(dir_data, sat, time_start, time_end)

    das = []
    for file_dt, path in files_times:
        ds = xr.open_dataset(path)
        try:
            da = ds[var_raw]

            if da.latitude[0] > da.latitude[-1]:
                da = da.sortby("latitude")

            da = _sel_lon_wrap(da, lonmin, lonmax)
            da = da.sel(latitude=slice(latmin, latmax))
            da.coords["time"] = _to_pd_index(da.time.values)

            das.append(da.load())
        finally:
            ds.close()

    full = xr.concat(das, dim="time").sortby("time")
    # on retire d'éventuels doublons de temps (fichiers qui se recouvrent)
    _, unique_idx = np.unique(full.time.values, return_index=True)
    full = full.isel(time=np.sort(unique_idx))

    # on referme strictement sur [time_start, time_end]
    full = full.sel(time=slice(pd.to_datetime(time_start), pd.to_datetime(time_end)))

    if full.sizes.get("time", 0) == 0:
        raise ValueError(
            f"Aucun pas de temps disponible entre {time_start} et {time_end} "
            f"après chargement des fichiers."
        )

    return full


def _pick_snapshot_times(time_index, n=4):
    """
    Choisit n pas de temps dans time_index (DatetimeIndex), répartis aussi
    régulièrement que possible entre le premier et le dernier temps
    disponibles. Pour chaque temps cible, on retient l'index du pas de temps
    réellement disponible le plus proche (sans doublon si possible).
    """
    time_index = pd.DatetimeIndex(time_index)
    if len(time_index) == 0:
        raise ValueError("time_index vide, impossible de choisir des pas de temps.")

    if len(time_index) <= n:
        return list(range(len(time_index)))

    targets = pd.date_range(time_index[0], time_index[-1], periods=n)

    chosen = []
    used = set()
    for t in targets:
        diffs = np.abs((time_index - t).total_seconds())
        order = np.argsort(diffs)
        for idx in order:
            if idx not in used:
                chosen.append(int(idx))
                used.add(int(idx))
                break

    chosen.sort()
    return chosen


# --------------------------------------------------------------------------
# Composantes connexes à partir d'un seuil de Tb
# --------------------------------------------------------------------------
def _label_field(arr, threshold_k, connectivity=2):
    """
    Calcule les composantes connexes du masque (Tb finie et Tb <= threshold_k).

    connectivity=1 -> 4-connectivité (voisins orthogonaux uniquement)
    connectivity=2 -> 8-connectivité (voisins orthogonaux + diagonaux)

    Retourne (labels_masked, n_labels) où labels_masked est un array masqué
    (0 / fond -> masqué) de même forme que arr.
    """
    mask = np.isfinite(arr) & (arr <= threshold_k)
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, n_labels = ndimage.label(mask, structure=structure)
    labels_masked = np.ma.masked_where(labels == 0, labels)
    return labels_masked, n_labels


def _label_cmap_norm(max_label, base_cmap="tab20b"):
    """
    Construit une colormap discrète pour colorer des labels entiers
    1..max_label (0 = fond, transparent), en cyclant sur une palette
    qualitative (par défaut tab20, 20 couleurs) pour que des labels voisins
    en numéro soient visuellement distincts.
    """
    max_label = max(int(max_label), 1)
    base = plt.get_cmap(base_cmap)
    ncolors = base.N
    colors = [(1, 1, 1, 0)]  # label 0 -> transparent
    for i in range(1, max_label + 1):
        colors.append(base(((i - 1) % ncolors) / max(ncolors - 1, 1)))
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, max_label + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# --------------------------------------------------------------------------
# Figure : un seul cas, 2 seuils (lignes) x 4 pas de temps (colonnes)
# --------------------------------------------------------------------------
def plot_case_labels_range(
    case_id: str,
    cases_df,
    show_coords: dict,
    dir_data: str,
    var_raw: str,
    # time_start: str,
    # time_end: str,
    threshold1_k: float,
    threshold2_k: float,
    lag_hours: float,   # <- intervalle de temps entre panneaux
    lag_centering=[-2,-1,0,1],
    n_snapshots: int = 4,
    delta_lat: float = 30,
    delta_lon: float = 30,
    output_path: str = "case_labels_range.png",
    cmap_tb: str = "gray",
    vmin: float = 170.0,
    vmax: float = 245.0,
    connectivity: int = 2,
    panel_size: float = 3,
    dpi: int = 300,
    coast_lw: float = 0.3,
    grid_alpha: float = 0.8,
    label_alpha: float = 0.75,
) -> str:
    """
    Génère une figure 2 lignes x n_snapshots colonnes pour UN cas donné,
    entre time_start et time_end :
      - ligne 1 : composantes connexes du masque Tb <= threshold1_k
      - ligne 2 : composantes connexes du masque Tb <= threshold2_k
    superposées (en couleurs) au champ Tb (en niveaux de gris), pour
    n_snapshots pas de temps choisis dans la plage.

    Tous les fichiers NetCDF du cas compris entre time_start et time_end sont
    chargés (gather_files_for_case_between / _load_case_range) et concaténés
    en une seule série temporelle avant sélection des pas de temps affichés.
    """
    ccrs, cfeature = _require_cartopy()

    if case_id not in show_coords:
        raise KeyError(f"Cas '{case_id}' absent de show_coords.")

    # compute start and end times
    ref_time_str, lat_c, lon_c = show_coords[case_id]
    ref_time = pd.to_datetime(ref_time_str)
    time_start = (ref_time + timedelta(hours=lag_hours*lag_centering[0])).strftime("%Y-%m-%dT%H:%M")
    time_end = (ref_time + timedelta(hours=lag_hours*lag_centering[-1])).strftime("%Y-%m-%dT%H:%M")

    # table case_id -> satellite / New ID, à partir de cases_df["New ID"]
    cases_df = cases_df.copy()
    cases_df["ID_clean"] = cases_df["New ID"].map(_clean_case_id)
    sat_by_case = (
        cases_df.drop_duplicates("ID_clean").set_index("ID_clean")["Satellite"].to_dict()
    )
    newid_by_case = (
        cases_df.drop_duplicates("ID_clean").set_index("ID_clean")["New ID"].to_dict()
    )
    sat = sat_by_case.get(case_id)
    new_id = newid_by_case.get(case_id, case_id)
    if sat is None or (isinstance(sat, float) and np.isnan(sat)):
        raise ValueError(f"Satellite inconnu pour le cas {case_id}.")
    sat = str(sat).strip()

    # ------------------------------------------------------------------
    # Chargement de toute la plage temporelle, puis choix des pas de temps
    # ------------------------------------------------------------------
    full = _load_case_range(
        dir_data, sat, time_start, time_end, lat_c, lon_c, delta_lat, delta_lon, var_raw
    )
    time_index = pd.DatetimeIndex(full.time.values)
    snap_idx = _pick_snapshot_times(time_index, n=n_snapshots)
    n_cols = len(snap_idx)

    thresholds = [float(threshold1_k), float(threshold2_k)]
    n_rows = len(thresholds)

    tb_norm = Normalize(vmin, vmax, clip=True)
    tb_cmap = copy.copy(plt.get_cmap(cmap_tb))

    fig = plt.figure(figsize=(panel_size * n_cols, panel_size * n_rows), dpi=dpi, facecolor="white")

    panel_letters = list(string.ascii_lowercase)
    last_tb_im = None
    panel_count = 0

    for row, thresh in enumerate(thresholds):

        arr_full = full.values

        # composantes connexes
        labels_masked, n_labels = _label_field(arr_full, thresh, connectivity=connectivity)
        lab_cmap, lab_norm = _label_cmap_norm(n_labels,base_cmap="Set3")

        for col, ti in enumerate(snap_idx):
            frame = full.isel(time=ti)
            lat = frame.latitude.values
            lon = frame.longitude.values
            arr = frame.values
            extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
            Lon2d, Lat2d = np.meshgrid(lon, lat)
            t_used = pd.to_datetime(frame.time.values).strftime("%Y-%m-%d %H:%M")

            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines("110m", linewidth=coast_lw)
            ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=coast_lw * 0.8)
            gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                               linewidth=coast_lw * 0.5, linestyle="-", color="gray", alpha=grid_alpha)
            gl.top_labels = gl.right_labels = False
            gl.left_labels = (col == 0)
            gl.bottom_labels = (row == n_rows - 1)

            # fond : champ Tb en niveaux de gris
            im_tb = ax.pcolormesh(
                Lon2d, Lat2d, arr,
                transform=ccrs.PlateCarree(),
                cmap=tb_cmap, norm=tb_norm, shading="auto",
            )
            last_tb_im = im_tb

            # composantes connexes pour le seuil de cette ligne
            labels_masked_t = labels_masked[ti]
            ax.pcolormesh(
                Lon2d, Lat2d, labels_masked_t,
                transform=ccrs.PlateCarree(),
                cmap=lab_cmap, norm=lab_norm, shading="auto",
                alpha=label_alpha,
            )

            # contour du seuil, pour repère visuel
            ax.contour(
                Lon2d, Lat2d, arr,
                levels=[thresh], colors="red", linewidths=0.4,
                transform=ccrs.PlateCarree(),
            )

            title = f"{new_id} - {t_used}" #if row == 0 else t_used
            ax.set_title(title, fontsize=11, pad=2)

            if col == 0:
                ax.text(-0.27, 0.5, f"Tb ≤ {thresh:.0f} K",
                         transform=ax.transAxes, fontsize=13, rotation=90,
                         va="center", ha="center")

            panel_label = f"({panel_letters[panel_count]})" if panel_count < len(panel_letters) else f"({panel_count + 1})"
            ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=11,
                     fontweight="bold", va="top", ha="left", zorder=10,
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                               edgecolor="none", alpha=0.7))
            panel_count += 1

    if last_tb_im is not None:
        cax = fig.add_axes([0.30, 0, 0.4, 0.02])
        cb = fig.colorbar(last_tb_im, cax=cax, orientation="horizontal")
        cb.set_label('Infrared brightness temperature (K)', fontsize=11)
        cb.ax.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.10, wspace=0.05, hspace=0.15)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved labels-range figure to {output_path}")
    return output_path


# --------------------------------------------------------------------------
# Exemple d'utilisation
# --------------------------------------------------------------------------
if __name__ == "__main__":

    # Load cases info
    cases_list_file = '/home/bfildier/analyses/FildierSaba2026/input/cases.csv'
    cases_df = pd.read_csv(cases_list_file, sep=';')

    CASE_ID = "RC18"  # <- cas choisi par l'utilisateur

    plot_case_labels_range(
        case_id=CASE_ID,
        cases_df=cases_df,
        show_coords=show_coords,
        dir_data=DIR_DATA,
        var_raw="Harmonized_irBT",
        # time_start="2016-04-19T14:45",
        # time_end="2016-04-20T05:45",
        threshold1_k=220.0,   # seuil 1
        threshold2_k=210.0,   # seuil "coeur" convectif
        lag_hours=24,   # <- intervalle de temps entre panneaux
        lag_centering=[-1,0,1,2],# [-2,-1,0,1], #
        n_snapshots=4,
        delta_lat=30,
        delta_lon=30,
        output_path=f"../figures/tests/fig_naive_labels_{CASE_ID}.png",
    )