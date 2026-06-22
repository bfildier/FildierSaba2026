# -------------------------------------------------------------------------
from __future__ import annotations
import os,re
import glob
import cv2
import hashlib
import logging, warnings, gc
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any

import xarray as xr
import dask.array as da
import numpy as np
import pandas as pd
import numcodecs
import zarr, json
import shutil
from scipy import ndimage
from tqdm import tqdm
import dask
from zarr.storage import ZipStore
from netCDF4 import Dataset as NetCDF4Dataset
from xarray.conventions import SerializationWarning

from code_cwt.algo_version_6.file_thermal_conversion import (
    ensure_tb_data
)

# Logger dédié pour ce module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#=========== Fonctions utilitaires de chargement de données ===========
def _compute_file_list_hash(
    file_list: List[str],
    params: Dict[str, Any],
) -> str:
    """
    Calcule un hash MD5 stable à partir :
      - du chemin absolu des fichiers,
      - de leur taille,
      - de leur date de modification,
      - des paramètres qui influencent réellement la donnée produite.

    Cela évite de réutiliser un cache si un fichier a été modifié "en place"
    sans changer de nom.
    """
    files_meta = []
    for f in sorted(file_list):
        p = Path(f).resolve()
        st = os.stat(p)
        files_meta.append(
            {
                "path": str(p),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            }
        )

    payload = {
        "files": files_meta,
        "params": params,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()



@contextmanager
def _ignore_serialization_warnings():
    """Masque les warnings Xarray/Zarr sur la sérialisation."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SerializationWarning)
        warnings.filterwarnings(
            "ignore",
            message="The specified chunks separate the stored chunks",
            category=UserWarning
        )
        yield


def _filter_valid_files(file_list: List[str]) -> List[str]:
    """
    Teste rapidement chaque fichier NetCDF et ne garde que ceux lisibles.
    """
    valid = []
    for f in file_list:
        try:
            with NetCDF4Dataset(f, 'r'):
                valid.append(f)
        except Exception as e:
            logger.warning(f"Fichier corrompu ou illisible, ignoré : {f} ({e})")
    return valid


def _ensure_clean_store(store_path: Path):
    """Supprime proprement l'ancien store ZIP."""
    if store_path.exists():
        if store_path.is_file():
            store_path.unlink()
            logger.info(f"Ancien ZIP supprimé: {store_path}")
    # création du dossier parent
    store_path.parent.mkdir(parents=True, exist_ok=True)

#=========== Fonctions de seuil et chargement de données NetCDF ===========
#----------------------------------------------------------------------
# 1. Appliquer un seuil sur les données
# 2. Charger plusieurs fichiers NetCDF en Zarr optimisé
#----------------------------------------------------------------------
def threshold_data(
    volume: xr.DataArray,
    lat_range: Tuple[float, float] = (-30, 30),
    bt_min: float = 170.0,
    bt_max: float = 300.0
) -> Tuple[xr.DataArray, np.ndarray]:
    """
    Applique un seuil sur un DataArray par latitude et plage de BT.

    Parameters
    ----------
    volume
        xarray.DataArray avec dimension 'latitude'.
    lat_range
        Tuple (min_lat, max_lat) en degrés.
    bt_min
        Float: température minimale en unités de brightness temperature (Kelvin).
    bt_max
        Float: température maximale en unités de brightness temperature (Kelvin).

    Returns
    -------
    vol_thresh
        DataArray tronqué et cast en float32, avec NaN en-dehors de [bt_min, bt_max].
    lats
        Tableau des latitudes sélectionnées.
    """
    if 'latitude' not in volume.dims:
        raise ValueError(f"Missing latitude dimension in {list(volume.dims)}")
    # on extrait toutes les latitudes
    lats = volume.coords['latitude'].values
    # masque des latitudes dans l’intervalle souhaité
    mask_lat = (lats >= lat_range[0]) & (lats <= lat_range[1])
    if not mask_lat.any():
        raise ValueError(f"No latitude in range {lat_range}")

    # sous-ensemble par latitude et cast en float32
    vol_sub = volume.sel(latitude=lats[mask_lat]).astype('float32')

    # on ne conserve que les valeurs entre bt_min et bt_max
    vol_thresh = vol_sub.where((vol_sub >= bt_min) & (vol_sub <= bt_max))

    return vol_thresh, lats[mask_lat]


def _standardize_latlon_names(
    obj: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Standardise les noms spatiaux vers:
      - latitude
      - longitude

    Gère les cas fréquents:
      lat/lon -> latitude/longitude
    """
    rename_map = {}

    if "latitude" not in obj.dims and "lat" in obj.dims:
        rename_map["lat"] = "latitude"
    elif "latitude" not in obj.coords and "lat" in obj.coords:
        rename_map["lat"] = "latitude"

    if "longitude" not in obj.dims and "lon" in obj.dims:
        rename_map["lon"] = "longitude"
    elif "longitude" not in obj.coords and "lon" in obj.coords:
        rename_map["lon"] = "longitude"

    if rename_map:
        obj = obj.rename(rename_map)

    return obj


def _extract_model_step_from_filename(path: str) -> int:
    """
    Extrait le step numérique depuis un nom du type :
    DYAMOND_..._0000000240.LWNTA.2D.nc
    """
    name = Path(path).name
    m = re.search(r"_(\d+)\.[^.]+\.2D\.nc$", name)
    if not m:
        raise ValueError(f"Impossible d'extraire le step depuis le nom: {name}")
    return int(m.group(1))


def _assign_time_from_filename_step(
    ds: xr.Dataset,
    *,
    step_seconds: float = 7.5,
    time_origin: str = "1970-01-01",
) -> xr.Dataset:
    """
    Reconstruit un vrai timestamp à partir du nom du fichier.

    Hypothèse:
      - 1 fichier = 1 snapshot temporel
      - le nombre dans le nom du fichier est le step modèle
      - pas de temps modèle = step_seconds

    Exemple:
      step=240, step_seconds=7.5 -> 1800 s -> 00:30
    """
    ds = _standardize_latlon_names(ds)

    src = ds.encoding.get("source", "")
    if not src:
        raise ValueError("ds.encoding['source'] introuvable: impossible de reconstruire le temps depuis le nom du fichier.")

    step = _extract_model_step_from_filename(src)
    ts = pd.Timestamp(time_origin) + pd.to_timedelta(step * step_seconds, unit="s")

    # On remplace complètement l'axe/coord time éventuel
    if "time" in ds.dims:
        if int(ds.sizes["time"]) != 1:
            raise ValueError(
                f"_assign_time_from_filename_step suppose 1 seul temps par fichier, "
                f"mais ce fichier en contient {ds.sizes['time']} : {src}"
            )
        ds = ds.isel(time=0, drop=True)

    if "time" in ds.coords:
        ds = ds.drop_vars("time", errors="ignore")

    ds = ds.expand_dims(time=[ts])
    return ds


def load_multiple_nc_files(
    file_path: Union[str, List[str]],
    var_name: str = "Harmonized_irBT",            # nom dans les fichiers source
    output_var_name: str = "Harmonized_irBT",     # nom standardisé en sortie
    source_kind: str = "auto",                    # "auto", "tb", "olr"
    olr_min: float = 40.0,
    olr_max: float = 500.0,
    time_source: str = "auto",
    filename_time_step_seconds: float = 7.5,
    time_origin: str = "1970-01-01",
    read_chunks: Optional[Dict[str, int]] = None,
    zarr_store: str = "temp_zarr",
    force_overwrite: bool = False,
    block_size: int = 10,
    gzip_level: int = 5,
    lat_range: Tuple[float, float] = (-30, 30),
    bt_min: float = 170.0,
    bt_max: float = 235.0,
    remove_blinking: bool = False
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Charge et concatène des fichiers NetCDF en Zarr optimisé (ZIP).

    Fonction robuste qui :
      1. Récupère le listing (glob ou liste) et le trie.
      2. Filtre les fichiers corrompus.
      3. Calcule un hash MD5 robuste pour le cache.
      4. Réutilise ou supprime l'ancien ZIP.
      5. Charge par blocs avec fallback fichier par fichier si besoin.
      6. Concatène paresseusement.
      7. Convertit si nécessaire OLR -> Tb.
      8. Applique le seuillage en Tb + correction adaptative optionnelle des flashs.
      9. Rechunk.
     10. Écrit en ZIP Zarr compressé.

    Returns
    -------
    data : xr.DataArray
        Champ final en Tb [K], seuillé, chunké, prêt pour la suite.
    time : xr.DataArray
    latitude : xr.DataArray
    longitude : xr.DataArray
    """

    # ------------------------------------------------------------------
    # Helpers locaux : détection/correction adaptative des flashs
    # ------------------------------------------------------------------
    def _clean_flash_artifacts_adaptive(
        da: xr.DataArray,
    ) -> Tuple[xr.DataArray, np.ndarray]:
        """
        Version ultra-rapide :
          - sous-échantillonnage spatial agressif
          - aucun masque 3D de pixels anormaux
          - uniquement des diagnostics 1D par frame
          - si une frame semble corrompue localement OU globalement, on la supprime

        Détection basée sur 4 séries temporelles calculées sur un échantillon spatial :
          - jump_mean  : anomalie moyenne vs 0.5*(t-1+t+1)  -> flash global
          - jump_max   : anomalie max                       -> flash local fort
          - score_mean : jump_mean / cohérence voisins
          - score_max  : jump_max  / cohérence voisins

        Puis seuils robustes automatiques via médiane + MAD.
        """
        def _robust_stats_1d_np(arr_1d: np.ndarray) -> Tuple[float, float]:
            arr = np.asarray(arr_1d, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return 0.0, 0.0

            med = float(np.nanmedian(arr))
            mad = float(np.nanmedian(np.abs(arr - med)))
            sigma = 1.4826 * mad
            if not np.isfinite(sigma):
                sigma = 0.0
            return med, sigma

        if da.sizes.get("time", 0) < 3:
            logger.warning(
                "remove_blinking=True demandé, mais moins de 3 timestamps : "
                "aucune correction de flash appliquée."
            )
            return da, np.array([], dtype="datetime64[ns]")

        da_clean = da.astype("float32", copy=False)
        removed_times_all: List[np.ndarray] = []

        # Cible de sous-échantillonnage très agressive pour aller vite
        target_lat = 128
        target_lon = 128

        for ipass in range(2):
            if da_clean.sizes.get("time", 0) < 3:
                break

            step_lat = max(1, da_clean.sizes["latitude"] // target_lat)
            step_lon = max(1, da_clean.sizes["longitude"] // target_lon)

            sample = da_clean.isel(
                latitude=slice(None, None, step_lat),
                longitude=slice(None, None, step_lon),
            )

            logger.info("Calcul des diagnostics de flash sur l'échantillon spatial...")
            sample_np = np.asarray(
                sample.transpose("time", "latitude", "longitude").data.compute(),
                dtype=np.float32,
            )
            
            nt_s = sample_np.shape[0]
            
            # tableaux 1D de sortie, avec NaN aux bords
            jump_mean_np = np.full(nt_s, np.nan, dtype=np.float32)
            jump_max_np = np.full(nt_s, np.nan, dtype=np.float32)
            neigh_mean_np = np.full(nt_s, np.nan, dtype=np.float32)
            
            if nt_s >= 3:
                prev_np = sample_np[:-2]
                cur_np  = sample_np[1:-1]
                next_np = sample_np[2:]
            
                valid = np.isfinite(prev_np) & np.isfinite(cur_np) & np.isfinite(next_np)
            
                with np.errstate(invalid="ignore"):
                    ref_np = 0.5 * (prev_np + next_np)
            
                    err_np = np.abs(cur_np - ref_np)
                    err_np[~valid] = np.nan
            
                    neigh_np = np.abs(prev_np - next_np)
                    neigh_np[~valid] = np.nan
            
                    jump_mean_np[1:-1] = np.nanmean(err_np, axis=(1, 2)).astype(np.float32)
                    jump_max_np[1:-1] = np.nanmax(err_np, axis=(1, 2)).astype(np.float32)
                    neigh_mean_np[1:-1] = np.nanmean(neigh_np, axis=(1, 2)).astype(np.float32)
            
                del prev_np, cur_np, next_np, ref_np, err_np, neigh_np, valid
            
            del sample_np
            gc.collect()
            logger.info("Diagnostics de flash calculés sur l'échantillon spatial...")

            denom = np.maximum(neigh_mean_np, 1e-3)
            score_mean_np = jump_mean_np / denom
            score_max_np  = jump_max_np / denom

            if jump_mean_np.size > 2:
                core = slice(1, -1)
            else:
                core = slice(None)

            # diagnostics globaux rapides pour détecter les frames corrompues par flash :
            logger.info("Calcul des seuils de détection de flash...")
            jm_med, jm_sig = _robust_stats_1d_np(jump_mean_np[core])
            jx_med, jx_sig = _robust_stats_1d_np(jump_max_np[core])
            sm_med, sm_sig = _robust_stats_1d_np(score_mean_np[core])
            sx_med, sx_sig = _robust_stats_1d_np(score_max_np[core])
            logger.info("Seuils de détection de flash calculés.")

            jm_sig = max(jm_sig, 1e-6)
            jx_sig = max(jx_sig, 1e-6)
            sm_sig = max(sm_sig, 1e-6)
            sx_sig = max(sx_sig, 1e-6)

            global_bad = (
                (jump_mean_np > (jm_med + 6.0 * jm_sig)) |
                (jump_max_np  > (jx_med + 6.0 * jx_sig)) |
                (score_mean_np > (sm_med + 6.0 * sm_sig)) |
                (score_max_np  > (sx_med + 6.0 * sx_sig))
            )

            # pas de décision aux bords
            if global_bad.size >= 1:
                global_bad[0] = False
                global_bad[-1] = False

            n_bad = int(global_bad.sum())

            logger.info(
                f"[flash ultra-fast pass {ipass+1}/2] "
                f"frames supprimées={n_bad}, "
                f"sample_shape=({sample.sizes['time']}, {sample.sizes['latitude']}, {sample.sizes['longitude']})"
            )

            if n_bad == 0:
                break

            bad_times = np.asarray(sample.time.values)[global_bad]
            removed_times_all.append(bad_times)

            da_clean = da_clean.drop_sel(time=bad_times)

        if removed_times_all:
            removed_times_all = [bt for bt in removed_times_all if len(bt) > 0]
            if removed_times_all:
                removed_times = np.unique(np.concatenate(removed_times_all))
            else:
                removed_times = np.array([], dtype="datetime64[ns]")
        else:
            removed_times = np.array([], dtype="datetime64[ns]")

        return da_clean.astype("float32"), removed_times

    # ------------------------------------------------------------------
    # 1) Listing des fichiers
    # ------------------------------------------------------------------
    if isinstance(file_path, list):
        file_list = sorted(file_path)
    else:
        file_list = sorted(glob.glob(file_path, recursive=True))

    if not file_list:
        raise FileNotFoundError(f"Aucun fichier trouvé pour : {file_path}")
    logger.info(f"{len(file_list)} fichiers trouvés au total.")

    # ------------------------------------------------------------------
    # 2) Filtrage rapide des fichiers corrompus
    # ------------------------------------------------------------------
    file_list = _filter_valid_files(file_list)
    if not file_list:
        raise FileNotFoundError("Aucun fichier valide après filtrage.")
    logger.info(f"{len(file_list)} fichiers valides retenus.")

    # ------------------------------------------------------------------
    # 3) Hash cache robuste
    # ------------------------------------------------------------------
    params = dict(
        var_name=var_name,
        output_var_name=output_var_name,
        source_kind=source_kind,
        olr_min=olr_min,
        olr_max=olr_max,
        time_source=time_source,
        filename_time_step_seconds=filename_time_step_seconds,
        time_origin=time_origin,
        lat_range=lat_range,
        bt_min=bt_min,
        bt_max=bt_max,
        read_chunks=read_chunks,
        remove_blinking=remove_blinking,
        # version interne de l'algorithme de correction des flashs
        blink_method_version="adaptive_temporal_mad_v1",
    )

    file_list_hash = _compute_file_list_hash(file_list, params)
    store_path = Path(zarr_store if zarr_store.endswith(".zip") else zarr_store + ".zip")

    # ------------------------------------------------------------------
    # 4) Réutilisation éventuelle du cache Zarr
    # ------------------------------------------------------------------
    if store_path.exists() and not force_overwrite:
        try:
            ds = xr.open_zarr(store_path, consolidated=True)
            cached_hash = ds.attrs.get("source_hash", "")

            if cached_hash == file_list_hash:
                logger.info("Réutilisation du store Zarr existant.")

                cached_var = output_var_name if output_var_name in ds.data_vars else var_name
                data = ds[cached_var]

                data_hash = data.attrs.get("source_hash") or cached_hash
                data.attrs["source_hash"] = data_hash

                return (
                    data,
                    ds.coords["time"],
                    ds.coords["latitude"],
                    ds.coords["longitude"],
                )
            else:
                logger.info("Hash mismatch – suppression de l'ancien ZIP.")
                _ensure_clean_store(store_path)

        except Exception as e:
            logger.warning(f"Échec réouverture store existant ({e}) – suppression.")
            _ensure_clean_store(store_path)

    # ------------------------------------------------------------------
    # 5) Lecture par blocs avec fallback fichier par fichier
    # ------------------------------------------------------------------
    groups = [file_list[i:i + block_size] for i in range(0, len(file_list), block_size)]
    blocks: List[xr.Dataset] = []

    for grp in tqdm(groups, desc="Loading"):
        try:
            with _ignore_serialization_warnings():
                if time_source == "filename_step":
                    ds_blk = xr.open_mfdataset(
                        grp,
                        combine="nested",
                        concat_dim="time",
                        data_vars=[var_name],
                        chunks=read_chunks or {},
                        parallel=False,
                        preprocess=lambda ds: _assign_time_from_filename_step(
                            ds,
                            step_seconds=filename_time_step_seconds,
                            time_origin=time_origin,
                        ),
                    )
                else:
                    ds_blk = xr.open_mfdataset(
                        grp,
                        combine="by_coords",
                        data_vars=[var_name],
                        chunks=read_chunks or {},
                        parallel=False,
                    )
                    ds_blk = _standardize_latlon_names(ds_blk)

            if var_name not in ds_blk:
                raise KeyError(f"Variable '{var_name}' absente dans le bloc.")

            ds_blk = ds_blk[[var_name]].astype("float32")
            blocks.append(ds_blk)

        except Exception as e:
            logger.warning(f"Bloc entier échec ({e}), fallback fichier-à-fichier :")
            for f in grp:
                try:
                    with _ignore_serialization_warnings():
                        ds_single = xr.open_dataset(
                            f,
                            engine="netcdf4",
                            chunks=read_chunks or {},
                        )

                    if time_source == "filename_step":
                        ds_single = _assign_time_from_filename_step(
                            ds_single,
                            step_seconds=filename_time_step_seconds,
                            time_origin=time_origin,
                        )
                    else:
                        ds_single = _standardize_latlon_names(ds_single)

                    if var_name not in ds_single:
                        raise KeyError(f"Variable '{var_name}' absente dans {f}")

                    ds_single = ds_single[[var_name]].astype("float32")
                    blocks.append(ds_single)
                    logger.info(f"  Ajouté : {f}")

                except Exception as e2:
                    logger.warning(f"  Ignoré {f} ({e2})")

    if not blocks:
        raise RuntimeError("Tous les fichiers ont échoué à l'ouverture.")

    # ------------------------------------------------------------------
    # 6) Concaténation
    # ------------------------------------------------------------------
    combined = xr.concat(blocks, dim="time", join="outer").astype("float32")
    combined = _standardize_latlon_names(combined)
    combined.attrs["source_hash"] = file_list_hash

    if var_name not in combined:
        raise KeyError(f"Variable '{var_name}' absente après concaténation.")

    logger.info(
        f"Concaténé: dims={combined.dims}, dtype={combined[var_name].dtype}"
    )

    # ------------------------------------------------------------------
    # 7) Garantir une sortie Tb
    # ------------------------------------------------------------------
    data_in = combined[var_name].astype("float32")

    data_tb, detected_kind = ensure_tb_data(
        data_in,
        source_kind=source_kind,
        target_name=output_var_name,
        olr_min=olr_min,
        olr_max=olr_max,
    )

    data_tb.name = output_var_name
    data_tb.attrs["source_hash"] = file_list_hash

    logger.info(
        "Variable thermique détectée: %s -> sortie '%s' [%s]",
        detected_kind,
        output_var_name,
        data_tb.dtype,
    )

    # ------------------------------------------------------------------
    # 8) Seuillage sur la Tb
    # ------------------------------------------------------------------
    vol_thresh, sel_lats = threshold_data(
        data_tb,
        lat_range=lat_range,
        bt_min=bt_min,
        bt_max=bt_max,
    )

    vol_thresh.name = output_var_name
    vol_thresh.attrs.update(data_tb.attrs)
    vol_thresh.attrs["source_hash"] = file_list_hash

    logger.info(
        f"Après seuil : {vol_thresh.sizes}, latitudes retenues : {len(sel_lats)}"
    )

    # ------------------------------------------------------------------
    # 8.1) Correction adaptative des flashs temporels (optionnel)
    # ------------------------------------------------------------------
    if remove_blinking:
        logger.info("Détection/correction adaptative des flashs temporels ...")

        nt_before = int(vol_thresh.sizes["time"])

        vol_thresh, removed_times = _clean_flash_artifacts_adaptive(vol_thresh)

        nt_after = int(vol_thresh.sizes["time"])
        n_removed = nt_before - nt_after

        vol_thresh.name = output_var_name
        vol_thresh.attrs.update(data_tb.attrs)
        vol_thresh.attrs["source_hash"] = file_list_hash
        vol_thresh.attrs["blink_method"] = "adaptive_temporal_mad_v1"
        vol_thresh.attrs["blink_removed_n_times"] = int(n_removed)

        logger.info(
            f"Correction flash terminée : {n_removed} pas de temps supprimés."
        )
        if len(removed_times) > 0:
            logger.info(
                f"Premiers temps supprimés : {removed_times[:min(5, len(removed_times))]}"
            )

    # ------------------------------------------------------------------
    # 9) Rechunk
    # ------------------------------------------------------------------
    if read_chunks is None:
        uniform_chunks = {
            "time": 1,
            "latitude": min(512, vol_thresh.sizes["latitude"]),
            "longitude": min(512, vol_thresh.sizes["longitude"]),
        }
    else:
        uniform_chunks = dict(read_chunks)

    vol_thresh = vol_thresh.chunk(uniform_chunks)
    logger.info(f"Chunks uniformisés pour Zarr : {uniform_chunks}")

    # ------------------------------------------------------------------
    # 10) Écriture ZIP Zarr compressé
    # ------------------------------------------------------------------
    logger.info("Démarrage de l'écriture ...")
    store_path.parent.mkdir(parents=True, exist_ok=True)

    comp = numcodecs.Blosc(
        cname="zstd",
        clevel=gzip_level,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )

    encoding = {
        output_var_name: {
            "compressor": comp,
            "dtype": "float32",
        }
    }

    ds_out = vol_thresh.to_dataset(name=output_var_name)

    ds_out.attrs["source_hash"] = file_list_hash
    ds_out.attrs["thermal_output_kind"] = "tb"
    ds_out.attrs["thermal_detected_input_kind"] = detected_kind

    ds_out[output_var_name].attrs["source_hash"] = file_list_hash
    ds_out[output_var_name].attrs["thermal_output_kind"] = "tb"
    ds_out[output_var_name].attrs["thermal_detected_input_kind"] = detected_kind

    zstore = zarr.ZipStore(str(store_path), mode="w")
    try:
        ds_out.to_zarr(store=zstore, consolidated=True, encoding=encoding)
    finally:
        zstore.close()

    # ------------------------------------------------------------------
    # 11) Réouverture du store final
    # ------------------------------------------------------------------
    ds_final = xr.open_zarr(store_path, consolidated=True)

    if output_var_name not in ds_final:
        raise KeyError(f"Variable '{output_var_name}' absente dans le store final.")

    data = ds_final[output_var_name]
    data_hash = data.attrs.get("source_hash") or ds_final.attrs.get("source_hash", "")
    data.attrs["source_hash"] = data_hash

    result = (
        data,
        ds_final.coords["time"],
        ds_final.coords["latitude"],
        ds_final.coords["longitude"],
    )

    # ------------------------------------------------------------------
    # 12) Libération mémoire
    # ------------------------------------------------------------------
    del blocks, combined, ds_final, file_list, groups, vol_thresh, ds_out, data_tb, data_in
    gc.collect()

    return result


def load_data(
    file_path: Union[str, List[str]],
    **kwargs
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Alias pour load_multiple_nc_files."""
    return load_multiple_nc_files(file_path, **kwargs)


#=========== Etapes d'interpolation cinématique ===========
# ----------------------------------------------------------
# Version structure-aware auto-adaptative :
#   - support continu via signed distance field (SDF)
#   - flow combiné Tb + SDF
#   - réglages morphologiques estimés automatiquement à chaque paire A->B
#   - interpolation morphologique du support
#   - interpolation thermique avec terme résiduel auto-déduit
# ----------------------------------------------------------

def _safe_to_datetimeindex(times) -> pd.DatetimeIndex:
    """Convertit un tableau de timestamps hétérogènes en DatetimeIndex."""
    try:
        return pd.to_datetime(times, errors="raise")
    except Exception:
        if hasattr(times, "to_datetimeindex"):
            try:
                return times.to_datetimeindex()
            except Exception:
                pass
        try:
            return pd.to_datetime([str(t) for t in times], errors="raise")
        except Exception as e:
            raise ValueError(f"Impossible de convertir les timestamps : {e}") from e


def _set_cv2_params(obj: Any, **kw) -> Any:
    """
    Applique dynamiquement des paramètres OpenCV via les setters setXxx(...)
    si ceux-ci existent sur l'objet.
    """
    for k, v in kw.items():
        setter = f"set{k[0].upper()}{k[1:]}"
        if hasattr(obj, setter):
            getattr(obj, setter)(v)
    return obj


def _compute_optical_flow(
    img0: np.ndarray,
    img1: np.ndarray,
    alg: str = "dis",
    flow_obj: Any = None,
    preset: str = "fast",
    initial_flow: Optional[np.ndarray] = None,
    **kw,
) -> np.ndarray:
    """
    Retourne un champ de flux (h, w, 2) en float32.

    Remarques :
      - DIS attend des images 8-bit mono-canal.
      - Les flow_kwargs sont réellement appliqués à DIS.
    """
    alg = alg.lower()

    # --- DIS ---------------------------------------------------------
    if alg == "dis":
        # on reçoit img0/img1 en float32 normalisés [0,1] ou en uint8
        # convertir en uint8 si besoin
        if img0.dtype != np.uint8:
            src0 = (np.clip(img0, 0, 1) * 255).astype(np.uint8)
            src1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
        else:
            src0, src1 = img0, img1

        if flow_obj is None:
            if preset == "ultrafast":
                flow_obj = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            elif preset == "fast":
                flow_obj = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            else:
                flow_obj = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

        flow_obj = _set_cv2_params(flow_obj, **kw)

        if initial_flow is not None:
            initial_flow = initial_flow.astype(np.float32, copy=False)

        flow = flow_obj.calc(src0, src1, initial_flow)
        return flow.astype(np.float32)

    # --- TV-L1 -------------------------------------------------------
    elif alg in {"tvl1", "tv-l1", "tv_l1"}:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        tvl1 = _set_cv2_params(tvl1, **kw)
        return tvl1.calc(img0, img1, None).astype(np.float32)

    # --- Farneback ---------------------------------------------------
    elif alg in {"farneback", "fb"}:
        p = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        p.update({k: v for k, v in kw.items() if k in p})

        flow0 = None
        if initial_flow is not None:
            flow0 = initial_flow.astype(np.float32, copy=False)
            p["flags"] = p["flags"] | cv2.OPTFLOW_USE_INITIAL_FLOW

        flow = cv2.calcOpticalFlowFarneback(
            img0, img1, flow0,
            p["pyr_scale"], p["levels"], p["winsize"],
            p["iterations"], p["poly_n"], p["poly_sigma"], p["flags"]
        )
        return flow.astype(np.float32)

    else:
        raise ValueError(f"Optical-flow inconnu : {alg}")


def _upsample_flow(
    flow_ds: np.ndarray,
    shape_full: Tuple[int, int],
    scale: int,
) -> np.ndarray:
    """Passe d’une résolution réduite à la résolution d’origine."""
    if scale == 1:
        return flow_ds.astype(np.float32, copy=False)
    flow_up = cv2.resize(flow_ds, shape_full[::-1], interpolation=cv2.INTER_LINEAR)
    return (flow_up * scale).astype(np.float32, copy=False)


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """
    Signed distance field:
      >0 à l'intérieur du support
      <0 à l'extérieur
    Aucun label objet n'est utilisé.
    """
    m = np.asarray(mask > 0.5)
    if m.size == 0:
        return np.zeros_like(mask, dtype=np.float32)

    d_in = ndimage.distance_transform_edt(m)
    d_out = ndimage.distance_transform_edt(~m)
    phi = d_in - d_out
    return phi.astype(np.float32)


def _phi_to_gray(phi: np.ndarray, scale: float = 6.0) -> np.ndarray:
    """
    Convertit un signed distance field en image [0,1]
    exploitable par l'optical flow.
    """
    scale = max(float(scale), 1e-3)
    return (0.5 + 0.5 * np.tanh(phi / scale)).astype(np.float32)


def _phi_to_soft_mask(phi: np.ndarray, softness: float = 2.0) -> np.ndarray:
    """
    Convertit un signed distance field en masque doux [0,1].
    """
    softness = max(float(softness), 1e-3)
    z = np.clip(phi / softness, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    """Centroïde d'un masque binaire/soft > 0.5."""
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size < 16:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def _robust_flow_vector(flow: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Vecteur robuste représentatif du flow sur un support.
    """
    sel = mask > 0.5
    if np.count_nonzero(sel) < 16:
        return np.array([0.0, 0.0], dtype=np.float32)

    fx = flow[..., 0][sel]
    fy = flow[..., 1][sel]

    fx = fx[np.isfinite(fx)]
    fy = fy[np.isfinite(fy)]

    if fx.size < 16 or fy.size < 16:
        return np.array([0.0, 0.0], dtype=np.float32)

    return np.array([np.median(fx), np.median(fy)], dtype=np.float32)


def _build_stationary_maps(
    step_flow: np.ndarray,
    total_steps: int,
    gx: np.ndarray,
    gy: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Construit les cartes target->source par composition discrète
    d'un champ de déplacement stationnaire.

    maps[n] = map après n sous-pas
    """
    maps = [(gx.copy(), gy.copy())]

    map_x = gx.copy()
    map_y = gy.copy()

    fx = step_flow[..., 0].astype(np.float32, copy=False)
    fy = step_flow[..., 1].astype(np.float32, copy=False)

    for _ in range(total_steps):
        sfx = cv2.remap(
            fx, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        sfy = cv2.remap(
            fy, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        map_x = map_x - sfx
        map_y = map_y - sfy
        maps.append((map_x.copy(), map_y.copy()))

    return maps


def cinem_interp_zip(
    src_zarr: Union[str, Path, xr.DataArray, xr.Dataset],
    dst_store: Union[str, Path],
    var_name: str = "Harmonized_irBT",
    freq: str = "30min",
    of_alg: str = "dis",
    preset: str = "fast",
    flow_kwargs: Optional[Dict] = None,
    downscale: int = 1,
    chunks_out: Optional[Dict[str, int]] = None,
    compressor: Optional[numcodecs.abc.Codec] = None,
    max_gap_steps: int = 6,
    fallback_gap_steps: int = 8,
    fallback_mode: str = "flow",   # "flow", "linear", "hold"
    flow_gain: float = 1.0,
    auto_tune: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Interpolation structure-aware auto-adaptative.

    Principe:
      1) Le support des structures est représenté par un signed distance field (SDF)
         construit à partir du masque des pixels finis.
      2) On calcule deux flows:
           - un flow thermique sur Tb normalisée
           - un flow géométrique sur le SDF
      3) Les poids de combinaison sont estimés automatiquement à chaque paire A->B.
      4) Le support au temps intermédiaire est obtenu par interpolation
         des SDF advectés depuis A et B.
      5) Le contenu Tb est interpolé par advection + blending
         + terme résiduel estimé automatiquement.

    Cela reste dense, sans objets ni labels, donc compatible
    avec un prétraitement avant tracking.
    """
    flow_kwargs = flow_kwargs or {}
    compressor = compressor or numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
    )
    chunks_out = chunks_out or {"time": 1, "latitude": 512, "longitude": 512}

    # --- Ouverture source -------------------------------------------
    if isinstance(src_zarr, (str, Path)):
        ds_in = xr.open_zarr(
            str(src_zarr),
            chunks={"time": 1, "latitude": -1, "longitude": -1},
        )
        da_src = ds_in[var_name] if var_name in ds_in else list(ds_in.data_vars.values())[0]
        source_hash = da_src.attrs.get("source_hash") or ds_in.attrs.get("source_hash", "")

    elif isinstance(src_zarr, xr.DataArray):
        da_src = src_zarr.chunk({"time": 1, "latitude": -1, "longitude": -1})
        source_hash = da_src.attrs.get("source_hash", "")

    else:
        da_src = src_zarr[var_name].chunk({"time": 1, "latitude": -1, "longitude": -1})
        source_hash = (
            src_zarr[var_name].attrs.get("source_hash")
            or getattr(src_zarr, "attrs", {}).get("source_hash", "")
        )

    # --- Temps : conversion + tri + vérification doublons -----------
    time_idx = pd.DatetimeIndex(_safe_to_datetimeindex(da_src.time.values))
    da_src = da_src.assign_coords(time=time_idx).sortby("time")
    time_idx = pd.DatetimeIndex(_safe_to_datetimeindex(da_src.time.values))

    if time_idx.has_duplicates:
        dup = pd.DatetimeIndex(time_idx[time_idx.duplicated()]).unique()
        raise ValueError(
            "Des timestamps dupliqués ont été détectés dans la source. "
            f"Exemples: {dup[:5].tolist()}"
        )

    times_pd = time_idx
    if len(times_pd) < 2:
        raise ValueError("Pas assez de timestamps pour interpoler (len < 2).")

    lat_src = np.asarray(da_src.latitude.values, dtype=np.float32)
    lon_src = np.asarray(da_src.longitude.values, dtype=np.float32)

    # --- Horloge cible régulière ------------------------------------
    freq_norm = freq.replace("T", "min") if str(freq).endswith("T") else str(freq)
    freq_off = pd.tseries.frequencies.to_offset(freq_norm)

    tstart = pd.Timestamp(times_pd[0]).floor(freq_off)
    tend = pd.Timestamp(times_pd[-1]).ceil(freq_off)
    full_time = pd.date_range(tstart, tend, freq=freq_off)

    nt, ny, nx = len(full_time), da_src.shape[1], da_src.shape[2]

    # clip chunks de sortie
    chunks_out = {
        "time": int(min(max(1, int(chunks_out["time"])), nt)),
        "latitude": int(min(max(1, int(chunks_out["latitude"])), ny)),
        "longitude": int(min(max(1, int(chunks_out["longitude"])), nx)),
    }

    # --- Grilles pour remap -----------------------------------------
    h, w = int(ny), int(nx)
    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    def _sample_img(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
        return cv2.remap(
            img.astype(np.float32, copy=False),
            map_x.astype(np.float32, copy=False),
            map_y.astype(np.float32, copy=False),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _sample_mask(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
        return cv2.remap(
            mask.astype(np.float32, copy=False),
            map_x.astype(np.float32, copy=False),
            map_y.astype(np.float32, copy=False),
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    def _robust_q(x: np.ndarray, q: float, default: float = 0.0) -> float:
        x = np.asarray(x, dtype=np.float32)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float(default)
        return float(np.quantile(x, q))

    def _robust_rmse(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        d = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        if mask is not None:
            m = np.asarray(mask) > 0
            d = d[m]
        d = d[np.isfinite(d)]
        if d.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(d * d)))

    def _robust_scale(x: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        x = np.asarray(x, dtype=np.float32)
        if mask is not None:
            m = np.asarray(mask) > 0
            x = x[m]
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 1.0
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        sig = 1.4826 * mad
        return float(max(sig, 1e-3))

    def _direct_map_from_flow(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map target->source correspondant à un flow total.
        """
        return (
            (gx - flow[..., 0]).astype(np.float32),
            (gy - flow[..., 1]).astype(np.float32),
        )

    # --- Rattachement des temps natifs à la grille -------------------
    try:
        tol = pd.Timedelta(freq_off.delta) / 2
    except Exception:
        tol = pd.Timedelta(freq_norm) / 2

    idx_native = full_time.get_indexer(times_pd, method="nearest", tolerance=tol)
    ok = idx_native >= 0
    if not np.all(ok):
        bad = times_pd[~ok]
        raise ValueError(
            f"Certains temps source ne tombent pas sur la grille {freq_norm} "
            f"(tol={tol}) : exemples={bad[:5].tolist()}"
        )

    # --- Bornes globales approx (~1 % des temps) --------------------
    step = max(1, len(times_pd) // 100)
    sample = da_src.isel(time=slice(None, None, step))
    gmin, gmax = map(float, dask.compute(sample.min(), sample.max()))
    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        gmin, gmax = 180.0, 320.0
    rng = gmax - gmin

    # --- Hash d'interpolation ---------------------------------------
    params = dict(
        freq=freq_norm,
        of_alg=of_alg,
        preset=preset,
        flow_kwargs=flow_kwargs,
        downscale=downscale,
        chunks_out=chunks_out,
        compressor=repr(compressor),
        source_hash=source_hash,
        gmin=gmin,
        gmax=gmax,
        max_gap_steps=max_gap_steps,
        fallback_gap_steps=fallback_gap_steps,
        fallback_mode=fallback_mode,
        flow_gain=flow_gain,
        auto_tune=auto_tune,
        interp_method_version="structure_aware_sdf_residual_auto_v2",
        time_start=str(times_pd[0]),
        time_end=str(times_pd[-1]),
        n_src_times=int(len(times_pd)),
    )
    interp_hash = hashlib.md5(
        json.dumps(params, sort_keys=True, default=str).encode()
    ).hexdigest()

    dst_zip = Path(dst_store).with_suffix(".zip")
    if dst_zip.exists():
        try:
            ds_cached = xr.open_zarr(dst_zip, consolidated=True, decode_cf=False)
            if ds_cached.attrs.get("interp_hash") == interp_hash:
                ds_cached = ds_cached.set_coords(["time", "latitude", "longitude"])
                da_filled = ds_cached["da_filled"]
                da_filled.attrs["interp_hash"] = ds_cached.attrs.get("interp_hash", "")
                da_filled.attrs["source_hash"] = ds_cached.attrs.get("source_hash", "")
                return (
                    da_filled,
                    ds_cached["mask_interp_time"].astype(bool),
                    ds_cached["time"],
                )
        except Exception:
            try:
                dst_zip.unlink()
            except Exception:
                pass

    # --- Store temporaire -------------------------------------------
    tmp_dir = Path(dst_store).with_suffix(".zarr")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    root = zarr.group(store=zarr.DirectoryStore(tmp_dir), overwrite=True)
    root.attrs["interp_hash"] = interp_hash
    root.attrs["source_hash"] = source_hash

    # data
    arr = root.create_dataset(
        "da_filled",
        shape=(nt, ny, nx),
        chunks=(chunks_out["time"], chunks_out["latitude"], chunks_out["longitude"]),
        dtype="float32",
        compressor=compressor,
        fill_value=np.nan,
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "latitude", "longitude"]
    arr.attrs["interp_hash"] = interp_hash
    arr.attrs["source_hash"] = source_hash

    # coords
    zt = root.create_dataset(
        "time",
        data=full_time.values.astype("datetime64[ns]"),
        dtype="datetime64[ns]",
    )
    zt.attrs["_ARRAY_DIMENSIONS"] = ["time"]

    zl = root.create_dataset(
        "latitude",
        data=lat_src.astype("float32"),
        dtype="float32",
        compressor=compressor,
    )
    zl.attrs["_ARRAY_DIMENSIONS"] = ["latitude"]

    zg = root.create_dataset(
        "longitude",
        data=lon_src.astype("float32"),
        dtype="float32",
        compressor=compressor,
    )
    zg.attrs["_ARRAY_DIMENSIONS"] = ["longitude"]

    # masque temporel (True si timestamp non-natif)
    native_set = set(times_pd.values.astype("datetime64[ns]"))
    mask_bool = np.array(
        [t.astype("datetime64[ns]") not in native_set for t in full_time.values],
        dtype="uint8",
    )
    zm = root.create_dataset(
        "mask_interp_time",
        data=mask_bool,
        dtype="uint8",
        compressor=compressor,
    )
    zm.attrs["_ARRAY_DIMENSIONS"] = ["time"]

    # --- Objet OF réutilisable --------------------------------------
    flow_obj = None
    if of_alg.lower() == "dis":
        if preset == "ultrafast":
            flow_obj = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        elif preset == "fast":
            flow_obj = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        else:
            flow_obj = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow_obj = _set_cv2_params(flow_obj, **flow_kwargs)

    def _auto_pair_controls(
        imgA_norm: np.ndarray,
        imgB_norm: np.ndarray,
        mA: np.ndarray,
        mB: np.ndarray,
        phiA: np.ndarray,
        phiB: np.ndarray,
        f_img_fwd: np.ndarray,
        f_img_bwd: np.ndarray,
        f_sdf_fwd: np.ndarray,
        f_sdf_bwd: np.ndarray,
    ) -> Dict[str, float]:
        """
        Estime automatiquement les contrôles structure-aware à partir des données
        de la paire A->B, sans réglage manuel.
        """
        union = (np.maximum(mA, mB) > 0.0)

        # --- qualité géométrique comparée : flow Tb vs flow SDF ------
        map_img_x, map_img_y = _direct_map_from_flow(f_img_fwd)
        map_sdf_x, map_sdf_y = _direct_map_from_flow(f_sdf_fwd)

        phiA_to_B_img = _sample_img(phiA, map_img_x, map_img_y)
        phiA_to_B_sdf = _sample_img(phiA, map_sdf_x, map_sdf_y)

        err_geom_img = _robust_rmse(phiA_to_B_img, phiB, union)
        err_geom_sdf = _robust_rmse(phiA_to_B_sdf, phiB, union)

        # poids automatique du flow géométrique
        w_struct = err_geom_img / (err_geom_img + err_geom_sdf + 1e-6)
        w_struct = float(np.clip(w_struct, 0.15, 0.85))

        # amplitude caractéristique du déplacement
        disp_img = np.hypot(f_img_fwd[..., 0], f_img_fwd[..., 1])
        disp_sdf = np.hypot(f_sdf_fwd[..., 0], f_sdf_fwd[..., 1])

        disp_char = 0.5 * (
            _robust_q(disp_img[union], 0.50, default=1.0) +
            _robust_q(disp_sdf[union], 0.50, default=1.0)
        )
        band = float(max(disp_char, 1.0))

        # douceur du support dérivée du déplacement caractéristique
        softness = float(max(0.75, 0.50 * band))

        # flow préliminaire combiné pour estimer résidu et gain
        edgeA = np.exp(-np.abs(phiA) / max(band, 1e-3)).astype(np.float32)
        edgeB = np.exp(-np.abs(phiB) / max(band, 1e-3)).astype(np.float32)
        edge = np.maximum(edgeA, edgeB)[..., None]

        wloc = np.float32(w_struct) * edge
        fwd0 = (1.0 - wloc) * f_img_fwd + wloc * f_sdf_fwd
        bwd0 = (1.0 - wloc) * f_img_bwd + wloc * f_sdf_bwd

        # gain global automatique via centroïde du support
        cA = _mask_centroid(mA)
        cB = _mask_centroid(mB)

        gain_fwd = 1.0
        gain_bwd = 1.0

        if cA is not None and cB is not None:
            obs = cB - cA
            obs_norm = float(np.hypot(obs[0], obs[1]))

            pred_fwd = _robust_flow_vector(fwd0, mA)
            pred_bwd = _robust_flow_vector(bwd0, mB)

            pred_fwd_norm = float(np.hypot(pred_fwd[0], pred_fwd[1]))
            pred_bwd_norm = float(np.hypot(pred_bwd[0], pred_bwd[1]))

            if obs_norm > 1e-3 and pred_fwd_norm > 1e-3:
                gain_fwd = float(np.clip(obs_norm / pred_fwd_norm, 0.90, 1.35))

            if obs_norm > 1e-3 and pred_bwd_norm > 1e-3:
                gain_bwd = float(np.clip(obs_norm / pred_bwd_norm, 0.90, 1.35))

        # poids du résidu automatique
        map_fwd_x, map_fwd_y = _direct_map_from_flow(fwd0 * np.float32(gain_fwd))
        A_to_B = _sample_img(imgA_norm, map_fwd_x, map_fwd_y)

        adv_err = _robust_rmse(A_to_B, imgB_norm, union)
        sigB = _robust_scale(imgB_norm, union)

        residual_weight = adv_err / (adv_err + sigB + 1e-6)
        residual_weight = float(np.clip(residual_weight, 0.05, 0.60))

        return {
            "band": band,
            "w_struct": w_struct,
            "softness": softness,
            "residual_weight": residual_weight,
            "gain_fwd": gain_fwd,
            "gain_bwd": gain_bwd,
        }

    def _compute_structure_aware_bidir_flow(
        imgA_norm: np.ndarray,
        imgB_norm: np.ndarray,
        mA: np.ndarray,
        mB: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Flow combiné Tb + SDF, avec réglages auto-déduits des données de la paire.
        """
        phiA = _signed_distance(mA)
        phiB = _signed_distance(mB)

        # échelle SDF -> image estimée automatiquement
        support_size = max(
            _robust_q(np.abs(phiA[mA > 0]), 0.75, default=4.0),
            _robust_q(np.abs(phiB[mB > 0]), 0.75, default=4.0),
            1.0,
        )
        sdf_scale = float(max(1.0, 0.5 * support_size))

        sdfA = _phi_to_gray(phiA, scale=sdf_scale)
        sdfB = _phi_to_gray(phiB, scale=sdf_scale)

        if downscale > 1:
            h_ds = max(1, imgA_norm.shape[0] // downscale)
            w_ds = max(1, imgA_norm.shape[1] // downscale)

            imgA_ds = cv2.resize(imgA_norm, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
            imgB_ds = cv2.resize(imgB_norm, (w_ds, h_ds), interpolation=cv2.INTER_AREA)

            sdfA_ds = cv2.resize(sdfA, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
            sdfB_ds = cv2.resize(sdfB, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
        else:
            imgA_ds, imgB_ds = imgA_norm, imgB_norm
            sdfA_ds, sdfB_ds = sdfA, sdfB

        # flow thermique
        f_img_fwd_ds = _compute_optical_flow(
            imgA_ds, imgB_ds,
            alg=of_alg,
            flow_obj=flow_obj,
            preset=preset,
            **flow_kwargs,
        )
        f_img_bwd_ds = _compute_optical_flow(
            imgB_ds, imgA_ds,
            alg=of_alg,
            flow_obj=flow_obj,
            preset=preset,
            **flow_kwargs,
        )

        # flow géométrique
        f_sdf_fwd_ds = _compute_optical_flow(
            sdfA_ds, sdfB_ds,
            alg=of_alg,
            flow_obj=flow_obj,
            preset=preset,
            **flow_kwargs,
        )
        f_sdf_bwd_ds = _compute_optical_flow(
            sdfB_ds, sdfA_ds,
            alg=of_alg,
            flow_obj=flow_obj,
            preset=preset,
            **flow_kwargs,
        )

        f_img_fwd = _upsample_flow(f_img_fwd_ds, imgA_norm.shape, downscale)
        f_img_bwd = _upsample_flow(f_img_bwd_ds, imgB_norm.shape, downscale)
        f_sdf_fwd = _upsample_flow(f_sdf_fwd_ds, imgA_norm.shape, downscale)
        f_sdf_bwd = _upsample_flow(f_sdf_bwd_ds, imgB_norm.shape, downscale)

        ctrl = _auto_pair_controls(
            imgA_norm, imgB_norm, mA, mB, phiA, phiB,
            f_img_fwd, f_img_bwd, f_sdf_fwd, f_sdf_bwd
        )

        band = ctrl["band"]
        w_struct = ctrl["w_struct"]

        edgeA = np.exp(-np.abs(phiA) / max(band, 1e-3)).astype(np.float32)
        edgeB = np.exp(-np.abs(phiB) / max(band, 1e-3)).astype(np.float32)
        edge = np.maximum(edgeA, edgeB)[..., None]

        wloc = np.float32(w_struct) * edge

        fwd = (1.0 - wloc) * f_img_fwd + wloc * f_sdf_fwd
        bwd = (1.0 - wloc) * f_img_bwd + wloc * f_sdf_bwd

        if auto_tune:
            fwd *= np.float32(ctrl["gain_fwd"] * flow_gain)
            bwd *= np.float32(ctrl["gain_bwd"] * flow_gain)
        else:
            fwd *= np.float32(flow_gain)
            bwd *= np.float32(flow_gain)

        return (
            fwd.astype(np.float32),
            bwd.astype(np.float32),
            phiA.astype(np.float32),
            phiB.astype(np.float32),
            ctrl,
        )

    def _write_slice(idx: int, imgK: np.ndarray) -> None:
        arr[idx, :, :] = imgK.astype(np.float32, copy=False)

    def _fill_segment_linear(
        kA: int,
        kB: int,
        imgAn: np.ndarray,
        imgBn: np.ndarray,
        mA: np.ndarray,
        mB: np.ndarray,
    ) -> None:
        """
        Fallback simple en blending linéaire.
        """
        for k in range(kA + 1, kB):
            alpha = (k - kA) / (kB - kA)

            wsum = (1.0 - alpha) * mA + alpha * mB
            with np.errstate(divide="ignore", invalid="ignore"):
                outn = ((1.0 - alpha) * mA * imgAn + alpha * mB * imgBn) / wsum
            outn[wsum <= 0] = np.nan

            out = np.where(np.isfinite(outn), outn * rng + gmin, np.nan)
            _write_slice(k, out)

    def _fill_segment_structure_aware(
        kA: int,
        kB: int,
        imgAn: np.ndarray,
        imgBn: np.ndarray,
        mA: np.ndarray,
        mB: np.ndarray,
    ) -> None:
        """
        Remplissage structure-aware auto-adaptatif, sans hyperparamètres exposés.
        """
        total_steps = kB - kA
        if total_steps <= 1:
            return

        fwd, bwd, phiA, phiB, ctrl = _compute_structure_aware_bidir_flow(
            imgAn, imgBn, mA, mB
        )

        step_fwd = (fwd / np.float32(total_steps)).astype(np.float32, copy=False)
        step_bwd = (bwd / np.float32(total_steps)).astype(np.float32, copy=False)

        maps_A = _build_stationary_maps(step_fwd, total_steps, gx, gy)
        maps_B = _build_stationary_maps(step_bwd, total_steps, gx, gy)

        # résidu morphologique
        mapAB_x, mapAB_y = maps_A[total_steps]
        mapBA_x, mapBA_y = maps_B[total_steps]

        A_to_B = _sample_img(imgAn, mapAB_x, mapAB_y)
        B_to_A = _sample_img(imgBn, mapBA_x, mapBA_y)

        resB = (imgBn - A_to_B).astype(np.float32)
        resA = (imgAn - B_to_A).astype(np.float32)

        softness = ctrl["softness"] if auto_tune else max(ctrl["softness"], 1.0)
        residual_weight = ctrl["residual_weight"] if auto_tune else 0.25

        for k in range(kA + 1, kB):
            nA = k - kA
            nB = kB - k
            alpha = nA / total_steps

            mapAx, mapAy = maps_A[nA]
            mapBx, mapBy = maps_B[nB]

            # support géométrique
            phiA_t = _sample_img(phiA, mapAx, mapAy)
            phiB_t = _sample_img(phiB, mapBx, mapBy)

            softA = _phi_to_soft_mask(phiA_t, softness=softness)
            softB = _phi_to_soft_mask(phiB_t, softness=softness)

            phi_t = ((1.0 - alpha) * phiA_t + alpha * phiB_t).astype(np.float32)
            support_t = _phi_to_soft_mask(phi_t, softness=softness)

            # validité hard sans seuil calibrable
            hardA = _sample_mask(mA, mapAx, mapAy)
            hardB = _sample_mask(mB, mapBx, mapBy)
            hard_union = np.maximum(hardA, hardB) > 0.0

            # thermique advecté
            w0 = _sample_img(imgAn, mapAx, mapAy)
            w1 = _sample_img(imgBn, mapBx, mapBy)

            denom = (1.0 - alpha) * softA + alpha * softB
            with np.errstate(divide="ignore", invalid="ignore"):
                base = ((1.0 - alpha) * softA * w0 + alpha * softB * w1) / denom
            base[denom <= 1e-6] = np.nan

            # résidu auto
            resA_t = _sample_img(resA, mapAx, mapAy)
            resB_t = _sample_img(resB, mapBx, mapBy)
            source_t = (alpha * resB_t - (1.0 - alpha) * resA_t).astype(np.float32)

            outn = base + np.float32(residual_weight) * source_t * support_t
            outn = np.clip(outn, 0.0, 1.0)

            # hors support advecté => NaN
            outn[~hard_union] = np.nan

            out = np.where(np.isfinite(outn), outn * rng + gmin, np.nan)
            _write_slice(k, out)

    # --- Boucle sur paires natives ----------------------------------
    for k in tqdm(range(len(times_pd) - 1), desc="Interpolation structure-aware"):
        i0, i1 = int(idx_native[k]), int(idx_native[k + 1])

        if i1 <= i0:
            continue

        img0_raw = da_src.isel(time=k).values.astype(np.float32, copy=False)
        img1_raw = da_src.isel(time=k + 1).values.astype(np.float32, copy=False)

        if k == 0:
            _write_slice(i0, img0_raw)
        _write_slice(i1, img1_raw)

        n_missing = i1 - i0 - 1
        if n_missing <= 0:
            continue

        finite0 = np.isfinite(img0_raw)
        finite1 = np.isfinite(img1_raw)

        m0 = finite0.astype(np.float32)
        m1 = finite1.astype(np.float32)

        # Normalisation [0,1] avec fond estimé automatiquement à partir des données
        vals0 = ((img0_raw[finite0] - gmin) / rng).astype(np.float32) if np.any(finite0) else np.array([], dtype=np.float32)
        vals1 = ((img1_raw[finite1] - gmin) / rng).astype(np.float32) if np.any(finite1) else np.array([], dtype=np.float32)

        allv = np.concatenate([vals0, vals1]) if (vals0.size + vals1.size) > 0 else np.array([1.0], dtype=np.float32)
        outside_fill_value = float(np.clip(np.quantile(allv, 0.95), 0.75, 1.0))

        img0n = np.full(img0_raw.shape, np.float32(outside_fill_value), dtype=np.float32)
        img1n = np.full(img1_raw.shape, np.float32(outside_fill_value), dtype=np.float32)

        if np.any(finite0):
            img0n[finite0] = np.clip((img0_raw[finite0] - gmin) / rng, 0.0, 1.0).astype(np.float32)
        if np.any(finite1):
            img1n[finite1] = np.clip((img1_raw[finite1] - gmin) / rng, 0.0, 1.0).astype(np.float32)

        if n_missing <= max_gap_steps:
            _fill_segment_structure_aware(i0, i1, img0n, img1n, m0, m1)
            continue

        if n_missing <= fallback_gap_steps:
            if fallback_mode == "flow":
                _fill_segment_structure_aware(i0, i1, img0n, img1n, m0, m1)

            elif fallback_mode == "linear":
                _fill_segment_linear(i0, i1, img0n, img1n, m0, m1)

            elif fallback_mode == "hold":
                for kk in range(i0 + 1, i1):
                    _write_slice(kk, img0_raw)

            else:
                raise ValueError(
                    f"fallback_mode inconnu: {fallback_mode}. "
                    "Valeurs acceptées: 'flow', 'linear', 'hold'."
                )

        # sinon : on laisse NaN sur le gap

    # --- Consolidation + zip ----------------------------------------
    zarr.convenience.consolidate_metadata(tmp_dir)

    if dst_zip.exists():
        dst_zip.unlink()

    shutil.make_archive(dst_zip.with_suffix(""), "zip", tmp_dir)
    shutil.rmtree(tmp_dir)

    ds_out = xr.open_zarr(dst_zip, consolidated=True, decode_cf=False)
    ds_out = ds_out.set_coords(["time", "latitude", "longitude"])

    da_filled = ds_out["da_filled"]
    da_filled.attrs["interp_hash"] = ds_out.attrs.get("interp_hash", "")
    da_filled.attrs["source_hash"] = ds_out.attrs.get("source_hash", "")

    mask_interp = ds_out["mask_interp_time"].astype(bool)

    gc.collect()
    return da_filled, mask_interp, da_filled.time