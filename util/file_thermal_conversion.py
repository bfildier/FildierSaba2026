from __future__ import annotations

import logging
import re
from typing import Literal

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Constantes de conversion OLR -> Tb équivalent
SIGMA_SB = np.float64(5.670374419e-8)   # Stefan-Boltzmann [W m-2 K-4]
OHRING_A = np.float64(1.228)
OHRING_B = np.float64(-1.106e-3)        # [K-1]


def _units_look_like_olr(units: str) -> bool:
    u = (units or "").strip().lower()
    if not u:
        return False
    if any(s in u for s in ["w m-2", "w/m2", "w m^-2", "wm-2", "watt"]):
        return True
    return bool(re.search(r"w\s*/?\s*m\s*[-^]?\s*2", u))


def _units_look_like_kelvin(units: str) -> bool:
    u = (units or "").strip().lower()
    return u in {"k", "kelvin", "degk", "degree_k", "degrees_k"}


def infer_thermal_kind(
    da: xr.DataArray,
    *,
    var_name: str | None = None,
) -> Literal["tb", "olr"]:
    """
    Déduit si la variable ressemble à :
      - du Tb [K]
      - de l'OLR [W m-2]

    Stratégie volontairement prudente :
      - on se base d'abord sur attrs/nom de variable
      - si ambigu => on lève une erreur au lieu de deviner
    """
    name = (var_name or da.name or "").lower()
    units = str(da.attrs.get("units", ""))

    text = " ".join(
        str(da.attrs.get(k, "")).lower()
        for k in ("standard_name", "long_name", "description", "comment")
    )

    # Cas Tb
    if (
        "brightness_temperature" in text
        or "brightness temperature" in text
        or "irbt" in name
        or name == "tb"
        or name.endswith("_tb")
        or "tb_" in name
        or _units_look_like_kelvin(units)
    ):
        return "tb"

    # Cas OLR
    if (
        "outgoing_longwave" in text
        or "outgoing longwave" in text
        or "olr" in name
        or _units_look_like_olr(units)
    ):
        return "olr"

    raise ValueError(
        "Impossible d'inférer automatiquement si la variable est du Tb ou de l'OLR. "
        "Passez source_kind='tb' ou source_kind='olr'."
    )


def _olr_to_tb_numpy(
    olr: np.ndarray,
    *,
    sigma_sb: float = float(SIGMA_SB),
    a: float = float(OHRING_A),
    b: float = float(OHRING_B),
    olr_min: float = 40.0,
    olr_max: float = 500.0,
    tb_min_physical: float = 100.0,
    tb_max_physical: float = 400.0,
) -> np.ndarray:
    """
    Convertit OLR [W m-2] -> Tb équivalent [K].

    Étapes :
      1) Tf = (OLR / sigma_sb)^(1/4)
      2) résoudre b*Tb^2 + a*Tb - Tf = 0
         et prendre la racine physique

    Convention :
      - valeurs hors plage physique => NaN
      - calcul en float64, sortie float32
    """
    x = np.asarray(olr, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)

    valid = np.isfinite(x)
    valid &= (x >= olr_min) & (x <= olr_max)

    if not np.any(valid):
        return out.astype(np.float32)

    tf = np.full(x.shape, np.nan, dtype=np.float64)
    tf[valid] = np.power(x[valid] / sigma_sb, 0.25)

    disc = a * a + 4.0 * b * tf
    valid &= np.isfinite(disc) & (disc >= 0.0)

    # Racine physique
    out[valid] = (-a + np.sqrt(disc[valid])) / (2.0 * b)

    valid_tb = np.isfinite(out) & (out >= tb_min_physical) & (out <= tb_max_physical)
    out[~valid_tb] = np.nan

    return out.astype(np.float32)


def olr_to_tb(
    da_olr: xr.DataArray,
    *,
    target_name: str = "Harmonized_irBT",
    olr_min: float = 40.0,
    olr_max: float = 500.0,
    keep_attrs: bool = True,
) -> xr.DataArray:
    """
    Version xarray/dask-safe de la conversion OLR -> Tb.
    """
    out = xr.apply_ufunc(
        _olr_to_tb_numpy,
        da_olr.astype("float32"),
        input_core_dims=[[]],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=False,
        output_dtypes=[np.float32],
        kwargs=dict(
            olr_min=float(olr_min),
            olr_max=float(olr_max),
        ),
    )

    out.name = target_name

    if keep_attrs:
        out.attrs = dict(da_olr.attrs)

    out.attrs.update(
        {
            "units": "K",
            "thermal_input_kind": "olr",
            "thermal_output_kind": "tb",
            "converted_from_name": da_olr.name,
            "converted_from_units": da_olr.attrs.get("units"),
            "olr_to_tb_method": "Ohring_Gruber_1984_via_flux_temperature",
            "olr_to_tb_formula": "Tf=(OLR/sigma_sb)^0.25 ; b*Tb^2 + a*Tb - Tf = 0",
            "olr_to_tb_sigma_sb": float(SIGMA_SB),
            "olr_to_tb_a": float(OHRING_A),
            "olr_to_tb_b": float(OHRING_B),
            "olr_valid_min": float(olr_min),
            "olr_valid_max": float(olr_max),
        }
    )
    return out


def ensure_tb_data(
    da: xr.DataArray,
    *,
    source_kind: Literal["auto", "tb", "olr"] = "auto",
    target_name: str = "Harmonized_irBT",
    olr_min: float = 40.0,
    olr_max: float = 500.0,
) -> tuple[xr.DataArray, str]:
    """
    Garantit que la sortie est du Tb.
    Retourne (dataarray_tb, kind_detected).
    """
    kind = infer_thermal_kind(da) if source_kind == "auto" else source_kind

    if kind == "tb":
        out = da.astype("float32")
        out.name = target_name
        out.attrs = dict(da.attrs)
        out.attrs["units"] = "K"
        out.attrs["thermal_input_kind"] = "tb"
        out.attrs["thermal_output_kind"] = "tb"
        return out, kind

    if kind == "olr":
        out = olr_to_tb(
            da,
            target_name=target_name,
            olr_min=olr_min,
            olr_max=olr_max,
            keep_attrs=True,
        )
        return out, kind

    raise ValueError(f"source_kind invalide: {source_kind}")