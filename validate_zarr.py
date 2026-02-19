#!/usr/bin/env python3

"""Validator for grouped spectral Zarr schema (signal/continuum/representations)."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import zarr

try:
    import polars as pl

    HAS_POLARS = True
except Exception:
    HAS_POLARS = False


# -----------------------------
# Helpers
# -----------------------------

def fail(msg: str):
    print(f"\n❌ VALIDATION FAILED: {msg}\n")
    sys.exit(1)


def warn(msg: str):
    print(f"⚠️  WARNING: {msg}")


def ok(msg: str):
    print(f"✅ {msg}")


def _read_index_len(index_file: Path) -> int:
    if HAS_POLARS:
        return len(pl.read_parquet(index_file))
    try:
        import pandas as pd  # type: ignore

        return len(pd.read_parquet(index_file))
    except Exception as exc:
        fail(
            "Could not read index.parquet. Install `polars` or `pandas` with parquet support. "
            f"Underlying error: {exc}"
        )


# -----------------------------
# Core Checks
# -----------------------------

def check_required_groups(root):
    required = ["signal", "continuum", "representations", "params", "metadata"]
    for name in required:
        if name not in root:
            fail(f"Missing required group: '{name}'")
    ok("Required groups present")


def check_required_arrays(root):
    required = [
        "signal/flux_raw",
        "signal/wavelength",
        "signal/uncertainty_raw",
        "continuum/continuum_model",
        "continuum/continuum_uncertainty",
        "continuum/method",
    ]
    for name in required:
        if name not in root:
            fail(f"Missing required array: '{name}'")
    ok("Required arrays present")


def check_units(root):
    numeric_arrays = [
        "signal/flux_raw",
        "signal/wavelength",
        "signal/uncertainty_raw",
        "continuum/continuum_model",
        "continuum/continuum_uncertainty",
    ]
    has_norm_flux = "representations/normalized_flux" in root
    has_norm_unc = "representations/normalized_uncertainty" in root
    if has_norm_flux != has_norm_unc:
        fail(
            "representations/normalized_flux and representations/normalized_uncertainty "
            "must be both present or both absent"
        )
    if has_norm_flux:
        numeric_arrays.append("representations/normalized_flux")
        numeric_arrays.append("representations/normalized_uncertainty")

    for name in numeric_arrays:
        arr = root[name]
        if "units" not in arr.attrs:
            fail(f"{name} missing 'units' attribute")

    ok("Units present")


def check_shapes(root, expected_nspec=None):
    flux = root["signal/flux_raw"]
    if flux.ndim != 2:
        fail("signal/flux_raw must be 2D (N_spec, N_pix)")

    nspec, npix = flux.shape
    if expected_nspec is not None and nspec != expected_nspec:
        fail(f"N_spec mismatch: expected {expected_nspec}, got {nspec}")

    wave = root["signal/wavelength"]
    if wave.ndim == 1:
        if wave.shape[0] != npix:
            fail("signal/wavelength length != N_pix")
    elif wave.ndim == 2:
        if wave.shape != flux.shape:
            fail("signal/wavelength (2D) must match signal/flux_raw shape")
    else:
        fail("signal/wavelength must be 1D or 2D")

    for name in (
        "signal/uncertainty_raw",
        "continuum/continuum_model",
        "continuum/continuum_uncertainty",
    ):
        arr = root[name]
        if arr.ndim != 2 or arr.shape != flux.shape:
            fail(f"{name} must be 2D with shape ({nspec}, {npix})")

    has_norm_flux = "representations/normalized_flux" in root
    has_norm_unc = "representations/normalized_uncertainty" in root
    if has_norm_flux != has_norm_unc:
        fail(
            "representations/normalized_flux and representations/normalized_uncertainty "
            "must be both present or both absent"
        )
    if has_norm_flux:
        arr = root["representations/normalized_flux"]
        if arr.ndim != 2 or arr.shape != flux.shape:
            fail(f"representations/normalized_flux must be 2D with shape ({nspec}, {npix})")
        arr_unc = root["representations/normalized_uncertainty"]
        if arr_unc.ndim != 2 or arr_unc.shape != flux.shape:
            fail(f"representations/normalized_uncertainty must be 2D with shape ({nspec}, {npix})")

    method = root["continuum/method"]
    if method.shape != (nspec,):
        fail(f"continuum/method must have shape ({nspec},)")

    ok(f"Shapes valid (N_spec={nspec}, N_pix={npix})")
    return nspec, npix


def check_params_alignment(root, nspec):
    params = root["params"]
    for name, arr in params.arrays():
        if arr.shape != (nspec,):
            fail(f"Parameter '{name}' misaligned: {arr.shape} != ({nspec},)")
    ok("Parameter alignment valid")


def check_metadata_alignment(root, nspec):
    meta = root["metadata"]
    for required in ("gaia_id", "gaia_dr"):
        if required not in meta:
            fail(f"Missing required metadata field: metadata/{required}")

    for name, arr in meta.arrays():
        if arr.shape != (nspec,):
            fail(f"Metadata '{name}' misaligned: {arr.shape} != ({nspec},)")
    ok("Metadata alignment valid")


def sample_signal(root, nspec, samples=64):
    flux = root["signal/flux_raw"]
    if nspec == 0:
        fail("Dataset contains zero spectra")

    idx = random.sample(range(nspec), min(samples, nspec))
    nan_count = 0
    zero_count = 0

    for i in idx:
        row = flux[i]
        if not np.isfinite(row).any():
            nan_count += 1
        if np.allclose(np.where(np.isfinite(row), row, 0.0), 0.0):
            zero_count += 1

    if nan_count > 0:
        fail(f"{nan_count}/{len(idx)} sampled signal rows are all-NaN")
    if zero_count > 0:
        warn(f"{zero_count}/{len(idx)} sampled signal rows are all zeros")

    ok("Sampled signal rows look finite")


def check_chunks(root):
    flux = root["signal/flux_raw"]
    chunks = flux.chunks

    if chunks is None:
        warn("signal/flux_raw is not chunked")
        return

    if chunks[0] <= 4:
        warn(f"Very small chunk size detected: {chunks}")

    ok(f"Chunking looks reasonable: {chunks}")


def check_provenance(root):
    required = ["schema_version", "created_at", "ingest_pipeline", "git_commit"]
    missing = [k for k in required if k not in root.attrs]
    if missing:
        fail(f"Missing required provenance fields: {missing}")
    ok("Provenance metadata present")


def check_index(path, nspec):
    index_file = Path(path) / "index.parquet"
    if not index_file.exists():
        fail("index.parquet not found")

    n_index = _read_index_len(index_file)
    if n_index != nspec:
        fail(f"Index mismatch: len(index)={n_index} != N_spec={nspec}")

    ok("Index parity valid")


def check_dtype(root):
    flux = root["signal/flux_raw"]
    if flux.dtype not in (np.float32, np.float64):
        warn(f"Unexpected dtype for signal/flux_raw: {flux.dtype}")
    ok(f"dtype(signal/flux_raw)={flux.dtype}")


def check_representation_consistency(root):
    has_norm_flux = "representations/normalized_flux" in root
    has_norm_unc = "representations/normalized_uncertainty" in root
    if has_norm_flux != has_norm_unc:
        fail(
            "representations/normalized_flux and representations/normalized_uncertainty "
            "must be both present or both absent"
        )
    if not has_norm_flux:
        ok("normalized_flux absent (allowed)")
        return

    norm_flux = np.asarray(root["representations/normalized_flux"][:])
    norm_unc = np.asarray(root["representations/normalized_uncertainty"][:])

    has_flux = np.isfinite(norm_flux).any(axis=1)
    has_unc = np.isfinite(norm_unc).any(axis=1)
    missing_unc = int(np.sum(has_flux & ~has_unc))
    if missing_unc > 0:
        warn(f"{missing_unc} rows have finite normalized_flux but all-NaN normalized_uncertainty")

    ok("Representation consistency checked")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path")
    parser.add_argument("--expected-nspec", type=int)
    parser.add_argument("--samples", type=int, default=64)

    args = parser.parse_args()

    path = Path(args.zarr_path)
    if not path.exists():
        fail(f"Path does not exist: {path}")

    print("\n🔎 Opening Zarr store...")
    root = zarr.open(str(path), mode="r")

    check_required_groups(root)
    check_required_arrays(root)
    check_units(root)

    nspec, _ = check_shapes(root, args.expected_nspec)

    check_dtype(root)
    check_params_alignment(root, nspec)
    check_metadata_alignment(root, nspec)
    check_representation_consistency(root)

    sample_signal(root, nspec, args.samples)
    check_chunks(root)
    check_provenance(root)
    check_index(path, nspec)

    print("\n🎉 ZARR VALIDATION PASSED\n")


if __name__ == "__main__":
    main()
