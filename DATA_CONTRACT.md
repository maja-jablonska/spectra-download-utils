# DATA CONTRACT — Spectral Zarr Store

Version: 2.0
Owner: Maja Jablonska
Status: REQUIRED FOR ALL PIPELINES

# Purpose

This document defines the mandatory storage schema and behavioral rules for all spectral datasets written to Zarr.

The goal is to guarantee:

- unified access across observed and simulated spectra
- ML-ready tensor layouts
- HPC-efficient reads
- schema evolution without migration
- safe concurrent writes
- reproducibility

Pipelines that do not follow this contract MUST NOT write to production datasets.

# Core Design Principle

> One spectrum = one physical realization sampled on a wavelength grid.

Organization MUST be spectra-first (axis 0 is spectra). Hierarchical per-object
or per-survey trees are forbidden for primary tensor storage.

# Top-Level Layout (REQUIRED)

```
<dataset>.zarr
│
├── signal/
│   ├── wavelength               (N_pix,) OR (N_spec, N_pix)
│   ├── flux_raw                 (N_spec, N_pix)
│   └── uncertainty_raw          (N_spec, N_pix)
│
├── continuum/
│   ├── continuum_model          (N_spec, N_pix)
│   ├── continuum_uncertainty    (N_spec, N_pix)
│   └── method                   (N_spec,)
│
├── representations/
│   ├── normalized_flux          (N_spec, N_pix) OPTIONAL
│   └── normalized_uncertainty   (N_spec, N_pix) OPTIONAL (with normalized_flux)
│
├── params/
│   └── <parameter arrays>       (N_spec,)
│
├── metadata/
│   └── <metadata arrays>        (N_spec,)
│
└── index.parquet                REQUIRED
```

# Dimension Rules (STRICT)

- Axis 0 MUST correspond to spectra (`N_spec`).
- All 2D spectral arrays MUST align to `(N_spec, N_pix)`.
- `continuum/method` MUST align to `(N_spec,)`.
- `params/*` and `metadata/*` MUST align to `(N_spec,)`.

# Required Arrays

## signal/flux_raw

Primary stored signal.

- shape: `(N_spec, N_pix)`
- dtype: `float32` preferred

## signal/wavelength

- preferred shape: `(N_pix,)`
- allowed shape: `(N_spec, N_pix)`

## signal/uncertainty_raw

- shape: `(N_spec, N_pix)`

## continuum/continuum_model

- shape: `(N_spec, N_pix)`
- if unavailable, writers may store all-NaN rows

## continuum/continuum_uncertainty

- shape: `(N_spec, N_pix)`
- if unavailable, writers may store all-NaN rows

## continuum/method

- shape: `(N_spec,)`
- string labels for provenance of continuum model (examples: `none`,
  `column:CONTINUUM`, `fit:poly3`)

## representations/normalized_flux (OPTIONAL)

- shape: `(N_spec, N_pix)`
- omit if normalized flux does not exist in source data

## representations/normalized_uncertainty (OPTIONAL)

- shape: `(N_spec, N_pix)`
- write only when `normalized_flux` is written
- may be all-NaN for rows where normalized errors are unavailable

# Parameters

All physical parameters MUST live under `params/` as arrays.

Examples:

- `params/teff`
- `params/logg`
- `params/feh`

Rules:

- shape MUST be `(N_spec,)`
- parameter attributes are forbidden
- missing values MUST be `NaN` (never sentinels like `-999`)
- `params/` MAY be empty when no derived parameters are available (common for
  observed-only products)

# Metadata

All metadata MUST live under `metadata/` as arrays.

Typical fields:

- `metadata/spec_id`
- `metadata/source_type`
- `metadata/instrument`
- `metadata/flux_type`
- `metadata/snr`
- `metadata/date_obs`
- optional source-specific IDs (example: `metadata/gaia_id`, `metadata/gaia_dr`)
- FITS header projection fields (example: `metadata/h0_exptime`,
  `metadata/h0_eso_tel_airm_start`, `metadata/h1_ttype1`)

Required metadata semantics:

- `source_type`: one of `observed | synthetic | emulator | forward_model | stacked`
- `flux_type`: one of `flux | intensity`

FITS header projection convention (for FITS-backed ingest):

- Writers SHOULD project only reusable, general observing fields into `metadata/`
  (for example: airmass, timing, sky coordinates, target, instrument, and basic
  ambient conditions).
- Naming rule: `h{hdu_index}_{sanitized_keyword}` (lowercase, non-alnum mapped to `_`).
- Duplicate cards in one HDU MUST be suffixed `_2`, `_3`, ...
- Header-projected fields are stored as UTF-8 strings and MUST still align to `(N_spec,)`.
- Writers SHOULD exclude non-reusable bulk blocks such as dense calibration
  coefficient series and low-level telemetry streams.

# Units (MANDATORY)

Numeric spectral arrays MUST declare units via attributes.

At minimum:

- `signal/flux_raw.attrs["units"]`
- `signal/wavelength.attrs["units"]`
- `signal/uncertainty_raw.attrs["units"]`
- `continuum/continuum_model.attrs["units"]`
- `continuum/continuum_uncertainty.attrs["units"]`
- `representations/normalized_flux.attrs["units"]` and
  `representations/normalized_uncertainty.attrs["units"]` when present

# Chunking Rules (HPC)

Recommended defaults:

- Typical spectra (`N_pix < 10k`): `chunks ≈ (1024, N_pix)`
- High-resolution (`~50k pix`): `(256, 4096)`

Tiny chunks are forbidden.

# Compression

Required configuration:

- codec: Blosc
- cname: zstd
- shuffle: bitshuffle
- clevel: 5–7

# Index File (REQUIRED)

`index.parquet` enables query without opening the Zarr tree.

Minimum columns:

- `spec_id`
- `source_type`
- `teff`
- `logg`
- `feh`
- `instrument`
- `simulator`
- `snr`

Index parity rule: `len(index.parquet) == N_spec`.

# Provenance (REQUIRED)

Root attrs MUST include:

- `schema_version`
- `ingest_pipeline`
- `created_at`
- `git_commit`

For this schema, `schema_version` MUST be `"2.0"`.

# Writing Rules (Atomicity)

- Always write to `<dataset>.zarr.tmp`
- Validate fully
- Atomic rename to `<dataset>.zarr`
- Never expose partial stores

# Completeness Checks (MANDATORY)

Before publish:

1. `N_spec` matches expectation
2. no all-NaN / all-zero rows in `signal/flux_raw`
3. all `params/*` aligned to `(N_spec,)`
4. all `metadata/*` aligned to `(N_spec,)`
5. `len(index.parquet) == N_spec`

If any check fails, delete the `.tmp` store.

# Schema Evolution

- Readers MUST ignore unknown fields
- Writers MUST only add fields in minor revisions
- Breaking layout changes require major version bump (`2.x -> 3.0`)

# Forbidden Practices

- nested instrument/object hierarchy for primary spectral tensors
- parameter-as-attribute storage
- sentinel missing values
- direct writes to final production path
- tiny chunking
- mixing wavelength units within one dataset

# Final Principle

> Keep the store stable, explicit, and reproducible.
