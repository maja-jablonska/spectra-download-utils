# ZARR WRITE SPEC — Pipeline Author Guide

Version: 2.0
Applies to: ALL spectral writers
Compliance: REQUIRED

This guide defines the required write process for the grouped spectral schema.

---

# Non-Negotiable Rules

## 1. Never write directly to final path

Always materialize into:

```
<dataset>.zarr.tmp
```

Publish only after full validation via atomic rename:

```
os.replace(tmp, final)
```

## 2. Two-phase write is mandatory

Phase A (materialization):

- create arrays/groups
- write all data
- write `index.parquet`

Phase B (validation + commit):

- run shape/alignment/completeness checks
- if pass: atomic rename
- if fail: delete `.tmp` and exit non-zero

## 3. No dynamic growth

Forbidden:

- `append()`
- repeated `resize()` as ingest strategy

Required:

- compute `N_spec` before writing
- preallocate arrays with final shape
- write by slices/batches

---

# Required Schema to Write

```
signal/wavelength
signal/flux_raw
signal/uncertainty_raw
continuum/continuum_model
continuum/continuum_uncertainty
continuum/method
params/*
metadata/*
index.parquet
```

`representations/normalized_flux` is optional and should be written only when
available in source data; `representations/normalized_uncertainty` should be
written alongside it (and omitted when normalized flux is omitted).
`params/` may be an empty group when no derived parameters exist.

For FITS-backed ingest, also write projected header fields under `metadata/`
using `h{hdu_index}_{sanitized_keyword}` names.

---

# Compression and Chunking

Recommended:

- Blosc zstd bitshuffle (`clevel 5-7`)
- chunks `(1024, N_pix)` for typical spectra
- chunks `(256, 4096)` for high-resolution spectra

Avoid tiny chunks and single-row write loops.

---

# Write Pattern (Required)

Preferred approach:

```python
# Preallocate
flux = signal.create_array("flux_raw", shape=(N_spec, N_pix), ...)

# Batched writes
flux[start:end, :] = flux_batch
```

Do not write spectrum-by-spectrum unless unavoidable.

---

# Continuum + Representation Rules

- `continuum/method` MUST be present for every spectrum (`shape (N_spec,)`)
- `continuum_model` and `continuum_uncertainty` may be all-NaN when unavailable
- `normalized_flux` should not be synthesized; write only if source provides it
- `normalized_uncertainty` should be written only when `normalized_flux` is
  written, with matching shape `(N_spec, N_pix)`
- if normalized representations are absent, omit both arrays

---

# FITS Header Projection (Recommended for Raw Ingest)

- Project only reusable, general observing FITS header fields into `metadata/`
  (airmass, exposure/time, coordinates, target/instrument, and basic conditions).
- Key naming: `h{hdu_index}_{sanitized_keyword}` where:
  - keyword is lowercased
  - non-alphanumeric characters are replaced with `_`
  - repeated `_` collapse to one `_`
  - duplicate cards in the same HDU receive suffixes `_2`, `_3`, ...
- Store projected values as UTF-8 strings.
- Keep existing canonical metadata fields (`spec_id`, `date_obs`, etc.) in
  addition to projected header fields.
- Exclude high-volume, low-reuse families (for example HARPS calibration
  coefficient blocks and low-level telemetry streams).

---

# Required Validation Before Publish

Validate all of the following against the `.tmp` store:

1. Required groups exist: `signal`, `continuum`, `representations`, `params`, `metadata`
2. Required arrays exist and align
3. `signal/flux_raw` is 2D and non-empty
4. `signal/wavelength` is 1D `(N_pix,)` or 2D `(N_spec, N_pix)`
5. All row-wise arrays under `params` and `metadata` have shape `(N_spec,)`
6. No sampled all-NaN or all-zero rows in `signal/flux_raw`
7. Units attrs present on numeric spectral arrays
8. Provenance attrs present (`schema_version`, `ingest_pipeline`, `created_at`, `git_commit`)
9. `index.parquet` exists and `len(index) == N_spec`

If any check fails:

- delete `.tmp`
- exit with non-zero status
- emit actionable error logs

---

# Provenance Requirements

Set at root:

```python
root.attrs["schema_version"] = "2.0"
root.attrs["ingest_pipeline"] = "..."
root.attrs["created_at"] = "UTC ISO timestamp"
root.attrs["git_commit"] = "short hash"
```

---

# Operational Guidance

- One writer process per target store
- For parallel ingest, write immutable shards then merge
- Prefer local scratch storage for temporary materialization
- Keep schema changes additive unless doing a major version bump

---

# Final Principle

> Make writes boring and recoverable.

Predictable, validated, atomic pipelines outlive individual instruments and
projects.
