# Spectra Download Utilities

This repository provides a unified bulk download flow for spectra while keeping
separate, source-specific request logic. Each source implements its own URL and
response parsing, and the `bulk_download` orchestrator handles retries, logging,
error handling, and aggregation.

## Features

- **Unified bulk flow**: One consistent pipeline for fetching spectra from
  multiple sources.
- **Source-specific logic**: Individual request and parse logic per data source.
- **Comprehensive logging**: Structured logging with context for source, IDs,
  URLs, and result counts.
- **Extensible**: Add new sources by implementing `SpectraSource`.

## Quick Start

```python
import logging

from spectra_download import (
    GnpSource,
    MassBankSource,
    NistSource,
    SpectraRequest,
    bulk_download,
)

logging.basicConfig(level=logging.INFO)

sources = {
    "nist": NistSource(timeout=20, max_retries=3),
    "massbank": MassBankSource(timeout=20, max_retries=3),
    "gnps": GnpSource(timeout=20, max_retries=3),
}

requests = [
    SpectraRequest(source="nist", identifier="12345"),
    SpectraRequest(source="massbank", identifier="MB000123"),
    SpectraRequest(source="gnps", identifier="CCMSLIB00000001538"),
]

results = bulk_download(requests, sources)

for result in results:
    if result.success:
        print(result.request.identifier, len(result.spectra))
    else:
        print(result.request.identifier, result.error)
```

## Logging

The utilities emit logs at the following points:

- **INFO** when a source batch begins and completes.
- **INFO** when a download starts and finishes per identifier.
- **WARNING** for empty responses or retryable failures.
- **ERROR** for unknown sources.
- **EXCEPTION** when a download fails and the bulk flow continues.

You can customize logging with `logging.basicConfig` or a structured logging
setup to route logs to JSON or centralized logging systems.

## Adding a New Source

1. Create a new module in `spectra_download/sources/`.
2. Subclass `SpectraSource` and implement `build_request_url` and
   `parse_response`.
3. Register the new source in your `sources` mapping passed to
   `bulk_download`.

## Customizing File Processing

`SpectraSource` includes a single, explicit hook for how files are fetched and
prepared before persistence. Override this when a source needs custom handling
(for example, alternate authentication, decompression, or non-standard
DataLink resolution):

- `SpectraSource.fetch_fits_payload(access_url, spectrum)` returns the raw FITS
  bytes plus any metadata updates to apply (for example, a resolved `access_url`
  after DataLink hops).

For structured outputs, continue to override:

- `spectrum_to_zarr_components(...)` for spectra
- `ccf_to_zarr_components(...)` for CCF products

## Repository Layout

- `spectra_download/models.py`: Shared data models for requests and results.
- `spectra_download/http_client.py`: Common HTTP helper with retries.
- `spectra_download/sources/`: Source-specific downloaders.
- `spectra_download/bulk.py`: Unified bulk download flow.

## Raw FITS to Zarr Converters

Contract-compliant local converters are available for notebook-style raw FITS
directories:

```bash
python scripts/espadons_to_zarr.py data/CFHT /tmp/espadons_data.zarr --overwrite
python scripts/harps_to_zarr.py notebook_test/spectra/eso/harps /tmp/harps_data.zarr --overwrite
python scripts/feros_to_zarr.py data/eso/feros /tmp/feros_data.zarr --overwrite
python scripts/nirps_to_zarr.py data/eso/nirps /tmp/nirps_data.zarr --overwrite
python scripts/uves_to_zarr.py data/eso/uves /tmp/uves_data.zarr --overwrite
```

These scripts write to `<output>.zarr.tmp`, run completeness checks, then
atomically rename to `<output>.zarr`.

Both scripts write top-level `continuum` and `unnormalized_flux`
(`N_spec, N_pix`) with `NaN` fallback when unavailable.
They also write companion arrays for when these products carry their own grids
or errors:
- `continuum_wavelength`, `continuum_uncertainty`
- `unnormalized_wavelength`, `unnormalized_uncertainty`

Pairing behavior:
- Preferred pair: `flux + continuum`
- Alternative pair: `flux + unnormalized_flux`

ESPaDOnS-specific handling:
- If `COL4-6` (UnNormalized) exist, `flux` is written from those columns.
- If both normalized and unnormalized intensity exist, `continuum` is estimated
  as `unnormalized / normalized`.
- `metadata/gaia_id` and `metadata/gaia_dr` are written from `GAIAID` and
  `GAIADR` headers when present.

## Notes

These examples assume the source APIs are reachable and return JSON. If a source
API changes, update the corresponding `parse_response` implementation.
