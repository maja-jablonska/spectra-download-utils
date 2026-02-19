"""Downloader for ESPaDOnS spectra (PolarBase) via TAP."""

from __future__ import annotations

import logging
from typing import Any, Dict

from spectra_download.sources.polarbase import PolarBaseSource

logger = logging.getLogger(__name__)


class EspadonsSource(PolarBaseSource):
    """ESPaDOnS spectra downloads via CDS TAPVizieR (PolarBase)."""

    name = "espadons"
    _inst_value = "espadons"

    def _apply_coordinate_corrections(
        self,
        *,
        ra_deg: float,
        dec_deg: float,
        extra_params: Dict[str, Any],
        identifier: str,
    ) -> tuple[float, float]:
        """Apply proper motion correction when parameters are provided."""

        pm_ra = (
            extra_params.get("pm_ra_cosdec_mas_yr")
            if extra_params.get("pm_ra_cosdec_mas_yr") is not None
            else extra_params.get("pm_ra_mas_yr")
        )
        pm_dec = extra_params.get("pm_dec_mas_yr")
        if pm_ra is None or pm_dec is None:
            return ra_deg, dec_deg

        ref_epoch = float(extra_params.get("ref_epoch_year", 2000.0))
        obs_epoch = float(extra_params.get("obs_epoch_year", ref_epoch))
        if obs_epoch == ref_epoch:
            return ra_deg, dec_deg

        try:
            from astropy.coordinates import SkyCoord  # type: ignore
            import astropy.units as u  # type: ignore
            from astropy.time import Time  # type: ignore
        except Exception:
            logger.warning(
                "Proper motion correction requires astropy",
                extra={"source": self.name, "identifier": identifier},
            )
            return ra_deg, dec_deg

        try:
            c = SkyCoord(
                ra=ra_deg * u.deg,
                dec=dec_deg * u.deg,
                pm_ra_cosdec=float(pm_ra) * u.mas / u.yr,
                pm_dec=float(pm_dec) * u.mas / u.yr,
                obstime=Time(ref_epoch, format="jyear"),
                frame="icrs",
            )
            c2 = c.apply_space_motion(new_obstime=Time(obs_epoch, format="jyear"))
            logger.info(
                "ESPaDOnS proper motion applied",
                extra={
                    "source": self.name,
                    "identifier": identifier,
                    "ref_epoch_year": ref_epoch,
                    "obs_epoch_year": obs_epoch,
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "ra_corr_deg": float(c2.ra.deg),
                    "dec_corr_deg": float(c2.dec.deg),
                },
            )
            return float(c2.ra.deg), float(c2.dec.deg)
        except Exception:
            logger.warning(
                "Proper motion correction failed",
                extra={"source": self.name, "identifier": identifier, "ref_epoch_year": ref_epoch, "obs_epoch_year": obs_epoch},
            )
            return ra_deg, dec_deg
