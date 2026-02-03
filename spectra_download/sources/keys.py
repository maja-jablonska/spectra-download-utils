from enum import Enum

class SpectrumKeys(Enum):
    intensity = "intensity"
    wavelengths = "wavelengths"
    error = "error"
    

class DataKeys(Enum):
    exptime = "exptime"
    ra = "ra"
    dec = "dec"
    date = "date"
    reference_frame = "reference_frame"
    reference_frame_epoch = "reference_frame_epoch"
    mjd = "mjd"
    airmass = "airmass"
    object = "object"
    berv = "berv"
    frame = "frame"
    normalized = "normalized"
    snr = 'snr'


class ObservedFrame(Enum):
    barycentric = "barycentric"
    observer = "observer"
