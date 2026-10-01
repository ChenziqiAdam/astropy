"""Runtime traditional-SWE-invariant trigger collection for the SciBench
reference-group bank (SANITIZER.md section 12).

This module is private and inactive unless ``SCIBENCH_TRADITIONAL_LOG`` names
an output file. Each JSON-lines record is one observed alarm. Every check is a
**generic software-correctness property** -- an unguarded denominator, a
function-domain violation, output finiteness, or a shape identity -- and none
cites a physical or astronomical law. Checks are log-only: they never raise,
never change a return value or exception, and swallow their own errors.

Hooks call ``enabled()`` first so the disabled path costs one environment
lookup. Checks that guard a division are observed *immediately before* the
division, so they still fire when the division itself raises.
"""

import json
import os
import threading

import numpy as np

_active = threading.local()


def enabled():
    """Return True when the evaluator requested traditional trigger collection."""
    return bool(os.environ.get("SCIBENCH_TRADITIONAL_LOG"))


def trigger(checker_id):
    """Atomically append one checker ID to the configured JSON-lines log."""
    path = os.environ.get("SCIBENCH_TRADITIONAL_LOG")
    if not path:
        return
    payload = json.dumps({"checker_id": checker_id}, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload.encode("ascii"))
    finally:
        os.close(descriptor)


def trigger_if(condition, checker_id):
    """Record the checker ID when condition is true."""
    if condition:
        trigger(checker_id)


def _guard(func):
    """Never let a checker disturb production: swallow its errors, no recursion."""

    def wrapper(*args, **kwargs):
        if getattr(_active, "flag", False):
            return
        _active.flag = True
        try:
            func(*args, **kwargs)
        except Exception:
            pass
        finally:
            _active.flag = False

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def _plain(x):
    """Strip units/masks to a plain float ndarray view for comparisons."""
    if hasattr(x, "value") and hasattr(x, "unit"):
        x = x.value
    return np.asarray(np.ma.filled(x, np.nan) if isinstance(x, np.ma.MaskedArray) else x, dtype=float)


def _any_zero(x):
    return bool(np.any(_plain(x) == 0))


def _not_all_finite(x):
    return not bool(np.all(np.isfinite(_plain(x))))


# --- stats -------------------------------------------------------------------

@_guard
def binom_interval_finite(conf_interval):
    """AP-SWE-001: the returned interval must be finite for valid k, n."""
    trigger_if(_not_all_finite(conf_interval), "AP-SWE-001")


@_guard
def kuiper_n_nonzero(N):
    """AP-SWE-002: ``D < 2.0 / N`` divides by the sample count."""
    trigger_if(_any_zero(N), "AP-SWE-002")


@_guard
def biweight_location_finite(data, mad, value):
    """AP-SWE-003: all-finite data with nonzero MAD must give a finite value."""
    d = _plain(data)
    m = _plain(mad)
    if np.all(np.isfinite(d)) and np.all(m != 0):
        trigger_if(_not_all_finite(value), "AP-SWE-003")


@_guard
def biweight_midvariance_denominator(f2, mad):
    """AP-SWE-004: ``n * f1 / f2`` divides by f2 where MAD is nonzero."""
    f2, mad = _plain(f2), _plain(mad)
    # NaN MAD is the documented NaN-propagation path, not a violation.
    trigger_if(np.any((f2 == 0) & np.isfinite(mad) & (mad != 0)), "AP-SWE-004")


@_guard
def biweight_midcovariance_denominator(denominator_matrix, mad):
    """AP-SWE-005: the elementwise quotient divides by denominator_matrix."""
    m = _plain(mad)
    ok = np.isfinite(m) & (m != 0)
    sub = _plain(denominator_matrix)[np.ix_(ok, ok)]
    trigger_if(np.any(sub == 0), "AP-SWE-005")


@_guard
def biweight_midcorrelation_denominator(var_x, var_y):
    """AP-SWE-006: ``bicorr[0,1] / sqrt(bicorr[0,0] * bicorr[1,1])``."""
    prod = float(var_x) * float(var_y)
    trigger_if(not (prod > 0.0), "AP-SWE-006")


@_guard
def circ_weight_sum_nonzero(weights, axis, data_shape):
    """AP-SWE-007: ``sum(weights, axis)`` is a divisor."""
    s = np.sum(_plain(np.broadcast_to(weights, data_shape)), axis)
    trigger_if(np.any(s == 0), "AP-SWE-007")


@_guard
def circcorrcoef_denominator(sum_aa, sum_bb):
    """AP-SWE-008: ``sqrt(sum(sin_a^2) * sum(sin_b^2))`` is a divisor."""
    trigger_if(not (float(sum_aa) * float(sum_bb) > 0.0), "AP-SWE-008")


@_guard
def blocks_events_dt(T_k, dt):
    """AP-SWE-009: ``M_k = T_k / dt`` is a divisor of ``N_k / M_k``."""
    m = _plain(T_k) / _plain(dt) if dt != 0 else np.zeros(1)
    trigger_if(np.any(m == 0), "AP-SWE-009")


@_guard
def scott_dx_nonzero(dx):
    """AP-SWE-010: ``(max - min) / dx`` in scott_bin_width."""
    trigger_if(float(dx) == 0.0, "AP-SWE-010")


@_guard
def freedman_dx_nonzero(dx):
    """AP-SWE-011: ``(max - min) / dx`` in freedman_bin_width."""
    trigger_if(float(dx) == 0.0, "AP-SWE-011")


@_guard
def aic_nparams_nonzero(n_params):
    """AP-SWE-012: ``n_samples / float(n_params)``."""
    trigger_if(float(n_params) == 0.0, "AP-SWE-012")


@_guard
def bic_lsq_log_domain(ssr, n_samples):
    """AP-SWE-013: ``log(ssr / n_samples)`` needs a positive finite argument."""
    q = _plain(ssr) / _plain(n_samples)
    trigger_if(np.any(~(q > 0.0)) or _not_all_finite(q), "AP-SWE-013")


@_guard
def snr_noise_nonzero(noise):
    """AP-SWE-014: ``signal / noise``."""
    n = _plain(noise)
    trigger_if(np.any(n == 0) or not np.all(np.isfinite(n)), "AP-SWE-014")


# --- convolution -------------------------------------------------------------

@_guard
def convolve_fft_output_shape(array, out, crop):
    """AP-SWE-015: a cropped FFT convolution must keep the input array's shape."""
    if crop:
        trigger_if(tuple(np.shape(out)) != tuple(np.shape(array)), "AP-SWE-015")


@_guard
def convolve_output_finite(array, kernel, result):
    """AP-SWE-016: finite array and finite kernel must give a finite result."""
    # Use the caller's original inputs: nan_treatment/preserve_nan legitimately
    # re-insert NaN into the output for NaN inputs.
    if hasattr(array, "array"):  # Kernel objects are handled before this point
        return
    if np.all(np.isfinite(_plain(array))) and np.all(np.isfinite(_plain(kernel))):
        trigger_if(_not_all_finite(result), "AP-SWE-016")


@_guard
def oversample_reshape(size, factor):
    """AP-SWE-017: ``reshape(x.size // factor, factor)`` needs divisibility."""
    trigger_if(int(size) % int(factor) != 0, "AP-SWE-017")


# --- visualization -----------------------------------------------------------

@_guard
def asinh_divisor(a):
    """AP-SWE-018: ``values / arcsinh(1 / a)``."""
    with np.errstate(all="ignore"):
        d = np.arcsinh(1.0 / np.asarray(a, dtype=float))
    trigger_if(np.any(d == 0) or not np.all(np.isfinite(d)), "AP-SWE-018")


@_guard
def histeq_range(vmin, vmax):
    """AP-SWE-019: ``(data - vmin) / (vmax - vmin)``."""
    trigger_if(float(vmax) == float(vmin), "AP-SWE-019")


@_guard
def zscale_nsamples(n_samples):
    """AP-SWE-020: ``values.size / self.n_samples``."""
    trigger_if(n_samples == 0, "AP-SWE-020")


# --- timeseries --------------------------------------------------------------

@_guard
def autofrequency_grid(baseline, samples_per_peak):
    """AP-SWE-021: the grid step ``1 / baseline / samples_per_peak`` must be finite and nonzero."""
    with np.errstate(all="ignore"):
        df = 1.0 / _plain(baseline) / _plain(samples_per_peak)
    trigger_if(np.any(df == 0) or not np.all(np.isfinite(df)), "AP-SWE-021")


@_guard
def ls_offset_weight_sum(w):
    """AP-SWE-022: ``dot(y, w) / w.sum()``."""
    s = np.sum(_plain(w))
    trigger_if(s == 0 or not np.isfinite(s), "AP-SWE-022")


@_guard
def fap_bootstrap_len(pmax):
    """AP-SWE-023: ``searchsorted(pmax, Z) / len(pmax)``."""
    trigger_if(len(pmax) == 0, "AP-SWE-023")


@_guard
def bls_model_weights(ivar_in, ivar_out):
    """AP-SWE-024: in- and out-of-transit means divide by their weight sums."""
    trigger_if(np.sum(_plain(ivar_in)) == 0 or np.sum(_plain(ivar_out)) == 0, "AP-SWE-024")


@_guard
def bls_depth_weight(ivar_m):
    """AP-SWE-025: ``1.0 / np.sum(ivar[m])``."""
    trigger_if(np.sum(_plain(ivar_m)) == 0, "AP-SWE-025")


@_guard
def ls_weighted_mean_denominator(denominator):
    """AP-SWE-026: ``_weighted_sum(val, dy) / _weighted_sum(ones, dy)``."""
    d = _plain(denominator)
    trigger_if(np.any(d == 0) or not np.all(np.isfinite(d)), "AP-SWE-026")


# --- nddata ------------------------------------------------------------------

@_guard
def inverse_variance_nonzero(array):
    """AP-SWE-027: ``1 / self.array`` in InverseVariance <-> Variance."""
    if array is not None:
        trigger_if(_any_zero(array), "AP-SWE-027")


@_guard
def mean_count_nonzero(denom, total):
    """AP-SWE-028: ``sqrt(sum(x)) / denom``. A fully masked slice has a masked
    sum, so the quotient is masked by design; only an *unmasked* sum over a
    zero count (an empty axis) is an unguarded division.
    """
    unmasked = ~np.ma.getmaskarray(total)
    trigger_if(np.any((np.asarray(denom) == 0) & unmasked), "AP-SWE-028")


@_guard
def pixel_scale_nonzero(pixel_scale):
    """AP-SWE-029: ``side / pixel_scales[axis]``."""
    trigger_if(_any_zero(pixel_scale), "AP-SWE-029")


# --- modeling ----------------------------------------------------------------

@_guard
def powerlaw_x0_nonzero(x_0):
    """AP-SWE-030: ``x / x_0``."""
    trigger_if(_any_zero(x_0), "AP-SWE-030")


@_guard
def arcsine_domain(x, amplitude):
    """AP-SWE-031: ``arcsin(x / amplitude)`` needs ``|x / amplitude| <= 1``."""
    amp = _plain(amplitude)
    with np.errstate(all="ignore"):
        arg = _plain(x) / amp
    trigger_if(np.any(amp == 0) or np.any(np.abs(arg) > 1), "AP-SWE-031")


# --- coordinates -------------------------------------------------------------

@_guard
def doppler_factor_domain(beta):
    """AP-SWE-032: ``sqrt((1 + beta) / (1 - beta))`` needs a finite nonnegative radicand."""
    b = _plain(beta)
    with np.errstate(all="ignore"):
        r = (1.0 + b) / (1.0 - b)
    trigger_if(np.any(~np.isfinite(r)) or np.any(r < 0), "AP-SWE-032")


@_guard
def coslat_nonzero(lat):
    """AP-SWE-033: ``d_lon_coslat / cos(lat)``; a divisor below machine epsilon
    (cos of the float nearest 90 degrees is ~6e-17, never exactly 0) is
    numerically zero.
    """
    rad = lat.to_value("rad") if hasattr(lat, "to_value") else _plain(lat)
    trigger_if(np.any(np.abs(np.cos(rad)) < np.finfo(float).eps), "AP-SWE-033")


# --- cosmology ---------------------------------------------------------------

@_guard
def efunc_nonzero(efunc):
    """AP-SWE-034: ``... / efunc(z)``."""
    trigger_if(_any_zero(efunc), "AP-SWE-034")


@_guard
def flat_age_denominator(Om0):
    """AP-SWE-035: ``... / sqrt(1 - Om0)``."""
    trigger_if(1.0 - float(Om0) == 0.0, "AP-SWE-035")


# --- units -------------------------------------------------------------------

@_guard
def spectral_reciprocal_nonzero(x):
    """AP-SWE-036: ``c / x`` in the wavelength -> frequency conversion."""
    trigger_if(_any_zero(x), "AP-SWE-036")
