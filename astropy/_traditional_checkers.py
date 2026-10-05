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

import math

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


# Inputs this large overflow float64 in squares/products long before any real
# data does (the traditional bank's finiteness/domain checks are not about
# float64 range limits). Every checker is skipped when any argument holds a
# finite number above this; NaN/inf are ignored because many checkers receive
# a (possibly non-finite) *result* as an argument. Checkers whose own
# computation overflows earlier (e.g. QCP, degree 8) apply a stricter bound.
_GATE_LIMIT = 1e100
GATE_STATS = {}  # checker name -> number of calls skipped by the magnitude gate


def _has_extreme_magnitude(values, limit=_GATE_LIMIT, _depth=0):
    """True if any finite number reachable from ``values`` exceeds ``limit``."""
    try:
        for v in values:
            if v is None or isinstance(v, (bool, str, bytes)):
                continue
            if hasattr(v, "unit") and hasattr(v, "value"):  # astropy Quantity
                v = v.value
            if isinstance(v, (int, float, np.integer, np.floating)):
                if math.isfinite(v) and abs(v) > limit:
                    return True
            elif isinstance(v, np.ndarray):
                if v.dtype.kind in "fiu" and v.size:
                    a = np.ma.filled(v, 0) if isinstance(v, np.ma.MaskedArray) else v
                    a = np.abs(a[np.isfinite(a)]) if a.dtype.kind == "f" else np.abs(a)
                    if a.size and float(a.max()) > limit:
                        return True
            elif isinstance(v, dict) and _depth < 2:
                if _has_extreme_magnitude(v.values(), limit, _depth + 1):
                    return True
            elif isinstance(v, (list, tuple)) and _depth < 2 and len(v) <= 100000:
                if _has_extreme_magnitude(v, limit, _depth + 1):
                    return True
    except Exception:
        return False
    return False


def _guard(func):
    """Never let a checker disturb production: swallow its errors, no recursion."""

    def wrapper(*args, **kwargs):
        if getattr(_active, "flag", False):
            return
        _active.flag = True
        try:
            if _has_extreme_magnitude(args) or _has_extreme_magnitude(kwargs.values()):
                GATE_STATS[func.__name__] = GATE_STATS.get(func.__name__, 0) + 1
                return
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
    """AP-SWE-003: all-finite data of ordinary magnitude with nonzero MAD must
    give a finite value. Magnitudes above 1e150 are excluded: ``c * MAD`` and
    the weighted sums overflow float64 near its maximum by ordinary arithmetic.
    """
    d = _plain(data)
    m = _plain(mad)
    if np.all(np.isfinite(d)) and np.all(np.abs(d) < 1e150) and np.all(m != 0):
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
def biweight_midcorrelation_denominator(var_x, var_y, x, y):
    """AP-SWE-006: ``bicorr[0,1] / sqrt(bicorr[0,0] * bicorr[1,1])``. A constant
    variable has an undefined correlation (NaN by convention) and is excluded;
    for finite non-constant inputs the radicand must be strictly positive.
    """
    x, y = _plain(x), _plain(y)
    if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.ptp(x) > 0 and np.ptp(y) > 0:
        prod = float(var_x) * float(var_y)
        trigger_if(not (prod > 0.0), "AP-SWE-006")


@_guard
def circ_weight_sum_nonzero(weights, axis, data_shape):
    """AP-SWE-007: ``sum(weights, axis)`` is a divisor. An all-zero weight slice
    means "no data" and is excluded; a zero sum with nonzero weights
    (cancelling signed weights) is an unguarded divisor.
    """
    w = _plain(np.broadcast_to(weights, data_shape))
    s = np.sum(w, axis)
    has_weight = np.sum(np.abs(w), axis) > 0
    trigger_if(np.any((s == 0) & has_weight), "AP-SWE-007")


@_guard
def circcorrcoef_denominator(sum_aa, sum_bb, alpha, beta):
    """AP-SWE-008: ``sqrt(sum(sin_a^2) * sum(sin_b^2))`` is a divisor. A constant
    angle sample has an undefined correlation (NaN by convention) and is
    excluded; for finite non-constant samples the product must be positive.
    Spreads so small that the product of the two sums of squares underflows
    float64 (about ptp_a * ptp_b < 1e-145) are a floating-point limit, not a
    missing guard, and are excluded.
    """
    a, b = _plain(alpha), _plain(beta)
    if (
        np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
        and np.ptp(a) > 0
        and np.ptp(b) > 0
        and float(np.ptp(a)) * float(np.ptp(b)) > 1e-100
    ):
        trigger_if(not (float(sum_aa) * float(sum_bb) > 0.0), "AP-SWE-008")


@_guard
def blocks_events_dt(T_k, dt):
    """AP-SWE-009: ``M_k = T_k / dt`` is a divisor of ``N_k / M_k``."""
    m = _plain(T_k) / _plain(dt) if dt != 0 else np.zeros(1)
    trigger_if(np.any(m == 0), "AP-SWE-009")


@_guard
def scott_dx_nonzero(dx):
    """AP-SWE-010: ``(max - min) / dx`` in scott_bin_width (only evaluated when
    return_bins is requested; the caller observes it inside that branch).
    """
    trigger_if(float(dx) == 0.0, "AP-SWE-010")


@_guard
def freedman_dx_nonzero(dx, data_range):
    """AP-SWE-011: ``(max - min) / dx`` in freedman_bin_width. A zero IQR on
    non-constant data already raises a documented ValueError (astropy #7125)
    and is excluded; constant data (range 0, dx 0) is divided as 0/0 and
    silently produces zero-width bins.
    """
    trigger_if(float(dx) == 0.0 and float(data_range) == 0.0, "AP-SWE-011")


@_guard
def aic_nparams_nonzero(n_params):
    """AP-SWE-012: ``n_samples / float(n_params)``."""
    trigger_if(float(n_params) == 0.0, "AP-SWE-012")


@_guard
def bic_lsq_log_domain(ssr, n_samples):
    """AP-SWE-013: ``log(ssr / n_samples)`` needs a nonnegative finite argument.
    ssr == 0 (a perfect fit, log 0 = -inf) is a defined limit and excluded,
    as is a non-finite ssr (residuals large enough to overflow when squared
    are a floating-point limit).
    """
    s = _plain(ssr)
    q = s / _plain(n_samples)
    finite_ssr = np.all(np.isfinite(s))
    trigger_if(np.any(q < 0.0) or (finite_ssr and not np.all(np.isfinite(q))), "AP-SWE-013")


@_guard
def snr_noise_nonzero(noise, signal):
    """AP-SWE-014: ``signal / noise``. Zero noise with zero signal (zero
    exposure, 0/0) is excluded; a nonzero signal over zero noise is an
    unguarded divisor. An infinite noise is a floating-point limit (squared
    read noise or counts overflowing) and is excluded; NaN noise is not.
    """
    n, sg = _plain(noise), _plain(signal)
    trigger_if(np.any((n == 0) & (sg != 0)) or np.any(np.isnan(n)), "AP-SWE-014")


# --- convolution -------------------------------------------------------------

@_guard
def convolve_fft_output_shape(array, out, crop):
    """AP-SWE-015: a cropped FFT convolution must keep the input array's shape."""
    if crop:
        trigger_if(tuple(np.shape(out)) != tuple(np.shape(array)), "AP-SWE-015")


@_guard
def convolve_output_finite(array, kernel, result, mask=None):
    """AP-SWE-016: finite array and kernel must give a finite result, unless the
    worst-case sum of products could exceed float64 (excluded as ordinary overflow).
    Masked input is excluded: masked pixels are turned into NaN, and a window
    that contains only masked pixels gives NaN by documented design.
    """
    if mask is not None and np.any(np.asarray(mask) != 0):
        return
    if np.ma.is_masked(array):
        return
    # Use the caller's original inputs: nan_treatment/preserve_nan legitimately
    # re-insert NaN into the output for NaN inputs.
    if hasattr(array, "array"):  # Kernel objects are handled before this point
        return
    a, k = _plain(array), _plain(kernel)
    if np.all(np.isfinite(a)) and np.all(np.isfinite(k)):
        if np.sum(np.abs(a)) * np.sum(np.abs(k)) < 1e300:
            trigger_if(_not_all_finite(result), "AP-SWE-016")


@_guard
def oversample_reshape(size, factor):
    """AP-SWE-017: ``reshape(x.size // factor, factor)`` needs divisibility."""
    trigger_if(int(size) % int(factor) != 0, "AP-SWE-017")


# --- visualization -----------------------------------------------------------

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
def ls_offset_weight_sum(w, dy):
    """AP-SWE-022: ``dot(y, w) / w.sum()``. Uncertainties outside [1e-100, 1e100]
    are excluded: ``dy ** -2`` leaves float64 range there by ordinary arithmetic.
    """
    d = _plain(dy)
    if np.all(np.isfinite(d)) and np.all((np.abs(d) >= 1e-100) & (np.abs(d) <= 1e100)):
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
def bls_depth_weight(ivar_m, dy):
    """AP-SWE-025: ``1.0 / np.sum(ivar[m])``; uncertainties outside [1e-100, 1e100]
    are excluded (the inverse variance underflows or overflows by ordinary arithmetic).
    """
    d = _plain(dy)
    if np.all(np.isfinite(d)) and np.all((np.abs(d) >= 1e-100) & (np.abs(d) <= 1e100)):
        trigger_if(np.sum(_plain(ivar_m)) == 0, "AP-SWE-025")


@_guard
def ls_weighted_mean_denominator(denominator, dy):
    """AP-SWE-026: ``_weighted_sum(val, dy) / _weighted_sum(ones, dy)``; uncertainties
    outside [1e-100, 1e100] are excluded as ordinary float64 under/overflow.
    """
    dd = _plain(dy)
    if np.all(np.isfinite(dd)) and np.all((np.abs(dd) >= 1e-100) & (np.abs(dd) <= 1e100)):
        d = _plain(denominator)
        trigger_if(np.any(d == 0) or not np.all(np.isfinite(d)), "AP-SWE-026")


# --- nddata ------------------------------------------------------------------

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
    """AP-SWE-031: ``x / amplitude`` divides by the amplitude parameter. NaN from
    arcsin outside [-1, 1] is the model's mathematical domain and is excluded.
    """
    trigger_if(_any_zero(amplitude), "AP-SWE-031")


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
def flat_age_denominator(Om0):
    """AP-SWE-035: ``... / sqrt(1 - Om0)``."""
    trigger_if(1.0 - float(Om0) == 0.0, "AP-SWE-035")
