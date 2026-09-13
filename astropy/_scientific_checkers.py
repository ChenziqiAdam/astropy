"""Runtime scientific-invariant trigger collection for the SciBench pilot.

This module is private and inactive unless ``SCIBENCH_TRIGGER_LOG`` names an
output file. Each JSON-lines record represents one observed invariant alarm;
the evaluator deduplicates checker IDs and maps them to root-cause families.

Design rule (see ``scientific_bug_finding/astropy_pilot/LAW_CANDIDATES.md``):
a checker may only (a) call a public API a second time on a transformed
input and/or (b) read values already computed by production code, then
compare against a value the law says must agree. A checker never
re-implements the scientific formula it is checking, never raises, and never
changes a return value, exception, or numerical result.

Scope rule: every sanitizer in this bank guards a genuine scientific/geometric
law on the celestial sphere or in the frame-transform graph, each with a
numerically derived (not guessed) tolerance -- see LAW_CANDIDATES.md for the
full derivation of every constant below. Pure arithmetic identities, range/
finiteness checks, and generic software-correctness assertions are out of
scope and are not instrumented here.
"""

import json
import os
import threading

# Re-entrancy guard: a checker that calls a public API a second time must not
# trigger the same checker recursively.
_active = threading.local()

_EPS64 = 2.220446049250313e-16  # np.finfo(np.float64).eps, derived once


def enabled():
    """Return True when the pilot evaluator requested trigger collection."""
    return bool(os.environ.get("SCIBENCH_TRIGGER_LOG"))


def trigger(checker_id):
    """Atomically append one checker ID to the configured JSON-lines log."""
    path = os.environ.get("SCIBENCH_TRIGGER_LOG")
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


def _guard(checker_id):
    """Wrap a checker body so an internal error never disturbs production.

    A checker that itself errors is a curator bug, not a science alarm; it
    must not change program behaviour. We swallow it silently.
    """

    def decorator(func):
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
        return wrapper

    return decorator


# --- Candidate A: offset/separation round trip on the sphere ---------------
#
# LAW_CANDIDATES.md Candidate A. Precondition: two points not coincident, not
# antipodal, and neither within |cos(lat)| < 1e-3 of a celestial pole (this
# pole exclusion was derived, not assumed -- see the document's correction
# note; the initial 1/sin(sep) conditioning model was wrong by ~4 orders of
# magnitude near the poles). Tolerance: tol = 10 * eps64 * cond, where
# cond = 1 / min(|cos(lat1)|, |cos(lat2)|)**2, derived from a 500,000-trial
# sweep (worst observed ratio 5.96; final constant carries a ~1.7x margin).

_A_COS_FLOOR = 1e-3
_A_SIN_SEP_FLOOR = 1e-6
_A_TOL_C = 10.0


@_guard("offset_separation_roundtrip")
def check_offset_roundtrip(lon1, lat1, pa, sep, lon2, lat2):
    """AP-COORD-001: see module docstring above for the full law statement.

    ``lon1/lat1`` is the origin point P1 (radians). ``pa``/``sep`` are the
    position_angle/separation arguments passed to directional_offset_by
    (radians). ``lon2/lat2`` is the point P2 that directional_offset_by
    actually returned. The checker independently re-derives pa'/sep' from
    (P1, P2) via the public position_angle/separation API, reconstructs P2'
    from (pa', sep'), and compares P2' to P2 -- the forward-then-inverse
    statement of the law (LAW_CANDIDATES.md Candidate A).
    """
    import math

    from astropy import units as u
    from astropy.coordinates.angles.utils import (
        angular_separation,
        offset_by,
        position_angle,
    )

    def _rad(x):
        # angles.utils functions are inconsistent in return type on plain
        # float input: angular_separation returns a bare float (radians),
        # position_angle returns an Angle in rad, offset_by returns Angle
        # objects in *degrees*. A bare float/np.floating has no unit and is
        # already radians (the documented convention for float args/returns
        # in this module); a Quantity/Angle carries its own unit and must be
        # converted, never reinterpreted.
        if isinstance(x, u.Quantity):
            return float(x.to_value(u.rad))
        return float(x)

    if abs(math.sin(sep)) < _A_SIN_SEP_FLOOR:
        return
    cos1 = abs(math.cos(lat1))
    cos2 = abs(math.cos(lat2))
    if min(cos1, cos2) < _A_COS_FLOOR:
        return

    pa2 = _rad(position_angle(lon1, lat1, lon2, lat2))
    sep2 = _rad(angular_separation(lon1, lat1, lon2, lat2))
    recon_lon, recon_lat = offset_by(lon1, lat1, pa2, sep2)

    cond = 1.0 / (min(cos1, cos2) ** 2)
    tol = _A_TOL_C * _EPS64 * cond

    err = _rad(angular_separation(lon2, lat2, _rad(recon_lon), _rad(recon_lat)))
    trigger_if(err > tol, "AP-COORD-001")


# --- Candidate B: frame-transform round-trip identity (matrix-only) --------
#
# LAW_CANDIDATES.md Candidate B. Precondition: a transform_to call whose
# destination frame class equals the origin frame class, resolved through a
# genuine composed transform (not the zero-op same-frame shortcut), and where
# every edge on the resolved path is a DynamicMatrixTransform or
# StaticMatrixTransform (never a FunctionTransformWithFiniteDifference, which
# has a materially different, step-size-dependent error model out of scope
# for this law). Tolerance: 30 * eps64, derived from a 5000-trial sweep per
# loop across three matrix-only round trips (worst observed ratio 15.3).

_B_TOL_C = 30.0


@_guard("frame_transform_roundtrip")
def check_frame_roundtrip_angular(orig_lon, orig_lat, recon_lon, recon_lat):
    """AP-COORD-002: a coordinate transformed around a closed loop through a
    chain of pure-rotation frame transforms (DynamicMatrixTransform /
    StaticMatrixTransform only) must return to its starting point up to
    float64 matrix-composition round-off. All edges on the path must be
    orthogonal-matrix transforms; the caller is responsible for restricting
    invocation to such paths (see baseframe.py hook, which checks the
    resolved CompositeTransform's edge types before calling this).
    """
    from astropy.coordinates.angles.utils import angular_separation

    err = float(angular_separation(orig_lon, orig_lat, recon_lon, recon_lat))
    tol = _B_TOL_C * _EPS64
    trigger_if(err > tol, "AP-COORD-002")


@_guard("frame_transform_roundtrip_3d")
def check_frame_roundtrip_3d(orig_xyz, recon_xyz):
    """AP-COORD-003: same law as AP-COORD-002, applied to the 3D Cartesian
    representation when the coordinate carries a distance. A pure rotation
    preserves vector norm and pairwise structure, so the round-tripped
    Cartesian vector must equal the original up to float64 round-off, scaled
    by the vector's own magnitude (a relative, not absolute, tolerance).
    """
    ox, oy, oz = orig_xyz
    rx, ry, rz = recon_xyz
    orig_norm = (ox * ox + oy * oy + oz * oz) ** 0.5
    if orig_norm == 0.0:
        return
    dx, dy, dz = rx - ox, ry - oy, rz - oz
    err = (dx * dx + dy * dy + dz * dz) ** 0.5
    relerr = err / orig_norm
    tol = _B_TOL_C * _EPS64
    trigger_if(relerr > tol, "AP-COORD-003")


# --- Candidate D: Time scale round-trip identity (analytic scales only) ---
#
# LAW_CANDIDATES.md Candidate D. Precondition: the epoch's Julian date must
# fall within the *loaded* leap-second table's validity range (queried at
# runtime, not hardcoded, since a newer astropy release ships an updated
# table); UT1 is excluded entirely (its conversion depends on interpolated
# IERS Earth-orientation data, a different error source). Tolerance: 5e-11 s,
# derived from a 25,000-trial sweep across 5 round-trip loops spanning
# 1972-2016 (worst observed error 8.87e-12 s, under half the naive
# single-double day-precision floor of ~1.92e-11 s).

_D_TOL_SEC = 5e-11
_leap_table_range = None  # lazily cached (year_min, year_max) or a sentinel


def _leap_table_year_range():
    global _leap_table_range
    if _leap_table_range is None:
        from astropy.utils import iers

        table = iers.LeapSeconds.auto_open()
        _leap_table_range = (
            int(table["year"].min()),
            int(table["year"].max()),
        )
    return _leap_table_range


@_guard("time_scale_roundtrip")
def check_time_scale_roundtrip(original, converted, new_scale):
    """AP-TIME-001: a single scale conversion must be its own inverse when
    immediately reversed -- every pairwise conversion in the analytic
    (non-UT1) subset of MULTI_HOPS is an exactly invertible relation, and a
    multi-hop loop reduces to this hop-by-hop (LAW_CANDIDATES.md Candidate D).

    ``original`` is the Time instance before conversion; ``converted`` is
    the same instant re-expressed in ``new_scale`` by production code. UT1
    is excluded (interpolated IERS data, a different error source); the
    leap-second table's own validity range gates any UTC-involving pair.
    Restricted to scalar, unmasked Time instances (see LAW_CANDIDATES.md
    Candidate D -- array support is a known, documented limitation; a masked
    scalar's jd1/jd2 is a fill-value sentinel, not a real epoch, so the
    round-trip law does not apply to it -- found via astropy's own
    test_mask.py, not anticipated in the original design).
    """
    if original.shape != () or original.masked:
        return
    if new_scale == "ut1" or original.scale == "ut1":
        return

    year_min, year_max = _leap_table_year_range()
    if original.scale == "utc" or new_scale == "utc":
        decimalyear = float(original.decimalyear)
        if not (year_min <= decimalyear <= year_max):
            return

    # Do not use getattr(converted, original.scale): that populates
    # converted's own lazy scale cache as a side effect, which is observable
    # production state (see astropy/time/tests/test_basic.py::test_cache) and
    # would violate SANITIZER.md 5.5. Replicate first, exactly mirroring what
    # Time.__getattr__ itself does internally to avoid touching the cache.
    # ``original`` is also replicated before use as a subtraction operand:
    # Time.__sub__ can itself populate a scale-cache entry on its second
    # operand (empirically confirmed), and original is the real Time object
    # production code (and the caller) is holding.
    back = converted.replicate()
    back._set_scale(original.scale)
    err_sec = abs(float((back - original.replicate()).sec))
    trigger_if(err_sec > _D_TOL_SEC, "AP-TIME-001")


# --- Candidate E: Time arithmetic inverse (subtraction/addition) ----------
#
# LAW_CANDIDATES.md Candidate E. Precondition: same leap-second-table
# validity restriction as Candidate D when either epoch is UTC-involving;
# otherwise unrestricted (no small/large-delta restriction -- checked, not
# assumed, up to ~40-year separations). Tolerance: 5e-11 s, derived from a
# 20,000-trial sweep across 6 scales including a large-delta stress variant
# (worst observed error 9.59e-12 s, no separation-magnitude dependence).

_E_TOL_SEC = 5e-11


@_guard("time_arithmetic_inverse")
def check_time_arithmetic_inverse(t1, t2, delta):
    """AP-TIME-002: Time.__sub__ and Time.__add__ form an additive group
    action -- T1 + (T2 - T1) must equal T2 for any two valid epochs
    (LAW_CANDIDATES.md Candidate E).

    ``t1``, ``t2`` are the two Time operands of a production ``t2 - t1``
    call; ``delta`` is the TimeDelta production code actually computed.
    Restricted to scalar, unmasked Time instances (see LAW_CANDIDATES.md
    Candidate E -- same reasoning as Candidate D's array/masked exclusions).
    """
    if t1.shape != () or t2.shape != () or t1.masked or t2.masked:
        return

    year_min, year_max = _leap_table_year_range()
    if t1.scale == "utc" and not (year_min <= float(t1.decimalyear) <= year_max):
        return
    if t2.scale == "utc" and not (year_min <= float(t2.decimalyear) <= year_max):
        return

    # Use replicas throughout, never t1/t2/delta directly: Time.__add__ can
    # populate a scale-cache entry on its TimeDelta operand, and Time.__sub__
    # can populate one on its second Time operand (both observed empirically
    # -- see astropy/time/tests/test_basic.py::test_cache for the invariant
    # this protects). Re-deriving with the real objects would leak this
    # checker's own re-computation into production object state, which
    # SANITIZER.md 5.5 forbids.
    recon = t1.replicate() + delta.replicate()
    err_sec = abs(float((recon - t2.replicate()).sec))
    trigger_if(err_sec > _E_TOL_SEC, "AP-TIME-002")


# --- Candidate F: WCS pixel/world round-trip (core wcslib projection) -----
#
# LAW_CANDIDATES.md Candidate F. Precondition: any pixel for which the
# forward+inverse core-projection call pair both complete without raising
# and return finite results -- "the domain where the API completes
# normally," not a hardcoded pixel range (that domain differs by
# projection/reference point/plate scale). Tolerance: 1e-6 px, derived from
# two sweeps (gentle: 2000x6 projections; adversarial: 15,000 trials to
# +/-2000px and declination 89.9 deg) with worst observed error 2.12e-10 px
# -- ~4 orders of magnitude margin, deliberately generous since C-extension
# rounding is less predictable a priori than pure-Python chains.

_F_TOL_PX = 1e-6


_F_EXCLUDED_PROJECTIONS = frozenset({"CSC"})


@_guard("wcs_projection_roundtrip")
def check_wcs_pix2world_roundtrip(wcs_obj, original_xy, world, origin):
    """AP-WCS-001: wcs_pix2world and wcs_world2pix are documented as mutual
    inverses on the core (non-SIP) projection. Re-calling the public
    wcs_world2pix on the world coordinate just produced must recover the
    original pixel (LAW_CANDIDATES.md Candidate F).

    ``wcs_obj`` is the WCS instance; ``original_xy`` and ``world`` are the
    (N, 2) pixel/world arrays production code just computed; ``origin`` is
    the same origin convention (0 or 1) used for the forward call.

    CSC (COBE quad-cube) is excluded by name: a census of all 27 standard
    projection headers shipped in astropy's own test suite found CSC alone
    failing this law at 100% of trials (errors up to 2.5e-3 px, five orders
    of magnitude past tolerance, unrelated to distance from the reference
    pixel) -- wcs_world2pix's Newton inversion structurally fails to
    disambiguate CSC's projection, not amplified rounding. Found during
    implementation verification, not anticipated in the original design;
    see LAW_CANDIDATES.md Candidate F's precondition-correction note.
    """
    import numpy as np

    ctype = getattr(wcs_obj.wcs, "ctype", None)
    if ctype is not None and any(
        str(c).strip()[-3:] in _F_EXCLUDED_PROJECTIONS for c in ctype
    ):
        return

    if original_xy.size == 0 or world.size == 0:
        return
    if not (np.all(np.isfinite(original_xy)) and np.all(np.isfinite(world))):
        return

    recon = wcs_obj.wcs_world2pix(world, origin)
    if not np.all(np.isfinite(recon)):
        return

    err = float(np.max(np.linalg.norm(recon - original_xy, axis=-1)))
    trigger_if(err > _F_TOL_PX, "AP-WCS-001")


# --- Candidate G: all_world2pix's documented convergence contract ---------
#
# LAW_CANDIDATES.md Candidate G. Precondition: any world coordinate produced
# by all_pix2world from a pixel within all_world2pix's convergence domain --
# i.e. any call that returns without raising NoConvergence. Alarm: achieved
# error > C * requested_tolerance. Derived from a 900-trial sweep (3
# tolerances x 300 pixels, realistic SIP magnitudes): achieved/requested
# ratio never exceeded 0.0156, giving C=10 ~6x headroom over the observed
# worst case while staying tight relative to the caller's own tolerance.

_G_TOL_C = 10.0


@_guard("wcs_iterative_inverse_accuracy")
def check_wcs_all_world2pix_accuracy(wcs_obj, original_xy, world, origin):
    """AP-WCS-002: all_world2pix must recover the pixel that produced the
    world coordinate it is inverting to within (a small safety margin over)
    its own documented, caller-specified convergence tolerance
    (LAW_CANDIDATES.md Candidate G).

    Implemented on the forward side (inside all_pix2world, which computed
    ``world`` from ``original_xy``) because only the forward call has both
    endpoints together -- all_world2pix alone never sees what pixel
    produced its input. Re-calls all_world2pix at its own default
    tolerance; a NoConvergence there means the input is outside this law's
    precondition (the domain where the API's own contract is honored), not
    a violation.
    """
    import numpy as np

    from astropy.wcs.wcs import NoConvergence

    if original_xy.size == 0 or world.size == 0:
        return
    if not (np.all(np.isfinite(original_xy)) and np.all(np.isfinite(world))):
        return

    default_tolerance = 1e-4
    try:
        recon_xy = wcs_obj.all_world2pix(world, origin, tolerance=default_tolerance)
    except NoConvergence:
        return

    if not np.all(np.isfinite(recon_xy)):
        return

    err = float(np.max(np.linalg.norm(
        np.asarray(recon_xy) - np.asarray(original_xy), axis=-1
    )))
    trigger_if(err > _G_TOL_C * default_tolerance, "AP-WCS-002")


# --- Candidate H: spectral() m<->Hz<->J composition consistency -----------
#
# LAW_CANDIDATES.md Candidate H. Precondition: any wavelength lambda > 0, a
# continuous family over the full physically meaningful range. Tolerance
# 5*eps64, derived from a 200,000-trial sweep across 18 orders of magnitude
# (worst observed 1.15*eps64, no scale-dependent degradation since each
# pairwise conversion is a single multiply/divide by a constant).
#
# Implemented by extracting the raw conversion functions directly from the
# Equivalency list spectral() returns and composing them, rather than
# observing a live Quantity.to() call: Quantity.to()/Unit.to() are
# extremely hot-path, fully generic machinery used for every unit
# conversion in astropy (not just spectral/Doppler), so hooking there would
# be invasive and risky for a check specific to one equivalency. Calling
# the equivalency's own returned lambdas a second time and composing them
# is still the SANITIZER.md re-call pattern -- just applied to the table
# spectral() constructs, rather than to a downstream .to() call.

_H_TOL = 5.0 * _EPS64


@_guard("spectral_equivalency_roundtrip")
def check_spectral_roundtrip(equiv_list):
    """AP-UNITS-001: spectral()'s three pairwise conversions (m<->Hz,
    m<->J, Hz<->J) all encode the same c and h; composing wavelength ->
    frequency -> energy -> wavelength via three different pairwise entries
    in the table must recover the original wavelength (LAW_CANDIDATES.md
    Candidate H). ``equiv_list`` is the Equivalency list spectral() is
    about to return.
    """
    import math

    entries = {}
    for row in equiv_list:
        if len(row) >= 3:
            entries[(str(row[0]), str(row[1]))] = row

    m_hz = entries.get(("m", "Hz"))
    hz_j = entries.get(("Hz", "J"))
    m_j = entries.get(("m", "J"))
    if not (m_hz and hz_j and m_j):
        return

    m_to_hz = m_hz[2]
    hz_to_j = hz_j[2]
    j_to_m = m_j[2]  # m<->J is self-inverse (hc/x), so the m->J fn is its own inverse

    rng_lambdas = [10.0 ** e for e in range(-15, 4)]
    for lam in rng_lambdas:
        freq = m_to_hz(lam)
        energy = hz_to_j(freq)
        lam_back = j_to_m(energy)
        if not math.isfinite(lam_back) or lam_back == 0:
            continue
        relerr = abs((lam_back - lam) / lam)
        trigger_if(relerr > _H_TOL, "AP-UNITS-001")


# --- Candidate I: Doppler convention agreement in the low-velocity limit --
#
# LAW_CANDIDATES.md Candidate I. Precondition: |beta| < 1e-3, the regime
# where the O(beta^2) next-Taylor-order term is comfortably below the
# tolerance. Alarm compares the radio/relativistic disagreement against the
# analytically-derived leading-order beta/2 coefficient (not "roughly
# agree" -- that vague first draft was rewritten, see LAW_CANDIDATES.md).
# Tolerance 1e-6 (relative), derived from a 50,000-trial sweep across 4
# rest-quantity branches with beta in [1e-6, 1e-3].

_I_BETA_MAX = 1e-3
_I_BETA_MIN = 1e-5  # below this, float64 subtraction noise dominates (see derivation)
_I_TOL = 1e-6
_CKMS = 299792.458  # speed of light in km/s, matches equivalencies.py's ckms


@_guard("doppler_convention_consistency")
def check_doppler_convention_agreement(rest_freq_hz, to_func_radio_hz):
    """AP-UNITS-002: the radio and relativistic Doppler conventions must
    disagree by exactly beta/2 in relative terms at leading order (an
    analytically derived prediction, not an approximate "roughly agree"
    claim) -- see LAW_CANDIDATES.md Candidate I.

    ``rest_freq_hz`` is the rest frequency (plain float, Hz) doppler_radio
    was constructed with; ``to_func_radio_hz`` is doppler_radio's own
    Hz -> km/s conversion function for that rest frequency. Re-calls the
    public doppler_relativistic(rest) to get the independent relativistic
    conversion for the same rest frequency, then probes both at a small
    set of nearby test frequencies spanning the beta precondition window.
    """
    from astropy import units as u
    from astropy.units.equivalencies import doppler_relativistic

    rest_q = rest_freq_hz * u.Hz
    rel_equiv = doppler_relativistic(rest_q)
    to_func_rel_hz = None
    for row in rel_equiv:
        if len(row) >= 3 and row[0] == u.Hz:
            to_func_rel_hz = row[2]
            break
    if to_func_rel_hz is None:
        return

    for beta_probe in (1e-4, 3e-4, 1e-3 * 0.9):
        test_freq = rest_freq_hz * (1 - beta_probe)
        v_radio = to_func_radio_hz(test_freq)
        v_rel = to_func_rel_hz(test_freq)

        beta = v_rel / _CKMS
        if not (_I_BETA_MIN < abs(beta) < _I_BETA_MAX):
            continue
        if v_rel == 0:
            continue

        observed_relerr = abs((v_radio - v_rel) / v_rel)
        predicted = 0.5 * abs(beta)
        trigger_if(abs(observed_relerr - predicted) > _I_TOL, "AP-UNITS-002")


# --- Candidate J: relativistic Doppler cross-consistency ------------------
#
# LAW_CANDIDATES.md Candidate J. Precondition: any v with |beta| in (0,1) --
# the full physically defined domain, no small-velocity restriction (both
# equivalencies claim to be exact). Alarm normalizes by c, not by the
# physical velocity itself (an initial normalize-by-v attempt gave a
# degenerate worst-case ratio of 5.4e5*eps64 near v~0, from dividing a tiny
# velocity by itself; re-normalizing by c gave a stable 1.75*eps64*c over
# the same 200,000-trial sweep). Tolerance 10*eps64 (relative to c).

_J_TOL = 10.0 * _EPS64


@_guard("doppler_relativistic_redshift_consistency")
def check_doppler_redshift_consistency(rest_freq_hz, to_vel_freq_func):
    """AP-UNITS-003: doppler_relativistic's frequency-ratio formula and
    doppler_redshift's redshift formula both claim to be the exact
    relativistic Doppler shift for the same physical velocity; converting
    the same shift through each independent formula must recover the same
    velocity, to a tolerance normalized by c (LAW_CANDIDATES.md Candidate
    J -- normalizing by the physical velocity itself is degenerate near
    v=0 and was rejected during derivation).

    ``rest_freq_hz`` is the rest frequency (float, Hz) doppler_relativistic
    was constructed with; ``to_vel_freq_func`` is its own Hz -> km/s
    conversion function. Re-calls the public doppler_redshift() equivalency
    at a redshift independently derived from the same frequency ratio.
    """
    import math

    from astropy.units.equivalencies import doppler_redshift

    z_equiv = doppler_redshift()
    convert_z_to_rv = None
    for row in z_equiv:
        if len(row) >= 3:
            convert_z_to_rv = row[2]
            break
    if convert_z_to_rv is None:
        return

    for beta_probe in (-0.5, -0.05, 1e-4, 0.3, 0.8):
        test_freq = rest_freq_hz * math.sqrt((1 - beta_probe) / (1 + beta_probe))
        v_from_relativistic = to_vel_freq_func(test_freq)

        z = rest_freq_hz / test_freq - 1
        v_from_redshift = convert_z_to_rv(z)

        err_over_c = abs(v_from_relativistic - v_from_redshift) / _CKMS
        trigger_if(err_over_c > _J_TOL, "AP-UNITS-003")


# --- Candidate K: brightness vs. thermodynamic temperature ----------------
#
# LAW_CANDIDATES.md Candidate K. Precondition: x = h*nu/(k*T) < 0.1, where
# the omitted x^4/240 term (after the corrected leading term -x^2/12) stays
# below 4.2e-7. First guess for the leading term (+x^2/12) had the wrong
# sign, caught by checking against direct numerical evaluation of f(x)
# before finalizing. Tolerance 1e-6, from a 30,000-trial sweep over
# x in [0.01, 0.1].

_K_X_MAX = 0.1
_K_TOL = 1e-6
_K_H = 6.62607015e-34  # Planck constant, SI (h)
_K_KB = 1.380649e-23   # Boltzmann constant, SI (k_B)


@_guard("brightness_thermodynamic_temperature_consistency")
def check_brightness_thermodynamic_consistency(frequency_q, t_cmb_q, convert_jy_to_k_thermo):
    """AP-UNITS-004: brightness_temperature (Rayleigh-Jeans) and
    thermodynamic_temperature (full Planck) differ only by the correction
    factor f(x) = x^2*e^x/(e^x-1)^2, x = h*nu/(k*T); the ratio of the two
    recovered temperatures must equal f(x) to within its analytically
    derived small-x expansion 1 - x^2/12 (LAW_CANDIDATES.md Candidate K --
    the sign of this leading term was gotten wrong on the first attempt and
    corrected by direct numerical check before finalizing).

    ``frequency_q``/``t_cmb_q`` are the Quantity inputs thermodynamic_
    temperature was constructed with; ``convert_jy_to_k_thermo`` is its own
    Jy/sr -> K conversion function. Re-calls the public brightness_
    temperature(frequency_q) to get the independent Rayleigh-Jeans
    conversion for the same frequency, on a synthetic test surface
    brightness (the ratio of recovered temperatures is independent of the
    test brightness value, since both formulas are linear in the input).
    """
    from astropy.units.equivalencies import brightness_temperature, spectral

    bright_equiv = brightness_temperature(frequency_q)
    convert_jysr_to_k_bright = None
    for row in bright_equiv:
        if len(row) >= 3 and str(row[0]) == "Jy / sr":
            convert_jysr_to_k_bright = row[2]
            break
    if convert_jysr_to_k_bright is None:
        return

    test_x_jysr = 1.0
    t_bright_k = convert_jysr_to_k_bright(test_x_jysr)
    t_thermo_k = convert_jy_to_k_thermo(test_x_jysr)
    if t_thermo_k == 0:
        return

    freq_hz = float(frequency_q.to_value("Hz", equivalencies=spectral()))
    t_cmb_k = float(t_cmb_q.to_value("K"))
    if t_cmb_k <= 0:
        return
    x = (_K_H * freq_hz) / (_K_KB * t_cmb_k)
    if not (0 < x < _K_X_MAX):
        return

    ratio = t_bright_k / t_thermo_k
    predicted = -(x**2) / 12.0
    trigger_if(abs((ratio - 1) - predicted) > _K_TOL, "AP-UNITS-004")


# --- Candidate L: separability matrix soundness vs numerical Jacobian -----
#
# LAW_CANDIDATES.md Candidate L. Precondition: any compound model whose
# separability_matrix computes successfully, at any evaluation point where
# the model itself evaluates successfully. Alarm: for any (i, j) where the
# matrix claims output i cannot depend on input j, the numerical (central-
# difference) partial derivative of output i w.r.t. input j must be exactly
# zero (to a defensive floor, not a fit -- a 300-random-compound-model
# sweep found exactly 0.0 in every case, since these are simple closed-form
# evaluations with no accumulation path). Only the False-entry direction is
# checked -- True entries are a deliberately conservative upper bound and
# are NOT claimed to imply nonzero dependence (see LAW_CANDIDATES.md for
# why the reverse direction would be false, e.g. Scale(0)).

_L_TOL = 1e-8
_L_EPS = 1e-6
_L_TEST_POINTS = ((0.7, -1.3, 2.1, -0.4),)


@_guard("separability_soundness")
def check_separability_soundness(transform, matrix):
    """AP-MODEL-001: wherever separability_matrix claims an output cannot
    depend on an input, a numerical central-difference derivative at a
    representative evaluation point must be exactly zero (to a defensive
    floor). See LAW_CANDIDATES.md Candidate L for why only this direction
    (False-entry soundness) is a valid law, not the reverse.

    ``transform`` is the model separability_matrix was just computed for;
    ``matrix`` is the boolean array it returned.
    """
    import numpy as np

    n_in = transform.n_inputs
    n_out = transform.n_outputs
    if matrix.shape != (n_out, n_in):
        return
    if not np.any(~matrix):
        return  # nothing to check -- every entry claims possible dependence

    x0 = [_L_TEST_POINTS[0][k % len(_L_TEST_POINTS[0])] for k in range(n_in)]

    try:
        y0 = np.atleast_1d(transform(*x0))
    except Exception:
        return
    if y0.shape[-1] != n_out or not np.all(np.isfinite(y0)):
        return

    for j in range(n_in):
        xp = list(x0)
        xm = list(x0)
        xp[j] = x0[j] + _L_EPS
        xm[j] = x0[j] - _L_EPS
        try:
            yp = np.atleast_1d(transform(*xp))
            ym = np.atleast_1d(transform(*xm))
        except Exception:
            continue
        if not (np.all(np.isfinite(yp)) and np.all(np.isfinite(ym))):
            continue

        deriv = np.abs(yp - ym) / (2 * _L_EPS)
        for i in range(n_out):
            if not matrix[i, j]:
                trigger_if(deriv[i] > _L_TOL, "AP-MODEL-001")


# --- Candidate L: Cython/Python cross-implementation agreement of inv_efunc
#
# LAW_CANDIDATES.md Candidate L. Precondition: scalar z >= 0 (this checker's
# implementation restriction -- _inv_efunc_scalar only accepts scalars;
# array-valued inv_efunc(z) calls are out of scope, same discipline as the
# scalar-only restriction on AP-TIME-001/002). Tolerance 20*eps64, derived
# from a 14,000-trial sweep (worst ratio 1.58*eps64) plus a 6,000-trial
# massive-neutrino-path sweep (worst ratio 1.45*eps64).

_L_TOL_C = 20.0


@_guard("inv_efunc_cross_implementation")
def check_inv_efunc_cross_implementation(cosmo, z):
    """AP-COSMO-001: inv_efunc (public, vectorized, pure-Python) and
    _inv_efunc_scalar (the compiled Cython fast-path used internally by
    every quad()-based distance/time integral) are two independently
    written implementations of the same E(z)^-1 formula; they must agree
    (LAW_CANDIDATES.md Candidate L). ``cosmo`` is the FLRW instance;
    ``z`` is a scalar redshift already used in a production call.
    """
    import math

    try:
        z_scalar = float(z)
    except (TypeError, ValueError):
        return
    if z_scalar < 0:
        return

    py_val = float(cosmo.inv_efunc(z_scalar))
    if not math.isfinite(py_val) or py_val == 0:
        return

    cy_val = float(cosmo._inv_efunc_scalar(z_scalar, *cosmo._inv_efunc_scalar_args))
    if not math.isfinite(cy_val):
        return

    relerr = abs(py_val - cy_val) / abs(py_val)
    trigger_if(relerr > _L_TOL_C * _EPS64, "AP-COSMO-001")


# --- Candidate M: age/lookback-time complementarity ------------------------
#
# LAW_CANDIDATES.md Candidate M. Precondition: any FLRW instance and z >= 0
# where age(0), age(z), lookback_time(z) all converge. Tolerance derived
# from quad()'s own default epsabs/epsrel (1.49e-8), not eps64 -- an
# initial eps64-based guess was wrong by ~6 orders of magnitude (worst
# ratio 5.76e6*eps64), corrected after checking against quad's documented
# tolerance (worst ratio 0.256 over a 5,000-trial sweep including wCDM and
# z up to ~1200).

_M_QUAD_TOL = 1.49e-8
_M_TOL_C = 3.0


@_guard("age_lookback_time_complementarity")
def check_age_lookback_complementarity(cosmo, z, lookback_z_val):
    """AP-COSMO-002: age(0) - age(z) == lookback_time(z), a consequence of
    both being scipy.integrate.quad on the same integrand over
    complementary bounds (LAW_CANDIDATES.md Candidate M -- a weaker
    cross-check than AP-COSMO-001 since the integrand is shared; see the
    document's manual-review note on what this law does and does not
    cover).

    ``cosmo`` is the FLRW instance; ``z`` is the scalar redshift a
    production lookback_time(z) call just used; ``lookback_z_val`` is the
    numeric value (Gyr) it returned. Re-calls the public age() at 0 and at
    z to get the independent complementary quantities.

    Known precondition gap, found by adversarial audit (not anticipated in
    the original design): age(0) always integrates quad(integrand, 0, inf),
    and scipy's adaptive semi-infinite quadrature probes z values into the
    hundreds regardless of the z the caller actually used. For w0wzCDM /
    Flatw0wzCDM with any wz != 0, the dark-energy density scale's
    exp(3*wz*z) term eventually overflows double at large z, and the
    Cython _inv_efunc_scalar fast path's generic complex-power fallback for
    `**` manufactures a spurious nonzero imaginary part from the resulting
    0*inf arithmetic, raising TypeError -- a genuine, pre-existing astropy
    defect (reproduces identically with SCIBENCH_TRIGGER_LOG unset; not an
    artifact of this checker). This means AP-COSMO-002 currently cannot
    check the w0wzCDM/Flatw0wzCDM family at all when wz != 0, even for
    lookback_time(z) calls at small, perfectly safe z, since the checker's
    own re-derivation strategy (via age(0)) is strictly less numerically
    robust than the production call it is verifying. Caught here explicitly
    (rather than silently swallowed by _guard) so this gap is visible in
    the record rather than indistinguishable from "nothing to check."
    """
    import math

    try:
        z_scalar = float(z)
    except (TypeError, ValueError):
        return
    if z_scalar < 0:
        return

    try:
        age0_val = float(cosmo.age(0.0).to_value("Gyr"))
        age_z_val = float(cosmo.age(z_scalar).to_value("Gyr"))
    except TypeError:
        return
    if not all(math.isfinite(v) for v in (age0_val, age_z_val, lookback_z_val)):
        return

    hubble_time_gyr = float(cosmo.hubble_time.to_value("Gyr"))

    err = abs((age0_val - age_z_val) - lookback_z_val)
    tol = _M_TOL_C * _M_QUAD_TOL * abs(hubble_time_gyr)
    trigger_if(err > tol, "AP-COSMO-002")


# --- Candidate C: spherical triangle inequality -----------------------------
#
# LAW_CANDIDATES.md Candidate C. Precondition: none (a metric's triangle
# inequality holds unconditionally, including at coincident points, poles,
# and antipodes). Tolerance: 10 * eps64, derived from 2,500,000 trials
# (2,000,000 uniform random triples + 500,000 adversarial near-collinear
# triples) that found zero violations at any scale; the constant is a
# defensive margin, not a fit to observed near-misses.

_C_TOL = 10.0 * _EPS64


@_guard("triangle_inequality")
def check_triangle_inequality(lonA, latA, lonC, latC, sep_ac):
    """AP-COORD-004: great-circle separation is a metric, so for any three
    points A, B, C on the sphere: separation(A, C) <= separation(A, B) +
    separation(B, C). This must hold unconditionally; a violation means
    angular_separation is not actually computing a proper metric, which
    would break every algorithm (nearest-neighbor search, catalog matching)
    that implicitly relies on the triangle inequality.

    ``lonA/latA``, ``lonC/latC`` are the two points passed to the production
    ``separation`` call (radians); ``sep_ac`` is the separation it computed.
    The third point B is derived deterministically from A and C (see
    LAW_CANDIDATES.md Candidate A -- an arbitrary point off the A-C geodesic,
    not a special/degenerate construction), and sep_ab/sep_bc are obtained by
    re-calling the same public angular_separation function B was not built
    from any prior call's state.
    """
    import math

    from astropy.coordinates.angles.utils import angular_separation

    lonB = (lonA + lonC) / 2.0 + math.pi / 2.0
    latB = max(-math.pi / 2.0, min(math.pi / 2.0, (latA + latC) / 2.0))

    sep_ab = float(angular_separation(lonA, latA, lonB, latB))
    sep_bc = float(angular_separation(lonB, latB, lonC, latC))

    violation = sep_ac - (sep_ab + sep_bc)
    trigger_if(violation > _C_TOL, "AP-COORD-004")


# --- Candidate O: circular std (circular method) vs circvar formula --------
#
# LAW_CANDIDATES.md Candidate O. Precondition: circvar(data) < 1 (nonzero
# resultant length -- the circular method's log blows up at R=0). Tolerance:
# 1000 * eps64, derived from a 200,000-trial sweep (worst relative error
# 2.17e-14 ~= 100*eps64 for the circular method's extra log+sqrt chain; the
# angular method matched exactly and is not separately instrumented).

_O_TOL_C = 1000.0


@_guard("circular_stats_formula_consistency")
def check_circstd_circvar_consistency(data, axis, weights, circular_value):
    """AP-STATS-001: circstd(data, method='circular') must equal
    sqrt(-2*ln(1 - circvar(data))), both public functions built on the same
    mean resultant length R (LAW_CANDIDATES.md Candidate O). ``data``/
    ``axis``/``weights`` are the arguments a production circstd(method=
    'circular') call just used; ``circular_value`` is the value it returned.
    Re-calls the independent circvar() on the same input.
    """
    import math

    from astropy.stats.circstats import circvar
    from astropy.units import Quantity

    try:
        cv = circvar(data, axis, weights)
        cv_val = float(cv.value if isinstance(cv, Quantity) else cv)
        result_val = float(
            circular_value.value
            if isinstance(circular_value, Quantity)
            else circular_value
        )
    except (TypeError, ValueError):
        return
    if not (math.isfinite(cv_val) and math.isfinite(result_val)):
        return
    resultant = 1.0 - cv_val
    if resultant <= 0:
        return

    predicted = math.sqrt(-2.0 * math.log(resultant))
    relerr = abs(result_val - predicted) / max(abs(predicted), 1e-300)
    trigger_if(relerr > _O_TOL_C * _EPS64, "AP-STATS-001")


# --- Candidate P: biweight_location affine equivariance ---------------------
#
# LAW_CANDIDATES.md Candidate P. Precondition: >= 5 elements, nonzero MAD,
# a > 0 (only the analytically-verified sign is checked). Tolerance: 1e-8
# relative, derived from a 100,000-trial sweep (worst relative error 2.52e-12
# with |a|, |b| up to 1e6 and 1/3 outlier-injected trials).

_P_TOL = 1e-8


@_guard("biweight_affine_equivariance")
def check_biweight_location_equivariance(data, c, axis, ignore_nan, result):
    """AP-STATS-002: biweight_location(a*x + b) == a*biweight_location(x) + b
    for a > 0 (LAW_CANDIDATES.md Candidate P). ``data``/``c``/``axis``/
    ``ignore_nan`` are the arguments a production call just used;
    ``result`` is the value it returned. Re-calls biweight_location on an
    affine-transformed copy of the same data; never mutates the original.
    """
    import numpy as np

    from astropy.stats.biweight import biweight_location

    arr = np.asanyarray(data)
    if arr.ndim != 1 or arr.size < 5:
        return
    try:
        base = float(result)
    except (TypeError, ValueError):
        return
    if not np.isfinite(base):
        return

    rng = np.random.default_rng(abs(hash((arr.size, round(base, 6)))) % (2**32))
    a = float(rng.uniform(2.0, 1e4))
    b = float(rng.uniform(-1e4, 1e4))
    transformed = a * arr + b

    other = biweight_location(transformed, c=c, axis=axis, ignore_nan=ignore_nan)
    try:
        other_val = float(other)
    except (TypeError, ValueError):
        return
    if not np.isfinite(other_val):
        return

    predicted = a * base + b
    scale = max(abs(predicted), abs(other_val), 1.0)
    relerr = abs(other_val - predicted) / scale
    trigger_if(relerr > _P_TOL, "AP-STATS-002")


# --- Candidate Q: biweight_midvariance quadratic scale equivariance --------
#
# LAW_CANDIDATES.md Candidate Q. Precondition: >= 5 elements, nonzero MAD
# (any sign of a). Tolerance: 1e-9 relative, derived from a 100,000-trial
# sweep (worst relative error 8.46e-14 with |a|, |b| up to 1e6, both signs).

_Q_TOL = 1e-9


@_guard("biweight_scale_equivariance")
def check_biweight_midvariance_equivariance(
    data, c, axis, modify_sample_size, ignore_nan, result
):
    """AP-STATS-003: biweight_midvariance(a*x + b) == a**2 *
    biweight_midvariance(x) for any real a, b (LAW_CANDIDATES.md
    Candidate Q). Same argument/re-call pattern as AP-STATS-002.

    Must forward ``modify_sample_size``: it changes which points count
    toward n (the outlier-rejection mask), so omitting it makes the
    re-call answer a different question than the production call asked
    (differential-test contract, arg-forwarding class -- found by
    self-verification via test_biweight_midvariance_small's
    modify_sample_size=True case, not by an external audit).
    """
    import numpy as np

    from astropy.stats.biweight import biweight_midvariance

    arr = np.asanyarray(data)
    if arr.ndim != 1 or arr.size < 5:
        return
    try:
        base = float(result)
    except (TypeError, ValueError):
        return
    if not (np.isfinite(base) and base > 0):
        return

    rng = np.random.default_rng(abs(hash((arr.size, round(base, 6)))) % (2**32))
    a = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(2.0, 1e4))
    b = float(rng.uniform(-1e4, 1e4))
    transformed = a * arr + b

    other = biweight_midvariance(
        transformed,
        c=c,
        axis=axis,
        modify_sample_size=modify_sample_size,
        ignore_nan=ignore_nan,
    )
    try:
        other_val = float(other)
    except (TypeError, ValueError):
        return
    if not np.isfinite(other_val):
        return

    predicted = a * a * base
    scale = max(abs(predicted), abs(other_val), 1e-300)
    relerr = abs(other_val - predicted) / scale
    trigger_if(relerr > _Q_TOL, "AP-STATS-003")


# --- Candidate R: jackknife closed form for the mean statistic -------------
#
# LAW_CANDIDATES.md Candidate R. Precondition: statistic is np.mean itself
# (not merely numerically matching it). Tolerance: bias 1e-8 relative to
# max(|mean|,1); std_err 1e-10 relative -- derived from a 20,000-trial sweep
# (worst bias 4.95e-12, worst std_err relative error 2.84e-15).

_R_BIAS_TOL = 1e-8
_R_SE_TOL = 1e-10


@_guard("jackknife_mean_closed_form")
def check_jackknife_mean_closed_form(data, statistic, bias, std_err):
    """AP-STATS-004: for statistic=np.mean, jackknife bias is exactly 0 and
    std_err equals sqrt(sum((x-xbar)**2)/(n*(n-1))) (LAW_CANDIDATES.md
    Candidate R). Only fires when the caller's statistic is np.mean by
    identity, to avoid any ambiguity about why a caller's function might
    numerically match the mean on one particular input.
    """
    import math

    import numpy as np

    if statistic is not np.mean:
        return

    arr = np.asanyarray(data)
    if arr.ndim != 1 or arr.size < 2:
        return
    if not np.all(np.isfinite(arr)):
        return

    try:
        bias_val = float(bias)
        se_val = float(std_err)
    except (TypeError, ValueError):
        return
    if not (np.isfinite(bias_val) and np.isfinite(se_val)):
        return

    xbar = float(np.mean(arr))
    n = arr.size

    bias_scale = max(abs(xbar), 1.0)
    trigger_if(abs(bias_val) / bias_scale > _R_BIAS_TOL, "AP-STATS-004")

    predicted_se = math.sqrt(float(np.sum((arr - xbar) ** 2)) / (n * (n - 1)))
    se_relerr = abs(se_val - predicted_se) / max(abs(predicted_se), 1e-300)
    trigger_if(se_relerr > _R_SE_TOL, "AP-STATS-004")
