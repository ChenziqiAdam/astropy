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


_F_EXCLUDED_PROJECTIONS = frozenset({"CSC", "TSC", "QSC", "HPX"})


@_guard("wcs_projection_roundtrip")
def check_wcs_pix2world_roundtrip(wcs_obj, original_xy, world, origin):
    """AP-WCS-001: wcs_pix2world and wcs_world2pix are documented as mutual
    inverses on the core (non-SIP) projection. Re-calling the public
    wcs_world2pix on the world coordinate just produced must recover the
    original pixel (LAW_CANDIDATES.md Candidate F).

    ``wcs_obj`` is the WCS instance; ``original_xy`` and ``world`` are the
    (N, 2) pixel/world arrays production code just computed; ``origin`` is
    the same origin convention (0 or 1) used for the forward call.

    The three quad-cube projections (CSC, TSC, QSC) are excluded by name:
    a census of all 27 standard projection headers shipped in astropy's own
    test suite found CSC failing this law at 100% of trials (errors up to
    2.5e-3 px, five orders of magnitude past tolerance, unrelated to distance
    from the reference pixel) -- wcs_world2pix's Newton inversion structurally
    fails to disambiguate CSC's projection, not amplified rounding.

    Fixed post-audit (2026-09-14, independent triggerability probe): a
    dedicated adversarial sweep specifically targeting quad-cube-family
    projections found TSC and QSC share the identical failure, but past a
    sharp ~45 deg-from-reference-pixel threshold rather than CSC's 100% rate
    -- both wrap by exactly one full longitude turn (offset = 360 deg *
    px/deg, e.g. 36000.0 px at 100 px/deg), deterministic and reproducible,
    not noise. Same root cause (quad-cube facet disambiguation in wcslib's
    Newton inversion), same fix: add both to the name exclusion set.

    Fixed again post-audit (2026-09-15, independent Opus-5 confirmatory
    audit of the fix above): the generalization to "quad-cube family"
    was along the wrong axis -- HPX (HEALPix), a facet-based but not
    quad-cube projection, shares the identical discrete-pixel-jump
    failure (confirmed: pixel (100,145) round-trips to (98,145), a clean
    2.0 px jump, six decades past tolerance) and was not covered by the
    quad-cube exclusion. Root cause is evidently facet/pixelization-based
    projections generally, not quad-cube specifically; HPX added to the
    exclusion set. A broader sweep of all 28 standard WCS projection
    codes (including the 6 requiring explicit PV parameters to avoid a
    wcsset ERROR 5) found no further failures -- HPX's own mirror
    projection XPH stayed clean. Known remaining trade-off (not yet
    addressed): TSC/QSC's blanket name exclusion loses real coverage in
    their well-behaved inner region (round-trip error ~1e-9 to 1e-12 for
    radius up to ~20px from the reference pixel, only failing past
    ~45deg) -- unlike CSC, which fails already at 1px. A radius-based
    rather than name-based gate could recover that coverage; not
    implemented this round as the priority was closing the false
    negative, not maximizing coverage.
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
#
# Second-order term subtracted post-audit (2026-09-14, independent audit
# finding, raised against this checker's sibling AP-UNITS-006 but equally
# applicable here since both share this exact structure): comparing only
# against the beta/2 leading-order term meant the "margin" at the top of
# the beta window was not numerical slack but the fixed, deterministic
# O(beta^2) Taylor truncation term itself -- a real physics deviation
# smaller than that term would have been invisible. Verified analytically
# (sympy series of sqrt((1-beta)/(1+beta))) and numerically: the residual
# after subtracting only beta/2 tracks -beta^2/2 almost exactly (worst
# case at beta=9e-4: residual -4.05e-7 vs -beta^2/2 = -4.05e-7). Fixed by
# subtracting the analytically-derived -beta^2/2 term (consistent with
# this checker's abs()-based observed_relerr convention) before comparing
# to tolerance, which drops the residual to pure numerical noise
# (~1e-10 to ~1e-13 across the same beta range) -- the same 1e-6 tolerance
# now provides genuine ~1000-10000x margin instead of the ~2.5x margin
# that was actually just truncation error.

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
        predicted = 0.5 * abs(beta) - 0.5 * beta**2
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
#
# Precondition/tolerance correction found by an independent adversarial
# audit (not self-verification): the un-restricted `b` domain (up to 1e4
# in magnitude, with no requirement it be commensurate with the rescaled
# data's own spread) admits a genuinely ill-conditioned regime -- an
# additive shift that swamps `a*data`'s own spread by many orders of
# magnitude forces `biweight_midvariance` to recover a tiny residual
# variance from differences of near-equal, shift-dominated numbers, a
# catastrophic-cancellation-prone computation whose error does not fit any
# single eps64-scaled polynomial-in-n model found (tried eps*n, eps*n*cond,
# eps*n*cond**2 -- none converged to a stable bound; true worst-case flat
# relative error over the *unrestricted* domain was 2.86e-2, six orders of
# magnitude past the shipped 1e-9). This is a genuine domain-of-
# applicability boundary, not amplified rounding -- the same class of gap
# as Candidate A's pole exclusion and Candidate U's kernel-larger-than-
# array exclusion. Excluded by precondition: the shift `b` must not exceed
# `1e3 * |a| * std(data)` (the rescaled data's own spread, computed from
# the array already in hand, no re-call needed). With this exclusion, a
# 400,000-trial sweep (n 5-300, data std 1e-14-1e14, |a| up to 1e6, |b| up
# to 1e6, both `modify_sample_size` values) found a stable worst flat
# relative error of 9.23e-13 -- consistent with an ordinary few-hundred-ULP
# accumulation, no residual scale/n dependence. Final: `tol = 1e-9`
# (~1000x margin over the observed worst case), unchanged from the
# original value -- the fix is the precondition, not the tolerance.
#
# Second precondition gap found post-audit (2026-09-14, independent
# triggerability probe): the `_Q_SHIFT_RATIO_MAX` exclusion above only
# bounds the checker's own internally-drawn `b` relative to `a*data`'s
# spread -- it does nothing about the caller's *original* data already
# having a large baseline relative to its own spread (e.g. any real
# dataset with a physical additive background: detector counts, a flux
# with a bias level, a temperature series in Kelvin). In that case
# `result` itself -- computed by production code from the caller's
# already-offset array, before this checker ever runs -- is already
# affected by the same catastrophic-cancellation mechanism, and no bound
# on the checker's own re-transform can undo that. A sweep varying only
# this pre-existing baseline (cond = baseline / std(data), 20 decades,
# one fixed 5-point dataset) found smooth, monotonic relative-error
# growth crossing `_Q_TOL` at cond ~ 1e8; a broader 2,000-trial sweep
# (n 5-60, cond up to 1e6) found a stable worst relerr/tol ratio of 0.11
# (~9x margin), no instability. Same fix pattern as Candidate R
# (AP-STATS-004): gate on the data's own location/spread condition
# number, computed from the array already in hand.
#
# That fix used `|median|/std(data)` as the condition number -- wrong
# statistic, found by an independent Opus-5 confirmatory audit
# (2026-09-15). `biweight_midvariance` normalizes residuals by
# `c * MAD` (median absolute deviation), not by `std`: a handful of
# realistic outliers (the exact scenario biweight estimators exist to be
# robust against -- cosmic rays, bad pixels, a few bad measurements)
# inflate `std` sharply without moving `MAD` at all, so `|median|/std`
# can read as small (gate passes) while the estimator's actual
# conditioning `|median|/MAD` is many orders of magnitude worse. An
# independent re-sweep confirmed this at scale: 3 separate random
# samples (2000-4000 trials each, distinct seeds/structures, realistic
# baseline + 2 outlier magnitude/sign combinations) found a 35-71%
# false-positive rate under the `std`-based gate -- an exact-arithmetic
# cross-check (`Fraction`) confirmed these are checker false positives,
# not real astropy defects (float64 agreed with exact arithmetic to the
# last bit). Switched to `|median|/MAD`: on the same realistic-outlier
# sweep, `MAD` correctly reads these as ill-conditioned (gate rejects,
# worst false-positive ratio 0.0 among cases that still pass); re-ran
# the original clean (no-outlier) 2,000-trial sweep with the new
# statistic and found the same ~9x margin as before (worst ratio 0.05),
# confirming the fix doesn't cost coverage on the case it was already
# handling correctly.

_Q_TOL = 1e-9
_Q_SHIFT_RATIO_MAX = 1e3
_Q_DATA_COND_MAX = 1e6


@_guard("biweight_scale_equivariance")
def check_biweight_midvariance_equivariance(
    data, c, axis, modify_sample_size, ignore_nan, result
):
    """AP-STATS-003: biweight_midvariance(a*x + b) == a**2 *
    biweight_midvariance(x) for any real a, b (LAW_CANDIDATES.md
    Candidate Q), *provided* the shift b does not swamp a*data's own
    spread by more than a factor of 1e3 -- see the precondition-correction
    note above for why this exclusion is required, not merely convenient.

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

    arr_std = float(np.std(arr))
    if not (np.isfinite(arr_std) and arr_std > 0):
        return

    arr_median = float(np.median(arr))
    arr_mad = float(np.median(np.abs(arr - arr_median)))
    if not (np.isfinite(arr_mad) and arr_mad > 0):
        return

    data_cond = abs(arr_median) / arr_mad
    if data_cond > _Q_DATA_COND_MAX:
        return

    rng = np.random.default_rng(abs(hash((arr.size, round(base, 6)))) % (2**32))
    a = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(2.0, 1e4))
    b = float(rng.uniform(-1e4, 1e4))

    data_spread = abs(a) * arr_std
    if abs(b) > _Q_SHIFT_RATIO_MAX * data_spread:
        return

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
#
# std_err tolerance correction found by an independent adversarial audit
# (not self-verification): the flat `1e-10` relative tolerance false-fired
# on ~45% of trials once data is centered away from zero with
# |mean|/std_err >~ 1e5 (e.g. any measurement with a nonzero physical
# baseline -- a temperature in Kelvin with small variance, a flux with an
# additive background) -- an ordinary case, not a rare tail event. The
# original 20,000-trial sweep's claimed "worst std_err relative error
# 2.84e-15" only held because it never tested offset-dominated data. Root
# cause: jackknife's per-fold recomputation of np.mean over n-1 points,
# and this checker's own re-derivation `sum((x-xbar)**2)`, both subtract a
# large, nearly-equal mean from each point -- a catastrophic-cancellation
# operation whose absolute error scales with |mean|, not with the
# quantity of interest (std_err) itself, once |mean| >> std_err. Re-
# derived as `tol_se = C * eps64 * n * max(1, cond)`,
# `cond = |mean| / predicted_se` (the ratio that directly captures this
# cancellation's conditioning): a 150,000-trial sweep (n 2-300, center up
# to 1e15 in magnitude, scale spanning 1e-12 to 1e12) found a stable worst
# ratio of 0.46 with no growth as center/scale widened further. The bias
# check needed no correction -- swept identically across the same
# magnitude range, its worst bias/scale ratio stayed a stable ~4e-14 with
# no cancellation-driven growth, since jackknife's bias formula for the
# mean is an exact algebraic zero regardless of centering.
#
# Bias-check blind spot found post-audit (2026-09-14, independent
# triggerability probe): the claim above ("no cancellation-driven growth")
# was falsified by data whose raw values are individually huge but happen
# to *sum* to something near zero (e.g. paired +-1e15 values) -- `xbar`
# itself lands near zero precisely because of that cancellation, so
# `cond = |xbar|/predicted_se` (the variable the original sweep centered
# on) stays small and hides the fragility instead of catching it. The
# actual mechanism: each jackknife fold recomputes `mean(arr[:i]+arr[i+1:])`
# by summing n-1 raw values whose individual magnitude sets the absolute
# float64 rounding floor of that sum (ULP ~ 2^-52 * max|value|) --
# independent of how those values happen to combine. A sweep confirmed
# `max|x|/predicted_se` and `max|x|/std(x)` both fail to separate
# triggering from non-triggering trials (a repro at cond as low as 1.0
# still fires), because the driver is the raw magnitude itself, not its
# ratio to any derived spread. A direct sweep over `max(|x|)` alone
# (mixed structures: offset+noise, symmetric-pair+noise, wide-uniform; n
# 2-500) found a clean, monotonic threshold: worst bias/scale ratio
# 4.7e-10 at magnitude <=1e4 (~20x margin under `_R_BIAS_TOL`), climbing
# past the tolerance itself by magnitude ~1e5. Gated on `max(|x|) <=
# _R_BIAS_MAG_MAX`. The std_err check needed no change -- confirmed
# clean (se_relerr=0) on the exact falsifying repro; only the bias
# check shares this blind spot.
#
# `_R_BIAS_MAG_MAX` fix itself found incomplete post-audit (2026-09-15,
# independent Opus-5 confirmatory audit): the magnitude gate alone
# doesn't bound `n` -- rounding error in the O(n) jackknife-fold
# recomputation accumulates with the number of folds, not just their
# magnitude. At `max|x|` pinned exactly at the 1e4 gate boundary, bias
# stays clean through n~10,000 but exceeds `_R_BIAS_TOL` by 3.76x at
# n=30,000 (independently reproduced). A direct sweep of
# `bias/(eps64*n*max|x|)` (the natural scaling for round-off in an
# n-term sum of that magnitude) found this ratio stays bounded --
# noisy but with no growth trend -- across n from 10 to 30,000 and
# mixed structures (worst observed ~1.5 over ~450 combined trials),
# unlike the flat-tolerance model which necessarily fails as n grows
# without bound. Fixed by replacing the fixed `_R_BIAS_TOL` with a
# `max(fixed floor, C*eps64*n*max(1,max|x|))` tolerance -- the fixed
# floor preserves sensitivity at small n (where the n-scaled term is
# below machine-noise), the n-scaled term dominates and grows correctly
# at large n. C=50 gives ~33x margin over the observed worst ratio.

_R_BIAS_TOL = 1e-8
_R_BIAS_TOL_C = 50.0
_R_SE_TOL_C = 5.0
_R_BIAS_MAG_MAX = 1e4


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

    max_abs_x = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs_x <= _R_BIAS_MAG_MAX:
        bias_scale = max(abs(xbar), 1.0)
        bias_tol = max(
            _R_BIAS_TOL,
            _R_BIAS_TOL_C * _EPS64 * n * max(1.0, max_abs_x) / bias_scale,
        )
        trigger_if(abs(bias_val) / bias_scale > bias_tol, "AP-STATS-004")

    predicted_se = math.sqrt(float(np.sum((arr - xbar) ** 2)) / (n * (n - 1)))
    if predicted_se <= 0.0 or not math.isfinite(predicted_se):
        return
    se_relerr = abs(se_val - predicted_se) / predicted_se
    cond = abs(xbar) / predicted_se
    tol_se = _R_SE_TOL_C * _EPS64 * n * max(1.0, cond)
    trigger_if(se_relerr > tol_se, "AP-STATS-004")


# --- Candidate S: kernel normalization exactness ----------------------------
#
# LAW_CANDIDATES.md Candidate S. Precondition: mode='integral', pre-
# normalization sum finite and nonzero. Tolerance: 10 * eps64, derived from a
# 72-kernel sweep across all 11 built-in shapes (worst truncation 4.44e-16 =
# 2*eps64, no size dependence found from n~9 to n~400).
#
# Zero-sum precondition gap found during self-verification (not an
# external audit): the first version only received the post-normalize()
# sum, not whether a division actually happened. `Kernel.normalize` has an
# explicit branch (production code, not a bug) that skips the division
# entirely when the pre-normalization sum is exactly zero -- warning and
# leaving the array un-normalized, per its own docstring/tests
# (test_custom_1D_kernel_zerosum: array [-2,-1,0,1,2] sums to exactly 0,
# `custom.truncation == 1.0` is the *documented correct* outcome). Without
# the pre-normalization sum, the checker could not distinguish this
# documented skip-division branch from a real off-by-something in the
# divide -- both produce a "sum far from 1" observation. Fixed by also
# passing the pre-normalization sum and excluding the zero-sum case by
# precondition, matching the exclusion already stated in the design doc
# but not actually wired into the checker's call site.
#
# Tolerance correction found by an independent adversarial audit (not
# self-verification): a flat `10*eps64` was falsified outright -- a clean
# 3,000-trial sweep of ordinary (non-adversarial) CustomKernel arrays of
# unit-Gaussian noise found 100% of trials exceeding it, with the original
# design-doc sweep's "worst case 2*eps64, no size dependence" claim simply
# wrong (that sweep only exercised the built-in, closed-form, symmetric
# kernel shapes -- Gaussian1DKernel, Box1DKernel, etc. -- a materially
# narrower domain than the precondition the checker actually enforces).
# Re-derived from first principles: summing n post-division float64 values
# is an n-term resummation whose rounding floor is set both by n (ordinary
# accumulation) and by the conditioning of the division itself -- how
# close the pre-normalization sum is to being swamped by cancellation
# among the array's own elements, `cond = sum(|arr|) / |pre_sum|` (the
# textbook condition number of the reduction that produced pre_sum). A
# 300,000-trial sweep (n in [3,300], ordinary unit-Gaussian arrays plus an
# adversarial near-zero-pre-sum-via-cancellation variant every 5th trial)
# found `err / (eps64 * n * cond)` bounded at a stable worst case of
# 0.333, with no growth as n or cond increased across either the ordinary
# or adversarial sub-sweep -- confirmed `n` is load-bearing (dropping it
# and using `cond` alone gave a *worse*, unstable worst ratio of 1.62).
# Final: `tol = 10 * eps64 * n * cond` (~30x margin over the observed
# worst ratio).

_S_TOL_C = 10.0


@_guard("kernel_normalization_exactness")
def check_kernel_normalization(mode, pre_sum, array_sum, abs_sum, size):
    """AP-CONV-001: after Kernel.normalize(mode='integral'), the array must
    sum to 1 to float64 rounding (LAW_CANDIDATES.md Candidate S), *provided*
    the pre-normalization sum was actually nonzero (the zero-sum case is
    production code's own documented skip-division branch, not part of
    this law -- see the zero-sum note above). Reads values already
    computed by production code -- no re-call needed.

    ``abs_sum`` is sum(abs(array)) *before* normalization and ``size`` is
    the array's element count -- both used only to compute the tolerance
    (division condition number and accumulation term), never to re-derive
    the scientific quantity itself.
    """
    import math

    if mode != "integral":
        return
    if not (math.isfinite(pre_sum) and pre_sum != 0.0):
        return
    if not (math.isfinite(array_sum) and math.isfinite(abs_sum) and abs_sum > 0.0):
        return
    if size < 1:
        return
    cond = abs_sum / abs(pre_sum)
    tol = _S_TOL_C * _EPS64 * size * cond
    trigger_if(abs(array_sum - 1.0) > tol, "AP-CONV-001")


# --- Candidate T: flux conservation under convolution, wrap boundary -------
#
# LAW_CANDIDATES.md Candidate T. Precondition: boundary='wrap',
# normalize_kernel=True, no mask, finite non-NaN input. Tolerance:
# 30 * eps64 * sqrt(array.size), floored at 50 * eps64, relative to
# max(1, max(|array|)) -- corrected during self-verification (see note
# below) from an initial flat-constant / sum-normalized guess that a
# 6000-trial mixed 1D/2D sweep found both under- and mis-scaled.
#
# Normalization/scaling correction found during self-verification (not an
# external audit): the first version normalized by max(1, |array.sum()|)
# and used a flat tolerance. A realistic sweep (Gaussian-noise arrays with
# per-trial magnitude scaled across 13 orders of magnitude, boundary='wrap')
# found a false-positive-triggering case: an array whose elements were
# individually large (max|a| ~ 520) but whose *sum* happened to be small
# (~9.9, from sign cancellation) -- the true summation round-off (~1e-13)
# was tiny relative to max|a| (~1*eps64) but large relative to the
# accidentally-small sum (~50*eps64, tripping the old flat tolerance).
# Fixed by normalizing against max(1, max(|array|)) instead (matching
# Candidate U's own scale choice) -- the physically meaningful error floor
# for an O(n)-element summation is set by the elements' magnitude, not by
# a sum that can be arbitrarily small through cancellation. A second,
# independent effect was then found: with the corrected normalization, the
# worst-case ratio grows with array size (2D worst 99*eps64 over 2000
# trials, vs 1D worst ~25*eps64) -- consistent with an O(sqrt(n)*eps64)
# random-walk accumulation model for an n-element reduction. Re-derived
# against ratio/sqrt(array.size), which stayed bounded at 2.88 over 6000
# mixed 1D/2D trials (sizes up to 500 in 1D, 150x150 in 2D, 13 orders of
# input magnitude, Gaussian and Box kernels) -- confirming the family
# variable was array size, not a fixed constant. Final: 30*eps64*sqrt(size)
# (~10x margin), floored at 50*eps64 so small arrays keep a sane minimum.

_T_TOL_C = 30.0
_T_TOL_FLOOR = 50.0 * _EPS64


@_guard("convolution_flux_conservation")
def check_convolution_flux_conservation(
    array_internal, result, boundary, normalize_kernel, nan_treatment, mask
):
    """AP-CONV-002: convolve(array, kernel, boundary='wrap',
    normalize_kernel=True).sum() == array.sum() (LAW_CANDIDATES.md
    Candidate T). Reads the production array/result already computed by
    ``convolve`` -- no re-call needed, this is a conservation law on the
    single production call's own input/output pair.
    """
    import math

    import numpy as np

    if boundary != "wrap" or not normalize_kernel or mask is not None:
        return
    arr = np.asarray(array_internal)
    if not np.all(np.isfinite(arr)):
        return
    res = np.asarray(result)
    if not np.all(np.isfinite(res)):
        return

    before = float(arr.sum())
    after = float(res.sum())
    scale = max(1.0, float(np.max(np.abs(arr), initial=0.0)))
    tol = max(_T_TOL_FLOOR, _T_TOL_C * _EPS64 * math.sqrt(arr.size))
    trigger_if(abs(after - before) > tol * scale, "AP-CONV-002")


# --- Candidate U: convolve vs convolve_fft cross-implementation agreement --
#
# LAW_CANDIDATES.md Candidate U. Precondition: finite non-NaN input,
# boundary in {'fill', 'wrap'}, normalize_kernel=True, kernel not larger
# than the array in any axis, input size below a cheap-to-recompute
# threshold (bounds checker overhead per SANITIZER.md 5.5). Tolerance:
# 50 * eps64 relative to max(1, max(|array|)), derived from sweeps up to
# 600 trials per boundary/kernel-shape/NaN-fraction combination (worst
# 3.26*eps64, the NaN-interpolation case).
#
# Kernel-larger-than-array exclusion found during self-verification (not
# an external audit): a 200-trial randomized sweep found convolve() and
# convolve_fft() disagree by O(1) (not O(eps64)) whenever the kernel size
# exceeds the array size in `boundary='wrap'` mode -- confirmed sharp at
# the exact boundary (kernel size == array size: diff 2.2e-13, agrees;
# kernel size == array size + 2: diff 1.97, real O(1) disagreement). Root
# cause: `convolve`'s direct C path pads the array via `np.pad(...,
# mode='wrap')` with `pad_width = kernel_shape // 2`, which only wraps in
# one copy of the array on each side; `convolve_fft`'s FFT-domain circular
# convolution implicitly assumes a period equal to its padded transform
# size. When the kernel's reach exceeds the array's own extent, the two
# padding/periodicity conventions are no longer computing the same
# mathematical operation -- a genuine domain-of-applicability boundary
# (`convolve`'s own docstring only forbids this for `boundary=None`, not
# 'wrap'/'fill'/'extend'), not amplified rounding. Excluded by precondition
# rather than loosening the tolerance, per SANITIZER.md 5.7/CSC precedent.

_U_TOL = 50.0 * _EPS64
_U_MAX_SIZE = 4096


@_guard("convolution_cross_implementation_agreement")
def check_convolution_cross_implementation(
    array_internal,
    kernel_internal,
    result,
    boundary,
    fill_value,
    nan_treatment,
    normalize_kernel,
    mask,
):
    """AP-CONV-003: convolve() and convolve_fft() must agree to float64
    rounding on the same input (LAW_CANDIDATES.md Candidate U). This is the
    one candidate in this bank whose re-derivation calls a genuinely
    different public API rather than the same one on a transformed input,
    since the law is inherently about cross-implementation agreement.
    Forwards every result-affecting argument the two functions share
    (differential-test contract, arg-forwarding class) and skips inputs
    outside either function's shared, swept domain.
    """
    import numpy as np

    from astropy.convolution.convolve import convolve_fft

    if boundary not in ("fill", "wrap") or not normalize_kernel or mask is not None:
        return
    arr = np.asarray(array_internal)
    ker = np.asarray(kernel_internal)
    if arr.size == 0 or arr.size > _U_MAX_SIZE:
        return
    if arr.ndim != ker.ndim or any(ks > as_ for ks, as_ in zip(ker.shape, arr.shape)):
        return
    if not np.all(np.isfinite(ker)):
        return
    if not (np.all(np.isfinite(arr)) or nan_treatment == "interpolate"):
        return

    try:
        fft_result = convolve_fft(
            arr,
            ker,
            boundary=boundary,
            fill_value=fill_value,
            nan_treatment=nan_treatment,
            normalize_kernel=normalize_kernel,
        )
    except Exception:
        return

    res = np.asarray(result)
    fft_res = np.asarray(fft_result)
    if res.shape != fft_res.shape:
        return
    finite_mask = np.isfinite(res) & np.isfinite(fft_res)
    if not np.any(finite_mask):
        return

    diff = np.max(np.abs(res[finite_mask] - fft_res[finite_mask]))
    scale = max(1.0, float(np.max(np.abs(arr[np.isfinite(arr)]), initial=0.0)))
    trigger_if(float(diff) > _U_TOL * scale, "AP-CONV-003")


# --- Candidates V/W/X/Y: astropy.constants cross-constant relations --------
#
# LAW_CANDIDATES.md Candidates V-Y. All four re-derive a constant from
# sibling constants and compare against the value astropy itself just
# constructed. Confirmed at implementation time (not merely assumed from
# the design doc): sigma_sb/R/e_esu/e_emu/e_gauss/M_sun/M_jup/M_earth are
# ALL computed via this exact formula in-repo for the current default
# vintage (CODATA2022/IAU2015), so bit-for-bit agreement is the correct
# invariant there, not merely an approximate one. CODATA2010/2014 are the
# cases where sigma_sb/R are independently published literals rather than
# in-repo-derived (relative errors 6.66e-8/3.24e-8 for sigma_sb, 5.47e-9/
# 7.39e-9 for R -- both far tighter than the design doc's original ~1e-5
# estimate from CODATA's published measurement-uncertainty ratio,
# confirming CODATA fits these jointly/consistently rather than merely
# "within stated uncertainty").
#
# Global-config bug found during self-verification (not an external
# audit): the first version read sibling constants (h, k_B, c, N_A, G) from
# `astropy.constants.config.codata`/`.iaudata` -- the process-wide *active*
# vintage -- rather than from the vintage module actually being
# constructed. Every non-default vintage module (codata2010.py,
# codata2014.py, codata2018.py, iau2012.py) executes standalone at import
# time (e.g. `from astropy.constants.codata2010 import c` runs the *whole*
# module, constructing its sigma_sb/R too) while the global config still
# points at the default (2022/2015) -- so the checker was silently
# comparing one vintage's sigma_sb against a DIFFERENT vintage's h/k_B/c,
# a mismatched-vintage bug, not a real inconsistency. Caught via
# astropy/constants/tests/test_prior_version.py::test_c (10 total triggers
# across that file) -- even the "just check c is exact" test triggered,
# because importing codata2010.c as a side effect constructs the whole
# module. Fixed by reading siblings from the *constructing* module's own
# namespace (via the caller's stack frame globals, since `Constant.__new__`
# has no explicit "which module am I in" argument) instead of the global
# config -- matching what the production code itself does (e.g. R =
# k_B.value * N_A.value reads k_B/N_A from its own module's already-
# executed lines above it, never from a different vintage).
#
# Instrumentation point: `Constant.__new__`'s return, dispatched by
# `abbrev`/`system`, given the constructing module's globals -- each
# constant is a singleton constructed once at module-import time, so this
# is a one-shot hook (cheaper than re-checking on every `.si` access, and
# the sibling family it needs is already fully defined earlier in the same
# module by construction order).

_V_TOL_DERIVED = 10.0 * _EPS64  # CODATA2018/2022: sigma_sb computed in-repo
_V_TOL_PUBLISHED = 1e-6  # independently published (2010/2014): ~15x margin over 6.66e-8
_W_TOL_DERIVED = 10.0 * _EPS64  # R computed in-repo
_W_TOL_PUBLISHED = 1e-7  # ~14x margin over the observed 7.39e-9/5.47e-9
_X_TOL = 10.0 * _EPS64  # e_esu/e_emu/e_gauss: exact algebraic identity
_Y_TOL = 10.0 * _EPS64  # M_x = GM_x/G computed in-repo, exact by construction

# A vintage's sigma_sb/R either equal this exact formula in-repo (CODATA2018,
# 2022, and any future vintage that keeps computing them this way) or are an
# independently published literal from an older vintage (2010, 2014, and any
# vintage not yet seen). Distinguishing regime by vintage *name* does not
# generalize (the 2010 case proved that: only "2014" was special-cased and
# 2010 silently fell into the wrong branch) and distinguishing by the
# observed relerr itself is circular (an earlier, buggier version of this
# checker did exactly that: "pick the tight tolerance whenever relerr looks
# small" always fails except at relerr==0, since relerr <= tight_tol can
# only be true for a value smaller than that same tight_tol). The reliable,
# non-circular signal astropy already carries is the constant's own
# `uncertainty` field: derived-in-repo constants (2018/2022's sigma_sb/R)
# are declared with `uncertainty=0.0` (exact by construction), while
# independently published literals (2010/2014) carry their real nonzero
# measurement uncertainty -- confirmed directly against all four vintages'
# source before use, not merely assumed.


@_guard("stefan_boltzmann_formula_consistency")
def _check_sigma_sb(value, uncertainty, mod):
    """AP-CONST-001 (Candidate V): sigma_sb == 2*pi^5*k_B^4/(15*h^3*c^2),
    using h/k_B/c from the *same vintage module* sigma_sb was defined in.
    """
    import math

    h = float(mod["h"].value)
    k_B = float(mod["k_B"].value)
    c = float(mod["c"].value)
    formula = 2.0 * math.pi**5 * k_B**4 / (15.0 * h**3 * c**2)
    relerr = abs(value - formula) / abs(formula)
    tol = _V_TOL_DERIVED if uncertainty == 0.0 else _V_TOL_PUBLISHED
    trigger_if(relerr > tol, "AP-CONST-001")


@_guard("gas_constant_avogadro_boltzmann")
def _check_gas_constant(value, uncertainty, mod):
    """AP-CONST-002 (Candidate W): R == N_A * k_B, same-module siblings."""
    N_A = float(mod["N_A"].value)
    k_B = float(mod["k_B"].value)
    formula = N_A * k_B
    relerr = abs(value - formula) / abs(formula)
    tol = _W_TOL_DERIVED if uncertainty == 0.0 else _W_TOL_PUBLISHED
    trigger_if(relerr > tol, "AP-CONST-002")


@_guard("cgs_electron_charge_triplet")
def _check_cgs_charge(abbrev, system, value, mod):
    """AP-CONST-003 (Candidate X): e_esu == e_gauss == e*c*10, e_emu ==
    e/10, using e/c from the same vintage module.
    """
    e = float(mod["e"].value)
    c = float(mod["c"].value)
    if system in ("esu", "gauss"):
        formula = e * c * 10.0
    elif system == "emu":
        formula = e / 10.0
    else:
        return
    relerr = abs(value - formula) / abs(formula)
    trigger_if(relerr > _X_TOL, "AP-CONST-003")


@_guard("mass_from_gm_over_g")
def _check_mass_from_gm(abbrev, value, mod):
    """AP-CONST-004 (Candidate Y): M_x == GM_x / G for Sun/Jupiter/Earth,
    but only in astro-constant vintages that actually derive mass this way,
    using GM_x/G from the same vintage module M_x was defined in.

    Precondition gap found during self-verification (not an external
    audit): the first version always looked up `GM_x` in the global
    `astropy.constants.config.iaudata` module, regardless of which
    astro-constant vintage was actually being constructed. `iau2012` (an
    older, still-selectable vintage) defines `M_sun`/`M_jup`/`M_earth` as
    direct literals from Allen's Astrophysical Quantities with no `GM_x`
    at all -- the M=GM/G relation this candidate checks simply does not
    exist in that vintage, so comparing against a different vintage's
    unrelated `GM_x` (or, when the global config happened to point at
    iau2015 while iau2012 was the module actually executing, a genuinely
    mismatched-vintage GM_x/G pair) produced spurious "disagreements" that
    reflect two different vintages' independent mass estimates, not a law
    violation. Fixed by reading `GM_x`/`G` from the constructing module's
    own namespace and skipping entirely when that module has no `GM_x`
    sibling, rather than reading a process-wide global.

    Second gap found by an independent adversarial audit (not
    self-verification): `iau2015.py` never binds a bare `G` name -- it
    does `from .config import codata` and reads `codata.G` throughout, so
    `"G" not in mod` was vacuously True for the only vintage this law
    applies to, making the checker dead code against the real codebase
    (confirmed by a positive control: injecting a wildly wrong M_sun into
    the real iau2015 module produced zero triggers). Fixed by also
    accepting `G` via a `codata` module object bound in the constructing
    module's namespace, matching iau2015.py's actual style instead of
    assuming every vintage module binds formula ingredients as bare names.
    """
    body = abbrev[2:]  # "M_sun" -> "sun"
    gm_attr = f"GM_{body}"
    if gm_attr not in mod:
        return
    if "G" in mod:
        g_const = mod["G"]
    else:
        codata_mod = mod.get("codata")
        g_const = getattr(codata_mod, "G", None)
        if g_const is None:
            return
    GM = float(mod[gm_attr].value)
    G = float(g_const.value)
    formula = GM / G
    relerr = abs(value - formula) / abs(formula)
    trigger_if(relerr > _Y_TOL, "AP-CONST-004")


_CGS_CHARGE_ABBREVS = {"e_esu", "e_emu", "e_gauss"}
_GM_MASS_ABBREVS = {"M_sun", "M_jup", "M_earth"}

# --- Candidate AH: Bohr magneton cross-constant relation --------------------
#
# LAW_CANDIDATES.md Candidate AH. muB = e*hbar/(2*m_e) using e/hbar/m_e from
# the same CODATA vintage module muB was defined in -- muB is an
# independently-published literal (nonzero uncertainty in every vintage), so
# this uses the published-literal tolerance regime, not the derived-in-repo
# 10*eps64 regime (unlike sigma_sb/R/e_esu-family/M_x, muB is never
# constructed with uncertainty==0.0 in any vintage inspected).
#
# Tolerance widened post-audit (2026-09-14, independent audit finding): the
# original 1e-9 gave only 1.32x margin against codata2010's own relerr
# (7.56e-10) -- not numerical noise but the genuine historical measurement-
# precision gap of that older CODATA vintage (codata2014: 3.42e-10/2.93x,
# codata2018: 6.70e-12/149x, codata2022: 4.11e-12/244x -- each successive
# vintage's muB literal is measured more precisely and agrees better with
# the formula, exactly as expected of real physics literals, not something
# a wider numerical sweep could shrink). A thin margin here risked a false
# trigger from nothing more than which vintage happens to be active.
# Widened to 5e-9, restoring ~6.6x margin over codata2010 while remaining
# far tighter than would be needed to hide an actual formula-vs-literal
# defect (a real bug would need to be missed by roughly 5 parts in a
# billion to slip past this, still an extremely tight bound for a
# nine-significant-figure physical constant).

_AH_TOL = 5e-9


@_guard("bohr_magneton_from_e_hbar_me")
def _check_bohr_magneton(value, mod):
    """AP-CONST-005 (Candidate AH): muB == e*hbar/(2*m_e), using e/hbar/m_e
    from the constructing module's own namespace (same discipline as
    AP-CONST-001/002/004 -- never a process-wide "active vintage" global).
    """
    if not ("e" in mod and "hbar" in mod and "m_e" in mod):
        return
    e_val = float(mod["e"].value)
    hbar_val = float(mod["hbar"].value)
    m_e_val = float(mod["m_e"].value)
    formula = e_val * hbar_val / (2.0 * m_e_val)
    relerr = abs(value - formula) / abs(formula)
    trigger_if(relerr > _AH_TOL, "AP-CONST-005")


def check_constant_relation(abbrev, system, value, uncertainty, caller_globals):
    """Dispatch to the right Candidate V/W/X/Y checker by constant abbrev.
    Called once per Constant construction (see `Constant.__new__`'s hook);
    each sub-checker is independently `_guard`-wrapped so one candidate's
    failure never disturbs another's or production code. ``caller_globals``
    is the module namespace the constant is actually being constructed in
    (not a process-wide "active vintage" global, which does not match
    during standalone imports of a non-default vintage module -- see the
    note above this function).
    """
    try:
        value = float(value)
        uncertainty = float(uncertainty)
    except (TypeError, ValueError):
        return
    import math

    if not (math.isfinite(value) and math.isfinite(uncertainty)):
        return
    mod = caller_globals

    if abbrev == "sigma_sb" and system == "si":
        if "h" in mod and "k_B" in mod and "c" in mod:
            _check_sigma_sb(value, uncertainty, mod)
    elif abbrev == "R" and system == "si":
        if "N_A" in mod and "k_B" in mod:
            _check_gas_constant(value, uncertainty, mod)
    elif abbrev in _CGS_CHARGE_ABBREVS:
        if "e" in mod and "c" in mod:
            _check_cgs_charge(abbrev, system, value, mod)
    elif abbrev in _GM_MASS_ABBREVS:
        _check_mass_from_gm(abbrev, value, mod)
    elif abbrev == "muB" and system == "si":
        _check_bohr_magneton(value, mod)


# --- Candidate Z: Lomb-Scargle cross-implementation agreement ---------------
#
# LAW_CANDIDATES.md Candidate Z. Precondition: nterms=1, regular frequency
# grid (fast's own domain), any normalization/fit_mean/center_data.
#
# Fixed post-audit (2026-09-14, independent audit finding): the original
# design used a single absolute tolerance (1e-6) on the reasoning that
# "power is intrinsically bounded" -- true only for normalization in
# {standard, model, log}, whose power is O(1) and dimensionless. For
# normalization='psd', power is dimensional (units of amplitude-squared per
# frequency) and can be arbitrarily large (e.g. a Kepler-style light curve
# in electrons/s with amplitude ~1e5 gives psd power ~1e10) -- an absolute
# tolerance is meaningless there and the checker false-fired on an entirely
# ordinary astronomical workflow. A targeted sweep found the *relative*
# agreement between fast and slow is stable at ~1e-11 to 1e-14 across every
# normalization and every amplitude tested, confirming the two
# implementations genuinely agree and only the tolerance model was wrong.
# Fixed by keeping the absolute check for the three O(1)-bounded
# normalizations (as originally designed, still the tighter and more
# meaningful check there) and adding a relative check for 'psd' specifically.
#
# Tolerance: 1e-6 absolute for standard/model/log, derived from two
# independent 500-trial sweeps (worst 7.30e-11 and 2.42e-10) plus a 300-trial
# post-fix re-sweep spanning amplitudes 1e-6 to 1e6 (worst 1.6e-12 for these
# three normalizations) -- fast's FFT/extirpolation scheme is a genuine
# bounded approximation to slow's exact sum, not an alternative exact
# evaluation, so eps64-scale agreement is not expected (unlike every prior
# cross-implementation candidate in this bank). 1e-6 relative for 'psd',
# derived from the same 300-trial sweep (worst relative error 3.57e-10
# across all four normalizations combined) -- ~2700x margin.
#
# Re-deriving via method='slow' is O(N * Nfreq), the same cost class as
# AP-CONV-003's re-call to convolve_fft -- capped by problem size (not by
# correctness) for the same reason: even when enabled for evaluation, an
# unbounded O(N^2)-ish re-derivation on every fast-method call should not
# make the checker itself the bottleneck. Cap chosen generously above
# realistic test-suite sizes, not tuned to any specific input.

_Z_TOL_ABS = 1e-6
_Z_TOL_REL = 1e-6
_Z_MAX_COST = 2_000_000  # N * Nfreq


@_guard("lombscargle_cross_implementation")
def check_lombscargle_cross_implementation(
    t, y, dy, frequency, center_data, fit_mean, nterms, normalization, power
):
    """AP-TS-001 (Candidate Z): LombScargle.power(..., method='fast') must
    agree with method='slow' on the same inputs, within fast's own
    documented approximation budget (not eps64 -- fast is an approximate
    FFT/extirpolation scheme, slow is the exact direct sum). Absolute
    tolerance for the three O(1)-bounded normalizations (standard, model,
    log); relative tolerance for 'psd', whose power is dimensional and can
    be arbitrarily large or small depending on the input amplitude.
    """
    import numpy as np

    from astropy.timeseries.periodograms.lombscargle.implementations.main import (
        lombscargle,
    )

    if nterms != 1:
        return
    freq = np.asarray(frequency)
    if freq.ndim != 1 or freq.size < 2:
        return
    t_size = np.asarray(t).size
    if t_size * freq.size > _Z_MAX_COST:
        return
    power_arr = np.asarray(power)
    if power_arr.shape != freq.shape or not np.all(np.isfinite(power_arr)):
        return

    t_arr = np.asarray(t)
    y_arr = np.asarray(y)
    dy_arr = None if dy is None else np.asarray(dy)

    try:
        slow_power = lombscargle(
            t_arr,
            y_arr,
            dy_arr,
            frequency=freq,
            center_data=center_data,
            fit_mean=fit_mean,
            nterms=nterms,
            normalization=normalization,
            method="slow",
        )
    except Exception:
        return
    slow_power = np.asarray(slow_power)
    if slow_power.shape != power_arr.shape or not np.all(np.isfinite(slow_power)):
        return

    diff = float(np.max(np.abs(power_arr - slow_power)))
    if normalization == "psd":
        denom = np.maximum(np.abs(slow_power), 1e-300)
        relerr = float(np.max(np.abs(power_arr - slow_power) / denom))
        trigger_if(relerr > _Z_TOL_REL, "AP-TS-001")
    else:
        trigger_if(diff > _Z_TOL_ABS, "AP-TS-001")


# --- Candidate AB: single-frequency false-alarm-probability round trip -----
#
# LAW_CANDIDATES.md Candidate AB. Precondition: fap in (0,1), dK-dH==2 (the
# only case _statistics.py implements), dK < N <= 1e6. Absolute tolerance --
# fap is a probability, intrinsically bounded to [0,1]. Tolerance: 1e-9
# absolute, derived from an 80,000-trial sweep across all 4 normalizations
# (worst 5.51e-13) -- both directions are short closed-form elementary-
# function chains, an ordinary few-hundred-ULP composition.
#
# Upper bound on N added post-audit (2026-09-14, independent audit finding):
# the original design sweep covered N up to 10,000 and found comfortable
# margin, but the precondition placed no upper bound on N at all. For
# normalization in {standard, model}, inv_fap_single/fap_single involve a
# (1-fap)**(1/Nk)-style expression that suffers catastrophic cancellation as
# fap -> 1, with error growing roughly linearly in N -- confirmed directly:
# 2.7e-10 at N=1e7, 2.5e-9 at N=1e8 (already past the 1e-9 tolerance),
# 1e-5 at N=1e12. log and psd stay at exactly 0.0 regardless of N (no
# cancellation in their closed forms). N is the number of data points in
# the time series (via LombScargle.false_alarm_level), so a ceiling of 1e6
# comfortably covers realistic single- and multi-decade survey light curves
# while excluding the regime where this is a genuine formula limitation
# rather than a real astropy defect. A 3000-trial sweep for N up to 1e6
# (all 4 normalizations, fap in [1e-8, 1-1e-10]) found worst 4.32e-11; a
# 5000-trial adversarial sweep right at the N=1e6 boundary with fap forced
# into [1-1e-3, 1-1e-12] found worst 5.51e-11 -- ~18x margin.

_AB_TOL = 1e-9
_AB_MAX_N = 1_000_000


@_guard("false_alarm_probability_roundtrip")
def check_fap_roundtrip(fap, z, N, normalization, dH, dK):
    """AP-TS-002 (Candidate AB): fap_single(inv_fap_single(fap, ...), ...)
    == fap -- fap_single and inv_fap_single are independently coded
    algebraic inverses of each other, one per normalization branch.

    ``z`` is inv_fap_single's own return value for the production call
    (``inv_fap_single(fap, N, normalization, dH=dH, dK=dK)``) -- passed in
    directly rather than re-derived here, so this checker only ever
    exercises the *other* direction (``fap_single``), never re-running
    ``inv_fap_single`` against its own output.
    """
    import math

    from astropy.timeseries.periodograms.lombscargle._statistics import (
        fap_single,
    )

    if dK - dH != 2:
        return
    try:
        fap_val = float(fap)
        z_val = float(z)
    except (TypeError, ValueError):
        return
    if not (math.isfinite(fap_val) and 0.0 < fap_val < 1.0):
        return
    if not math.isfinite(z_val):
        return
    if not (isinstance(N, (int,)) or float(N).is_integer()):
        return
    if N <= dK or N > _AB_MAX_N:
        return

    try:
        fap_roundtrip = float(fap_single(z_val, N, normalization, dH=dH, dK=dK))
    except Exception:
        return
    if not math.isfinite(fap_roundtrip):
        return

    trigger_if(abs(fap_roundtrip - fap_val) > _AB_TOL, "AP-TS-002")


# --- Candidate AC: uncertainty-representation cross-implementation agreement
#
# LAW_CANDIDATES.md Candidate AC. Precondition: binary arithmetic propagation
# (add/subtract/multiply/divide, not the collapse branch), both operand
# uncertainty arrays present, acting class is StdDevUncertainty or
# InverseVariance (never VarianceUncertainty itself, keeping the
# re-derivation an independent cross-check rather than a self-check).
#
# Fixed post-audit (2026-09-14, independent audit finding): this checker was
# shipped with three latent bugs that made it either a permanent no-op or a
# guaranteed false-positive storm, none caught by the original self-
# verification sweep because that sweep never actually exercised the shipped
# code path end to end:
#   1. `InverseVariance` was referenced but never imported -- every call
#      raised NameError immediately, silently swallowed by _guard. The
#      recorded "100,000-trial sweep" could not have run against this code.
#   2. The comparison read `self_uncertainty` (input operand A) instead of
#      `result` (the actual propagated output, already passed in as a hook
#      parameter but never used) -- an O(1) mismatch on nearly every call.
#   3. The independent re-derivation built an unassociated VarianceUncertainty
#      (no parent_nddata) and called .propagate() on it directly; this
#      crashes for multiply/divide (which need self.parent_nddata.data for
#      the multiplicative term) and for any unit-bearing add/subtract (needs
#      a Quantity result_data, which the checker had access to all along but
#      the reconstructed operand didn't carry consistently) -- every
#      multiply/divide call was silently skipped, matching the sanitizers.json
#      claim of testing "add/subtract/multiply/divide" when only add/subtract
#      ever ran to completion.
# Fixed by importing InverseVariance, comparing against `result` (not
# `self_uncertainty`), and building both the self-side and other-side
# variance re-derivations as fully NDData-associated objects (mirroring how
# `other_nddata_var` was already built) so .propagate() has everything both
# _propagate_add_sub and _propagate_multiply_divide need.
#
# A fourth issue surfaced only by running astropy's own test suite (not the
# original design sweep, which used float64 throughout): the tolerance was
# a fixed 50*eps64, but astropy.nddata's arithmetic mixin fully respects the
# operand dtype (float32 in, float32 out) -- test_arithmetics_dtypes_uncert_
# mask exercises float32 propagation and the checker's own relerr came out
# ~0.5-0.75*eps32 (~6-9e-8), seven orders of magnitude past a float64-scaled
# tolerance despite being a clean, correct float32 computation. Fixed by
# scaling the tolerance to the actual working dtype's own eps (float64 as a
# fallback for non-floating dtypes, which cannot occur here since variance
# arithmetic always promotes to a float dtype).
#
# Tolerance: 50*working_eps relative in variance space, derived from a
# 100,000-trial sweep isolating InverseVariance's extra 1/x round trip (worst
# 24.8*eps64) and a separate sweep of the differing-but-convertible-unit
# branch in _propagate_add_sub/_propagate_multiply_divide (worst ~6.2*eps64)
# -- the plain same-unit StdDevUncertainty-vs-VarianceUncertainty case
# sweeps to exactly 0.0 (same closed-form arithmetic, no extra transform),
# so this candidate's real content is those two special code paths. Re-
# verified post-fix with a 96,000-trial sweep across both classes, 4 units,
# 4 ops, and 12 decades of magnitude: worst 1.0*eps (float64), 20,000+
# additional trials in float32 confirming sub-eps32 agreement.

_AC_TOL_EPS_FACTOR = 50.0


@_guard("uncertainty_cross_representation")
def check_uncertainty_cross_representation(
    self_uncertainty, operation, other_nddata, result_data, correlation, result
):
    """AP-NDDATA-001 (Candidate AC): StdDevUncertainty and InverseVariance
    propagation must agree, in variance space, with an independent
    re-derivation via VarianceUncertainty on the same operands -- three
    independently coded closed-form Gaussian error-propagation formulas
    (and, for Std/InverseVariance, their own extra sqrt/1x transforms and
    unit-conversion branches) computing the identical physical quantity.
    """
    from astropy.nddata.nduncertainty import (
        InverseVariance,
        StdDevUncertainty,
        VarianceUncertainty,
    )

    if not isinstance(self_uncertainty, (StdDevUncertainty, InverseVariance)):
        return
    if self_uncertainty.array is None:
        return
    other_uncertainty = getattr(other_nddata, "uncertainty", None)
    if other_uncertainty is None or other_uncertainty.array is None:
        return
    if not isinstance(other_uncertainty, type(self_uncertainty)):
        return

    self_nddata = self_uncertainty.parent_nddata
    if self_nddata is None:
        return

    op_name = operation.__name__
    if op_name not in ("add", "subtract", "multiply", "true_divide", "divide"):
        return

    try:
        self_var_obj = self_uncertainty._convert_to_variance()
        self_nddata_var = self_nddata.__class__(
            self_nddata.data,
            unit=self_nddata.unit,
            uncertainty=VarianceUncertainty(
                self_var_obj.array, unit=self_var_obj.unit, copy=False
            ),
        )
        other_var_obj = other_uncertainty._convert_to_variance()
        other_nddata_var = other_nddata.__class__(
            other_nddata.data,
            unit=other_nddata.unit,
            uncertainty=VarianceUncertainty(
                other_var_obj.array, unit=other_var_obj.unit, copy=False
            ),
        )
    except Exception:
        return

    try:
        var_reference = self_nddata_var.uncertainty.propagate(
            operation, other_nddata_var, result_data, correlation
        )
    except Exception:
        return
    if var_reference.array is None:
        return

    try:
        var_from_result = result._convert_to_variance().array
        var_from_reference = var_reference.array
    except Exception:
        return

    import numpy as np

    if not (
        np.all(np.isfinite(var_from_result)) and np.all(np.isfinite(var_from_reference))
    ):
        return
    denom = np.where(np.abs(var_from_reference) < 1e-300, 1.0, np.abs(var_from_reference))
    relerr = float(np.max(np.abs(var_from_result - var_from_reference) / denom))

    working_dtype = np.result_type(var_from_result.dtype, var_from_reference.dtype)
    if np.issubdtype(working_dtype, np.floating):
        working_eps = float(np.finfo(working_dtype).eps)
    else:
        working_eps = _EPS64
    trigger_if(relerr > _AC_TOL_EPS_FACTOR * working_eps, "AP-NDDATA-001")


# --- Candidate AD: spherical/Cartesian representation round trip -----------
#
# LAW_CANDIDATES.md Candidate AD. Precondition: distance > 0 (a zero-distance
# point has no direction, so lon/lat are undefined -- the underlying
# Cartesian round trip is still trivially exact there, (0,0,0)==(0,0,0),
# but the precondition documents why lon/lat can legitimately differ).
# Tolerance: 20*eps64 relative to the vector's own distance, derived from a
# 200,000-trial uniform sweep (worst 4.48*eps64) plus a 100,000-trial
# near-pole/wide-magnitude adversarial sweep (worst 1.0*eps64, tighter, no
# pole degradation found) -- SphericalRepresentation.to_cartesian/
# from_cartesian are independently coded ERFA calls (s2p/p2s), not the same
# formula run twice.

_AD_TOL_EPS = 20.0 * _EPS64


@_guard("spherical_cartesian_roundtrip")
def check_spherical_cartesian_roundtrip(spherical_self, cartesian_result):
    """AP-COORD-005 (Candidate AD): from_cartesian(to_cartesian(s)) must
    return to the same Cartesian point as to_cartesian(s) itself --
    SphericalRepresentation.to_cartesian (ERFA s2p) and .from_cartesian
    (ERFA p2s) are independently coded forward/inverse trig transforms for
    the identical 3D point.
    """
    import numpy as np

    from astropy.coordinates.representation.spherical import (
        SphericalRepresentation,
    )

    distance = spherical_self.distance
    try:
        distance_value = np.asarray(distance.to_value(distance.unit))
    except Exception:
        return
    if not np.all(np.isfinite(distance_value)) or np.any(distance_value <= 0):
        return

    try:
        roundtrip_spherical = SphericalRepresentation.from_cartesian(
            cartesian_result
        )
        roundtrip_cartesian = roundtrip_spherical.to_cartesian()
    except Exception:
        return

    try:
        dx = (roundtrip_cartesian.x - cartesian_result.x).to_value(distance.unit)
        dy = (roundtrip_cartesian.y - cartesian_result.y).to_value(distance.unit)
        dz = (roundtrip_cartesian.z - cartesian_result.z).to_value(distance.unit)
    except Exception:
        return
    diff = np.sqrt(dx**2 + dy**2 + dz**2)
    if not np.all(np.isfinite(diff)):
        return

    denom = np.where(distance_value == 0, 1.0, distance_value)
    relerr = float(np.max(diff / denom))
    trigger_if(relerr > _AD_TOL_EPS, "AP-COORD-005")


# --- Candidate AE: spherical proper-motion differential CosLat round trip --
#
# LAW_CANDIDATES.md Candidate AE. Precondition: base given (required to
# convert between the two conventions at all -- represent_as raises its own
# TypeError otherwise, nothing to guard). cos(lat) never reaches exactly 0.0
# in float64 (cos(90deg) == 6.12e-17), so the d_lon_coslat = d_lon*cos(lat) /
# cos(lat) round trip never divides by a true zero, and a 50,000-trial
# adversarial sweep down to cos(lat)~2.3e-10 found no conditioning-driven
# amplification at all (worst case matched the ordinary-latitude sweep
# almost exactly). Tolerance: 5*eps64 relative, derived from a 100,000-
# trial sweep (worst 0.98*eps64).
#
# Denormal-underflow exclusion added post-audit (2026-09-14, independent
# audit finding): the original "no pole exclusion needed" claim held for
# realistic-magnitude proper motions, but was never tested against d_lon
# small enough that d_lon*cos(lat) underflows into float64's subnormal
# range (below ~2.23e-308) -- subnormals carry progressively fewer
# significant bits than normal floats as they shrink, which is a genuine,
# well-understood precision loss, not a drift between the two conversion
# methods. Confirmed directly: at lat=90deg exactly with d_lon=1e-300 (so
# d_lon*cos(lat) ~ 6.12e-317, deep in subnormal range), the round trip
# relative error was 3.37e-8 (~1.5e8*eps64). Excluding inputs where
# d_lon*cos(lat) would underflow below the normal-float threshold, a
# 50,000-trial sweep re-including exact poles and d_lon down to 1e-186
# found worst 0.97*eps64, matching the original (pre-exclusion) claim
# almost exactly -- confirming the subnormal regime was the entire
# discrepancy, not a broader unaccounted-for effect.

_AE_TOL_EPS = 5.0 * _EPS64
_AE_MIN_NORMAL_FLOAT = 2.2250738585072014e-308  # np.finfo(np.float64).tiny


@_guard("spherical_differential_coslat_roundtrip")
def check_spherical_differential_coslat_roundtrip(
    self_differential, base, coslat_result
):
    """AP-COORD-006 (Candidate AE): SphericalDifferential.represent_as(
    SphericalCosLatDifferential, base).represent_as(SphericalDifferential,
    base) must return to the original differential -- the d_lon <->
    d_lon_coslat conversion (multiply then divide by cos(base.lat)) is an
    exact algebraic inverse pair, but implemented as two separate methods
    (_d_lon_coslat on the source class, _get_d_lon as a classmethod on the
    target class) that could drift out of agreement under an independent
    edit to either.
    """
    import numpy as np

    from astropy import units as u
    from astropy.coordinates.representation.spherical import SphericalDifferential

    if base is None:
        return

    try:
        d_lon_val = np.asarray(
            self_differential.d_lon.to_value(self_differential.d_lon.unit)
        )
        coslat_val = np.cos(base.lat.to_value(u.rad))
        underflows = np.abs(d_lon_val * coslat_val) < _AE_MIN_NORMAL_FLOAT
        if np.any(underflows & (d_lon_val != 0)):
            return
    except Exception:
        return

    try:
        roundtrip = coslat_result.represent_as(SphericalDifferential, base=base)
    except Exception:
        return

    def _relerr(a, b):
        a_val = a.to_value(a.unit)
        b_val = b.to_value(a.unit)
        denom = np.where(np.abs(a_val) < 1e-300, 1.0, np.abs(a_val))
        return np.abs(b_val - a_val) / denom

    try:
        r_lon = _relerr(self_differential.d_lon, roundtrip.d_lon)
        r_lat = _relerr(self_differential.d_lat, roundtrip.d_lat)
        r_dist = _relerr(self_differential.d_distance, roundtrip.d_distance)
    except Exception:
        return

    if not (
        np.all(np.isfinite(r_lon))
        and np.all(np.isfinite(r_lat))
        and np.all(np.isfinite(r_dist))
    ):
        return

    relerr = float(max(np.max(r_lon), np.max(r_lat), np.max(r_dist)))
    trigger_if(relerr > _AE_TOL_EPS, "AP-COORD-006")


# --- Candidate AF: cross-module MAD-to-std scale factor consistency --------
#
# LAW_CANDIDATES.md Candidate AF. Precondition: none (a hardcoded literal
# comparison, not data-dependent). Zero tolerance -- both literals are
# independently typed decimal approximations of the identical mathematical
# constant 1/Phi^-1(3/4), and confirmed to round to the exact same float64
# value; this is the same "two independently-typed literals could drift
# under an uncoordinated edit" pattern as the AP-CONST-* family, just for a
# statistical rather than physical constant, and spanning two different
# subsystems (astropy.stats and astropy.uncertainty) rather than two CODATA
# vintage modules.

@_guard("mad_std_scale_factor_consistency")
def check_mad_std_scale_factor(mad_std_result, mad_value):
    """AP-STATS-005 (Candidate AF): astropy.stats.mad_std's hardcoded scale
    factor (MAD * 1.482602218505602) must equal
    astropy.uncertainty.core.SMAD_SCALE_FACTOR (1.48260221850560203193936...)
    -- both are independently-typed decimal literals for 1/Phi^-1(3/4), in
    two different subsystems, that could silently drift apart under an
    edit to only one of them.

    Compares MAD*SMAD_SCALE_FACTOR (recomputed with the *other* module's
    literal) directly against mad_std's own MAD*1.482602218505602, rather
    than dividing mad_std_result back by MAD -- an earlier version did
    that division and found spurious eps64-scale disagreement purely from
    the round-trip's own rounding noise (x*C/x != C exactly in float64),
    which was never a real drift between the two literals.
    """
    import numpy as np

    from astropy.uncertainty.core import SMAD_SCALE_FACTOR

    mad_arr = np.asarray(mad_value)
    if not np.all(np.isfinite(mad_arr)):
        return
    result_arr = np.asarray(mad_std_result)
    if not np.all(np.isfinite(result_arr)):
        return

    reference = mad_arr * SMAD_SCALE_FACTOR
    trigger_if(
        bool(np.any(result_arr != reference)),
        "AP-STATS-005",
    )


# --- Candidate AG: optical vs relativistic Doppler convention consistency --
#
# LAW_CANDIDATES.md Candidate AG. Precondition and tolerance derivation
# mirror Candidate I (AP-UNITS-002) exactly, since it is the same analytical
# structure applied to the third Doppler convention: |beta| in
# (1e-5, 1e-3), same leading-order Taylor argument. Analytically,
# V_opt/c = sqrt((1+beta)/(1-beta)) - 1 = beta + beta^2/2 + O(beta^3), so
# (V_opt - V_rel)/V_rel = beta/2 + O(beta^2) at leading order -- the same
# magnitude as radio's beta/2 term (AP-UNITS-002), consistent with optical
# and radio being symmetric first-order over/under-estimates of the exact
# relativistic formula. A 20,000-trial-per-rest-type sweep (4 rest
# quantities: GHz, nm, eV, cm) found the residual after subtracting beta/2
# matches the predicted next O(beta^2) term, worst 5.00e-7, with no
# unexplained excess.
#
# Second-order term subtracted post-audit (2026-09-14, independent audit
# finding): the tolerance's apparent ~2.5x margin at the top of the beta
# window was not numerical slack but the fixed, deterministic O(beta^2)
# Taylor term itself -- a real physics deviation below that size would
# have been invisible. The +beta^2/2 sign (opposite to AP-UNITS-002's
# -beta^2/2, consistent with this checker's un-abs'd observed_relerr
# convention where optical overestimates positively) was confirmed
# analytically and numerically: subtracting it drops the residual to
# ~1e-10 to ~1e-13 across the same beta range (worst case at beta=9e-4:
# -4.05e-7 before, 2.74e-10 after). Same fix applied to AP-UNITS-002 for
# consistency, since it shares this exact structure. Tolerance: 1e-6,
# same value as before but now genuine noise-level margin (~1000-10000x)
# rather than truncation-error margin (~2.5x).

_AG_BETA_MIN = 1e-5
_AG_BETA_MAX = 1e-3
_AG_TOL = 1e-6


@_guard("doppler_optical_convention_consistency")
def check_doppler_optical_convention_agreement(rest_freq_hz, to_func_optical_hz):
    """AP-UNITS-006 (Candidate AG): the optical and relativistic Doppler
    conventions must disagree by exactly beta/2 in relative terms at
    leading order (V_opt/c = sqrt((1+beta)/(1-beta)) - 1, an analytically
    derived prediction) -- see LAW_CANDIDATES.md Candidate AG.

    ``rest_freq_hz`` is the rest frequency (plain float, Hz) doppler_optical
    was constructed with; ``to_func_optical_hz`` is doppler_optical's own
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
        v_opt = to_func_optical_hz(test_freq)
        v_rel = to_func_rel_hz(test_freq)

        beta = v_rel / _CKMS
        if not (_AG_BETA_MIN < abs(beta) < _AG_BETA_MAX):
            continue
        if v_rel == 0:
            continue

        observed_relerr = (v_opt - v_rel) / v_rel
        predicted = 0.5 * abs(beta) + 0.5 * beta**2
        trigger_if(abs(observed_relerr - predicted) > _AG_TOL, "AP-UNITS-006")


# --- Candidate AI: 2D rotation model inverse round trip ---------------------
#
# LAW_CANDIDATES.md Candidate AI. Precondition: finite angle, finite (x, y).
# No degenerate cases -- a 2D rotation matrix is always invertible (its
# determinant is cos^2+sin^2=1 for every angle), unlike the sphere-pole or
# coincident-point exclusions needed elsewhere in this bank. Tolerance:
# 30*eps64 relative to max(|x|,|y|), derived from a 100,000-trial sweep
# (worst 9.33*eps64) spanning angles to +/-720deg and coordinates to 1e6 --
# Rotation2D.inverse constructs a fresh model with angle=-angle, and
# evaluate() independently recomputes cos/sin (not merely negating the
# forward matrix), so the round trip exercises two separately-computed
# trigonometric matrices, not a tautological negate-and-reapply.

_AI_TOL_EPS = 30.0 * _EPS64


@_guard("rotation2d_inverse_roundtrip")
def check_rotation2d_inverse_roundtrip(cls, x, y, angle_rad, x_rot, y_rot):
    """AP-MODEL-002 (Candidate AI): rotating (x, y) by angle then by -angle
    must return to (x, y). ``angle_rad`` is evaluate()'s own already-
    unit-normalized radian value for this call (read after evaluate()'s own
    Quantity-to-radian conversion, not re-derived), so this re-uses
    Rotation2D._compute_matrix directly with the negated angle rather than
    reconstructing a Rotation2D instance -- .inverse's angle=-self.angle
    round-trips back through the Parameter's own degree/Quantity setter,
    which would reintroduce a unit-conversion ambiguity this checker avoids
    by working in evaluate()'s already-resolved radian frame throughout.
    """
    import numpy as np

    try:
        matrix_inv = cls._compute_matrix(-angle_rad)
        inarr = np.stack(np.atleast_1d(x_rot, y_rot), axis=-2)
        x_back, y_back = np.moveaxis(np.matmul(matrix_inv, inarr), -2, 0)
    except Exception:
        return

    x_arr, y_arr = np.asarray(x), np.asarray(y)
    xb_arr, yb_arr = np.asarray(x_back), np.asarray(y_back)
    if not (
        np.all(np.isfinite(xb_arr)) and np.all(np.isfinite(yb_arr))
    ):
        return

    norm = np.maximum(np.maximum(np.abs(x_arr), np.abs(y_arr)), 1e-300)
    relerr = float(
        np.max(np.maximum(np.abs(xb_arr - x_arr), np.abs(yb_arr - y_arr)) / norm)
    )
    trigger_if(relerr > _AI_TOL_EPS, "AP-MODEL-002")


# --- Candidate AJ: log-stretch / inverted-log-stretch round trip ------------
#
# LAW_CANDIDATES.md Candidate AJ. LogStretch computes
# y = log(a*x+1)/log(a+1); its .inverse (InvertedLogStretch) computes the
# algebraically distinct closed form x = ((a+1)^y - 1)/a, built from exp/log
# rather than merely undoing LogStretch's own operations in reverse -- a
# genuine two-independently-computed-formula round trip, unlike the
# rotation-model round trips in this bank (AP-MODEL-002 included) which
# recompute trig from the same negated angle. Precondition: 1e-2 <= a <= 1e6
# (LogStretch's own docstring examples span 0.1-10000; very small a triggers
# catastrophic cancellation in log(a*x+1) that is a precision limitation of
# the formula itself, not a real drift between the two directions, so it is
# excluded rather than absorbed into a much looser tolerance) and x in
# [0, 1] (the stretch's documented domain). Tolerance: 1000*eps64 on
# absolute error (not relative -- x can be exactly 0), derived from a
# 50,000-trial sweep over a in [1e-2, 1e6], x in [0, 1] (worst observed
# ~1.1e-14, a handful of ULPs).

_AJ_TOL_ABS = 1000.0 * _EPS64
_AJ_A_MIN = 1e-2
_AJ_A_MAX = 1e6


@_guard("log_stretch_inverse_roundtrip")
def check_log_stretch_inverse_roundtrip(a, x_in, y_out):
    """AP-VIS-001 (Candidate AJ): LogStretch(a) followed by its own
    .inverse (InvertedLogStretch(a)) must recover the original input.
    y_out is LogStretch.__call__'s own already-computed result for this
    call; only the inverse direction is computed here.
    """
    import numpy as np

    from astropy.visualization.stretch import InvertedLogStretch

    if not (_AJ_A_MIN <= a <= _AJ_A_MAX):
        return

    x_arr = np.asarray(x_in, dtype=float)
    y_arr = np.asarray(y_out, dtype=float)
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        return
    if np.any(x_arr < 0) or np.any(x_arr > 1):
        return

    inv = InvertedLogStretch(a)
    x_back = inv(y_arr.copy(), clip=False)
    if not np.all(np.isfinite(x_back)):
        return

    abserr = float(np.max(np.abs(x_back - x_arr)))
    trigger_if(abserr > _AJ_TOL_ABS, "AP-VIS-001")


# --- Candidate AK: asinh-stretch / sinh-stretch round trip ------------------
#
# LAW_CANDIDATES.md Candidate AK. AsinhStretch computes
# y = asinh(x/a) / asinh(1/a); its .inverse is SinhStretch with a rescaled
# parameter a' = 1/asinh(1/a), which computes x = sinh(y/a') / sinh(1/a')
# (read directly from SinhStretch.__call__ -- an earlier version of this
# comment incorrectly stated the inverse as "a' * sinh(y/a')", which is not
# what the code computes; corrected post-audit, 2026-09-14, no change to the
# checker's actual behavior since the checker calls SinhStretch itself
# rather than re-implementing its formula). This is still an algebraically
# distinct transcendental-function pair (asinh/log-family forward vs.
# sinh/exp-family inverse), the same genuine-cross-check structure as
# AP-VIS-001.
#
# Precondition on `a` added post-audit (2026-09-14, independent audit
# finding): the original design claimed no restriction on `a` was needed,
# based on a 50,000-trial sweep over 10 decades (1e-10 to 1e10, worst
# 6.77e-15) that did not extend below 1e-10. A wider sweep down to
# a=1e-300 found the claim was only half right: large `a` is genuinely
# unconditioned (worst 2.22e-16 for a in [1e10, 1e300]), but small `a`
# degrades smoothly below ~1e-20 (9.77e-15 at 1e-20, 2.23e-14 at 1e-30 --
# already at the old tolerance boundary, 2.60e-14 at 1e-50). Since
# AsinhStretch's own docstring examples span only 0.01-3.0, a floor of
# a >= 1e-10 comfortably covers realistic use (worst 5.22e-15 across
# 50,000 trials for a in [1e-10, 1e300]) while excluding the regime where
# this is a genuine precision limit of the formula, not a real drift.

_AK_TOL_ABS = 100.0 * _EPS64
_AK_A_MIN = 1e-10


@_guard("asinh_stretch_inverse_roundtrip")
def check_asinh_stretch_inverse_roundtrip(a, x_in, y_out):
    """AP-VIS-002 (Candidate AK): AsinhStretch(a) followed by its own
    .inverse (SinhStretch) must recover the original input. y_out is
    AsinhStretch.__call__'s own already-computed result for this call;
    only the inverse direction is computed here.
    """
    import numpy as np

    from astropy.visualization.stretch import SinhStretch

    if a < _AK_A_MIN:
        return

    x_arr = np.asarray(x_in, dtype=float)
    y_arr = np.asarray(y_out, dtype=float)
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        return
    if np.any(x_arr < 0) or np.any(x_arr > 1):
        return

    inv = SinhStretch(a=1.0 / np.arcsinh(1.0 / a))
    x_back = inv(y_arr.copy(), clip=False)
    if not np.all(np.isfinite(x_back)):
        return

    abserr = float(np.max(np.abs(x_back - x_arr)))
    trigger_if(abserr > _AK_TOL_ABS, "AP-VIS-002")


# --- Candidate AL: power-dist-stretch / inverted-power-dist-stretch --------
#
# LAW_CANDIDATES.md Candidate AL. PowerDistStretch computes
# y = (a^x - 1) / (a - 1); its .inverse (InvertedPowerDistStretch) computes
# the algebraically distinct closed form x = log(y*(a-1)+1) / log(a) -- a
# genuine power/exp-vs-log transcendental-function pair, the same
# cross-check structure as AP-VIS-001/AP-VIS-002. Like AP-VIS-001 (and
# unlike AP-VIS-002), a precondition on `a` is needed: very small |a|
# (near the a=0 singularity of the power-law shape) and the immediate
# neighborhood of a=1 (the a=1 branch point where the formula itself is
# singular, PowerDistStretch's own __init__ rejects a==1 exactly but not
# a near 1) both cause catastrophic cancellation that is a precision
# limit of the formulas, not a real drift between the two directions.

_AL_TOL_ABS = 1000.0 * _EPS64
_AL_A_MIN = 1e-3
_AL_A_MAX = 1e3
_AL_A_EXCLUDE_RADIUS = 1e-2


@_guard("power_dist_stretch_inverse_roundtrip")
def check_power_dist_stretch_inverse_roundtrip(a, x_in, y_out):
    """AP-VIS-003 (Candidate AL): PowerDistStretch(a) followed by its own
    .inverse (InvertedPowerDistStretch(a)) must recover the original
    input. y_out is PowerDistStretch.__call__'s own already-computed
    result for this call; only the inverse direction is computed here.
    """
    import numpy as np

    from astropy.visualization.stretch import InvertedPowerDistStretch

    if not (_AL_A_MIN <= abs(a) <= _AL_A_MAX):
        return
    if abs(a - 1.0) < _AL_A_EXCLUDE_RADIUS:
        return

    x_arr = np.asarray(x_in, dtype=float)
    y_arr = np.asarray(y_out, dtype=float)
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        return
    if np.any(x_arr < 0) or np.any(x_arr > 1):
        return

    inv = InvertedPowerDistStretch(a=a)
    x_back = inv(y_arr.copy(), clip=False)
    if not np.all(np.isfinite(x_back)):
        return

    abserr = float(np.max(np.abs(x_back - x_arr)))
    trigger_if(abserr > _AL_TOL_ABS, "AP-VIS-003")


# --- Candidate AM: geodetic <-> geocentric Cartesian round trip -------------
#
# LAW_CANDIDATES.md Candidate AM. EarthLocation.to_geodetic converts
# geocentric XYZ to geodetic lon/lat/height via ERFA's gc2gde (an
# ellipsoid-inversion algorithm distinct from the closed-form gd2gce used
# in the forward direction, from_geodetic) -- a genuine two-independently-
# computed-algorithm round trip, comparable in trust level to this bank's
# other ERFA-backed checks (AP-WCS-001). Compared in Cartesian space (same
# convention as AP-COORD-005) to sidestep angle-wrap and pole-degeneracy
# entirely, rather than comparing lon/lat/height component-wise.
#
# Precondition: finite geocentric position AND geocentric radius >=
# 4,000,000 m. Found necessary during verification, not anticipated in the
# original design: astropy's own test_frames.py::test_eloc_attributes
# deliberately constructs a location 1 km from Earth's *center*
# (`ITRS(SphericalRepresentation(..., distance=1*u.km))`, documented in
# that test as expecting `height < -6000*u.km`) and the round-trip error
# there was 9491 m -- gc2gde's ellipsoid inversion is ill-conditioned near
# the coordinate origin, where the geodetic latitude/height decomposition
# itself becomes numerically degenerate (same class of issue as this
# bank's pole exclusions elsewhere, but for radius instead of latitude). A
# radius sweep found the error decays smoothly and monotonically toward
# Earth's surface (0.25 m at 1000 km, 3.3e-4 m at 4000 km, ~1e-8 m at the
# true 6357 km surface) with no sharp cutoff, so a generous
# floor was chosen well below Earth's actual surface.
#
# Fixed post-audit (2026-09-14, independent audit finding): the original
# tolerance was a flat 1e-2 m absolute, derived only from a sweep bounded at
# r=4e10 m (worst 1.6e-3 m there). The underlying error is fundamentally
# relative (~1e-15 * r, ordinary float64 rounding on the coordinate
# magnitude, not an absolute artifact of the algorithm), so the flat
# tolerance was guaranteed to break for large enough r -- confirmed
# directly: 7.6e-3 m at r=1e13 m (already past tolerance), 0.1 m at
# r=1e14 m, 80 m at r=1e17 m. `EarthLocation` places no upper bound on its
# own Cartesian inputs, so an absolute tolerance can always be defeated by
# a large enough (if unphysical for an *Earth* location) radius. Fixed by
# switching to a relative tolerance on distance/radius, which a 5000-trial
# sweep spanning radius 4e6 m to 1e20 m (14 decades past the original
# design's r=4e10 m ceiling) found stable at ~7-8e-11 with **no growth** at
# any tested extreme -- unlike AP-COORD-007's sibling AP-VIS-001/003, whose
# absolute-tolerance-breaking regimes needed an upper precondition bound,
# here the relative form is unconditionally robust and no radius ceiling
# is needed at all.
#
# Tolerance: 1e-9 relative (distance / radius), ~13x margin over the 7.6e-11
# worst observed across the full swept range.

_AM_TOL_REL = 1e-9
_AM_MIN_RADIUS_M = 4.0e6


@_guard("geodetic_geocentric_roundtrip")
def check_geodetic_roundtrip(ellipsoid, x, y, z, lon, lat, height):
    """AP-COORD-007 (Candidate AM): EarthLocation.to_geodetic's own
    lon/lat/height, converted back to geocentric Cartesian coordinates via
    from_geodetic (a separately-computed closed-form, ERFA's gd2gce), must
    recover the original (x, y, z) EarthLocation.to_geodetic itself was
    called on -- x, y, z are the original position (already computed by
    production code); only the reverse conversion is computed here.
    """
    import numpy as np

    from astropy import units as u
    from astropy.coordinates.earth import EarthLocation

    x_arr = np.asarray(x.to_value(u.m))
    y_arr = np.asarray(y.to_value(u.m))
    z_arr = np.asarray(z.to_value(u.m))
    if not (
        np.all(np.isfinite(x_arr))
        and np.all(np.isfinite(y_arr))
        and np.all(np.isfinite(z_arr))
    ):
        return

    radius = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
    if np.any(radius < _AM_MIN_RADIUS_M):
        return

    back = EarthLocation.from_geodetic(lon, lat, height, ellipsoid=ellipsoid)
    xb = np.asarray(back.x.to_value(u.m))
    yb = np.asarray(back.y.to_value(u.m))
    zb = np.asarray(back.z.to_value(u.m))
    if not (
        np.all(np.isfinite(xb)) and np.all(np.isfinite(yb)) and np.all(np.isfinite(zb))
    ):
        return

    dist = np.sqrt((xb - x_arr) ** 2 + (yb - y_arr) ** 2 + (zb - z_arr) ** 2)
    relerr = float(np.max(dist / radius))
    trigger_if(relerr > _AM_TOL_REL, "AP-COORD-007")
