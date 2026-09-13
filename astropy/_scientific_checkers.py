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
