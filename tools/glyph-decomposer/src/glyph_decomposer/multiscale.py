"""Truth-free stability certificate for skeletons generated at several resolutions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .skeleton import generate_skeleton
from .topology_certificate import TopologyCertificate, certify_skeleton


@dataclass(frozen=True)
class ScaleSkeleton:
    size: int
    skeleton: np.ndarray
    certificate: TopologyCertificate


@dataclass(frozen=True)
class ScaleAgreement:
    first_size: int
    second_size: int
    mean_distance: float
    p95_distance: float
    bidirectional_coverage: float


@dataclass(frozen=True)
class MultiScaleCertificate:
    scales: tuple[ScaleSkeleton, ...]
    agreements: tuple[ScaleAgreement, ...]
    same_topology_signature: bool
    all_individually_certified: bool
    maximum_p95_distance: float
    minimum_bidirectional_coverage: float
    certified: bool


def _coordinates(skeleton: np.ndarray) -> np.ndarray:
    size = skeleton.shape[0]
    return (np.argwhere(skeleton)[:, ::-1] + 0.5) * (100.0 / size)


def _nearest_distances(first: np.ndarray, second: np.ndarray, chunk: int = 512):
    output = []
    for start in range(0, len(first), chunk):
        delta = first[start : start + chunk, None, :] - second[None, :, :]
        output.append(np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
    return np.concatenate(output)


def _agreement(
    first: ScaleSkeleton, second: ScaleSkeleton, tolerance: float
) -> ScaleAgreement:
    first_points = _coordinates(first.skeleton)
    second_points = _coordinates(second.skeleton)
    if not len(first_points) or not len(second_points):
        return ScaleAgreement(
            first_size=first.size,
            second_size=second.size,
            mean_distance=math.inf,
            p95_distance=math.inf,
            bidirectional_coverage=0.0,
        )
    forward = _nearest_distances(first_points, second_points)
    backward = _nearest_distances(second_points, first_points)
    distances = np.concatenate((forward, backward))
    coverage = (
        float(np.mean(forward <= tolerance)) + float(np.mean(backward <= tolerance))
    ) / 2
    return ScaleAgreement(
        first_size=first.size,
        second_size=second.size,
        mean_distance=float(distances.mean()),
        p95_distance=float(np.percentile(distances, 95)),
        bidirectional_coverage=coverage,
    )


def certify_multiscale(
    geometry,
    sizes: tuple[int, ...] = (192, 256, 384),
    *,
    tolerance: float = 1.0,
    maximum_p95: float = 1.25,
    minimum_coverage: float = 0.97,
) -> MultiScaleCertificate:
    if len(set(sizes)) < 2:
        raise ValueError("multi-scale certification requires at least two sizes")
    scales = []
    for size in sorted(set(sizes)):
        ink, skeleton = generate_skeleton(geometry, size)
        scales.append(
            ScaleSkeleton(
                size=size,
                skeleton=skeleton,
                certificate=certify_skeleton(ink, skeleton),
            )
        )
    signatures = {
        (
            scale.certificate.skeleton_components,
            scale.certificate.skeleton_holes,
            scale.certificate.skeleton_euler,
        )
        for scale in scales
    }
    agreements = tuple(
        _agreement(first, second, tolerance)
        for first, second in combinations(scales, 2)
    )
    all_certified = all(scale.certificate.certified for scale in scales)
    maximum_distance = max(item.p95_distance for item in agreements)
    minimum_agreement = min(item.bidirectional_coverage for item in agreements)
    same_signature = len(signatures) == 1
    return MultiScaleCertificate(
        scales=tuple(scales),
        agreements=agreements,
        same_topology_signature=same_signature,
        all_individually_certified=all_certified,
        maximum_p95_distance=maximum_distance,
        minimum_bidirectional_coverage=minimum_agreement,
        certified=(
            all_certified
            and same_signature
            and maximum_distance <= maximum_p95
            and minimum_agreement >= minimum_coverage
        ),
    )
