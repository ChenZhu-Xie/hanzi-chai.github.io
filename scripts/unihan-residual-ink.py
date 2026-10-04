"""Route-bounded PDF ink ownership and sequential residual state."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ResidualInkLedger:
    unexplained: np.ndarray
    reusable_contact: np.ndarray
    visible_owner: np.ndarray
    consumed_edge_ids: frozenset[int]

    @property
    def available(self) -> np.ndarray:
        return self.unexplained | self.reusable_contact


@dataclass(frozen=True)
class StrokeRegion:
    mask: np.ndarray
    contact_mask: np.ndarray
    forbidden_leak_mask: np.ndarray


def initial_ledger(target: np.ndarray) -> ResidualInkLedger:
    target = np.asarray(target, dtype=bool)
    return ResidualInkLedger(
        unexplained=target.copy(),
        reusable_contact=np.zeros(target.shape, dtype=bool),
        visible_owner=np.full(target.shape, -1, dtype=np.int16),
        consumed_edge_ids=frozenset(),
    )


def _point_mask(shape: tuple[int, int], graph, point_indices: set[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if point_indices:
        points = graph.points[np.asarray(sorted(point_indices), dtype=int)]
        mask[points[:, 0], points[:, 1]] = True
    return mask


def _route_contacts(
    shape: tuple[int, int],
    target: np.ndarray,
    graph,
    directed,
    route_edge_ids: tuple[int, ...],
    contact_radius: int,
) -> np.ndarray:
    route_edges = [directed.edges[edge_id] for edge_id in route_edge_ids]
    gate_ids = {
        gate_id
        for edge in route_edges
        for gate_id in (edge.start_gate, edge.end_gate)
        if len(directed.gates[gate_id].outgoing) >= 3
    }
    points = {
        point_index
        for gate_id in gate_ids
        for point_index in directed.gates[gate_id].point_indices
    }
    mask = _point_mask(shape, graph, points)
    if contact_radius > 0 and mask.any():
        diameter = contact_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask & target


def recover_stroke_region(
    target: np.ndarray,
    graph,
    directed,
    route_edge_ids: tuple[int, ...],
    route_points: np.ndarray,
    ledger: ResidualInkLedger,
    contact_radius: int = 2,
) -> StrokeRegion:
    """Restore width around explicit roads without entering other roads.

    Selected and unselected skeleton roads form a local watershed.  Unlike the
    legacy owner flood, unselected roads are competitors/barriers rather than
    destinations that must be assigned to the current stroke.
    """

    del route_points  # edge identities, not candidate coordinates, define ownership
    target = np.asarray(target, dtype=bool)
    selected_indices = {
        point_index
        for edge_id in route_edge_ids
        for point_index in directed.edges[edge_id].point_indices
    }
    if not selected_indices:
        empty = np.zeros(target.shape, dtype=bool)
        return StrokeRegion(empty, empty.copy(), empty.copy())

    selected = _point_mask(target.shape, graph, selected_indices)
    all_skeleton_indices = set(range(len(graph.points)))
    forbidden_indices = all_skeleton_indices - selected_indices
    forbidden = _point_mask(target.shape, graph, forbidden_indices)

    selected_distance = ndimage.distance_transform_edt(~selected)
    if forbidden.any():
        forbidden_distance = ndimage.distance_transform_edt(~forbidden)
        selected_cell = selected_distance <= forbidden_distance
    else:
        selected_cell = np.ones(target.shape, dtype=bool)

    contact = _route_contacts(
        target.shape,
        target,
        graph,
        directed,
        route_edge_ids,
        contact_radius,
    )
    available = ledger.available & target
    mask = available & (selected_cell | contact)
    forbidden_leak = mask & forbidden & ~contact
    return StrokeRegion(mask=mask, contact_mask=contact & mask, forbidden_leak_mask=forbidden_leak)


def advance_ledger(
    ledger: ResidualInkLedger,
    region: StrokeRegion,
    stroke_index: int,
    consumed_edge_ids: tuple[int, ...],
) -> ResidualInkLedger:
    unexplained = ledger.unexplained & ~region.mask
    reusable_contact = ledger.reusable_contact | region.contact_mask
    visible_owner = ledger.visible_owner.copy()
    visible_owner[region.mask] = int(stroke_index)
    return ResidualInkLedger(
        unexplained=unexplained,
        reusable_contact=reusable_contact,
        visible_owner=visible_owner,
        consumed_edge_ids=ledger.consumed_edge_ids | frozenset(consumed_edge_ids),
    )
