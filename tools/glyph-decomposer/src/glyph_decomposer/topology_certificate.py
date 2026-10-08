"""Runtime certificate that raster ink and its skeleton are topologically equivalent."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

_FOREGROUND_OFFSETS = tuple(
    (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
)
_BACKGROUND_OFFSETS = ((-1, 0), (0, -1), (0, 1), (1, 0))


@dataclass(frozen=True)
class TopologyCertificate:
    ink_components: int
    skeleton_components: int
    ink_holes: int
    skeleton_holes: int
    ink_euler: int
    skeleton_euler: int
    skeleton_inside_ink: bool
    every_ink_component_represented: bool

    @property
    def certified(self) -> bool:
        return (
            self.ink_components == self.skeleton_components
            and self.ink_holes == self.skeleton_holes
            and self.ink_euler == self.skeleton_euler
            and self.skeleton_inside_ink
            and self.every_ink_component_represented
        )


def _label_components(mask: np.ndarray, offsets) -> tuple[int, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.full(mask.shape, -1, dtype=np.int32)
    component = 0
    height, width = mask.shape
    for seed_y, seed_x in np.argwhere(mask):
        if labels[seed_y, seed_x] >= 0:
            continue
        labels[seed_y, seed_x] = component
        queue = deque([(int(seed_y), int(seed_x))])
        while queue:
            y, x = queue.popleft()
            for dy, dx in offsets:
                next_y, next_x = y + dy, x + dx
                if not (0 <= next_y < height and 0 <= next_x < width):
                    continue
                if not mask[next_y, next_x] or labels[next_y, next_x] >= 0:
                    continue
                labels[next_y, next_x] = component
                queue.append((next_y, next_x))
        component += 1
    return component, labels


def _hole_count(mask: np.ndarray) -> int:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    background_components, _labels = _label_components(~padded, _BACKGROUND_OFFSETS)
    return max(0, background_components - 1)


def certify_skeleton(ink: np.ndarray, skeleton: np.ndarray) -> TopologyCertificate:
    ink = np.asarray(ink, dtype=bool)
    skeleton = np.asarray(skeleton, dtype=bool)
    if ink.shape != skeleton.shape:
        raise ValueError("ink and skeleton must use the same raster grid")
    ink_components, ink_labels = _label_components(ink, _FOREGROUND_OFFSETS)
    skeleton_components, _skeleton_labels = _label_components(
        skeleton, _FOREGROUND_OFFSETS
    )
    ink_holes = _hole_count(ink)
    skeleton_holes = _hole_count(skeleton)
    represented = {
        int(ink_labels[y, x])
        for y, x in np.argwhere(skeleton & ink)
        if ink_labels[y, x] >= 0
    }
    return TopologyCertificate(
        ink_components=ink_components,
        skeleton_components=skeleton_components,
        ink_holes=ink_holes,
        skeleton_holes=skeleton_holes,
        ink_euler=ink_components - ink_holes,
        skeleton_euler=skeleton_components - skeleton_holes,
        skeleton_inside_ink=bool(np.all(~skeleton | ink)),
        every_ink_component_represented=(
            ink_components > 0 and len(represented) == ink_components
        ),
    )
