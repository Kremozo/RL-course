"""Egocentric 3x3 observation window (assignment Option B)."""
from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet, Optional, Tuple

if TYPE_CHECKING:
    from world_model import Board, Pos
else:
    Pos = Tuple[int, int]

FREE, WALL, BOX = 0, 1, 2
Window = Tuple[Tuple[int, int, int], ...]


def window_at(pos: "Pos", board: "Board", small: FrozenSet["Pos"],
              heavy: Optional["Pos"]) -> Window:
    boxes = set(small)
    if heavy is not None:
        boxes.add(heavy)

    def read(dx: int, dy: int) -> int:
        if dx == 0 and dy == 0:
            return FREE
        x, y = pos[0] + dx, pos[1] + dy
        if not (0 <= x < board.width and 0 <= y < board.height):
            return WALL
        if board.walls[y, x]:
            return WALL
        return BOX if (x, y) in boxes else FREE

    return tuple(tuple(read(dx, dy) for dx in (-1, 0, 1)) for dy in (-1, 0, 1))