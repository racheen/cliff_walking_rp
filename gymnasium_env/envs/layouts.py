from __future__ import annotations


def cliff_positions_for_layout(xsize: int, ysize: int, layout: str = "bottom") -> set[tuple[int, int]]:
    """Return trap cells while keeping the start and goal cells open."""
    if xsize < 3 or ysize < 2:
        raise ValueError("Trap grids need at least width 3 and height 2.")

    if layout == "bottom":
        return {(i, ysize - 1) for i in range(1, xsize - 1)}

    if layout == "gap":
        gap = xsize // 2
        return {(i, ysize - 1) for i in range(1, xsize - 1) if i != gap}

    if layout == "middle":
        row = max(1, ysize - 2)
        return {(i, row) for i in range(2, xsize - 2)}

    if layout == "double":
        bottom = {(i, ysize - 1) for i in range(1, xsize - 1)}
        upper_row = max(1, ysize - 2)
        upper = {(i, upper_row) for i in range(2, xsize - 2, 2)}
        return bottom | upper

    if layout == "scattered":
        candidates = [
            (xsize // 3, max(0, ysize // 3)),
            ((xsize * 2) // 3, max(0, ysize // 3)),
            (xsize // 2, ysize // 2),
            (max(1, xsize // 4), max(1, (ysize * 2) // 3)),
            (min(xsize - 2, (xsize * 3) // 4), max(1, (ysize * 2) // 3)),
        ]
        return _valid_traps(candidates, xsize, ysize)

    if layout == "maze":
        traps: list[tuple[int, int]] = []
        for y in range(1, ysize - 1):
            if y != ysize // 2:
                traps.append((xsize // 3, y))
        for y in range(0, ysize - 2):
            if y != max(1, ysize // 3):
                traps.append(((xsize * 2) // 3, y))
        return _valid_traps(traps, xsize, ysize)

    if layout == "islands":
        traps = []
        centers = [
            (max(1, xsize // 3), max(1, ysize // 2)),
            (min(xsize - 2, (xsize * 2) // 3), max(1, ysize // 2)),
        ]
        for cx, cy in centers:
            traps.extend([(cx, cy), (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)])
        return _valid_traps(traps, xsize, ysize)

    if layout == "mixed":
        traps = set(cliff_positions_for_layout(xsize, ysize, "bottom"))
        traps.update(cliff_positions_for_layout(xsize, ysize, "scattered"))

        lower_barrier_row = max(1, ysize - 2)
        for x in range(2, max(2, xsize - 2)):
            traps.add((x, lower_barrier_row))

        for y in range(1, max(1, ysize - 2)):
            if y != max(1, ysize // 2):
                traps.add((xsize // 3, y))

        island_centers = [
            (min(xsize - 3, max(2, (xsize * 2) // 3)), max(1, ysize // 2)),
            (min(xsize - 3, max(2, (xsize * 3) // 4)), max(1, ysize // 3)),
        ]
        for cx, cy in island_centers:
            traps.update([(cx, cy), (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)])

        return _valid_traps(list(traps), xsize, ysize)

    raise ValueError(f"Unknown trap layout: {layout}")


def _valid_traps(cells: list[tuple[int, int]], xsize: int, ysize: int) -> set[tuple[int, int]]:
    start = (0, ysize - 1)
    goal = (xsize - 1, ysize - 1)
    return {
        (x, y)
        for x, y in cells
        if 0 <= x < xsize and 0 <= y < ysize and (x, y) not in {start, goal}
    }
