from __future__ import annotations


def generate_tiles(H: int, W: int, tile: int):
    for y0 in range(0, H, tile):
        y1 = min(H, y0 + tile)
        for x0 in range(0, W, tile):
            x1 = min(W, x0 + tile)
            yield y0, y1, x0, x1


def sort_tiles_by_point(tiles, center_yx: tuple[float, float]):
    """
    Sort tiles by distance to a point (y, x). This is robust across napari versions.
    """
    cy, cx = center_yx

    def key(tile):
        y0, y1, x0, x1 = tile
        ty = 0.5 * (y0 + y1)
        tx = 0.5 * (x0 + x1)
        dist2 = (ty - cy) ** 2 + (tx - cx) ** 2
        return dist2

    return sorted(list(tiles), key=key)
