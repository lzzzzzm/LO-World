"""Cull disjoint map geometry before invoking the unchanged nuScenes renderer.

Each map belongs to one rendering worker. The SDK still performs clipping,
rotation and rasterization, in its original record order. Its layer traversal
is sequential; the temporary candidate lists are restored after every call.
"""
from functools import cached_property, lru_cache

import numpy as np
from shapely.geometry import Polygon

from tools.trajectory_generation.bev_layout import BEVLayoutRenderer


class _CachedMapPolygon(Polygon):
    # SDK polygons are immutable during rendering. In complex maps GEOS
    # validity checks dominate runtime even when the same polygon is reused.
    @cached_property
    def is_valid(self):
        return super().is_valid


def enable_spatial_map_cache(map_api):
    if getattr(map_api, '_layout_spatial_cache', False):
        return map_api
    original_extract_polygon = map_api.extract_polygon
    map_api.extract_polygon = lru_cache(maxsize=None)(
        lambda token: _CachedMapPolygon(original_extract_polygon(token)))
    map_api.extract_line = lru_cache(maxsize=None)(map_api.extract_line)
    explorer = map_api.explorer
    original_polygon = explorer._get_layer_polygon
    original_line = explorer._get_layer_line
    indices = {}

    def render_candidates(patch_box, patch_angle, layer_name):
        polygon_layer = layer_name in map_api.non_geometric_polygon_layers
        extract = map_api.extract_polygon if polygon_layer else map_api.extract_line
        if layer_name not in indices:
            records = getattr(map_api, layer_name)
            entries, bounds = [], []
            for i, record in enumerate(records):
                tokens = (record['polygon_tokens'] if layer_name == 'drivable_area'
                          else [record['polygon_token' if polygon_layer else 'line_token']])
                for token in tokens:
                    geometry = extract(token)
                    if geometry.is_empty:
                        continue
                    entries.append((i, token))
                    bounds.append(geometry.bounds)
            indices[layer_name] = (records, entries, np.asarray(bounds).reshape(-1, 4))
        records, entries, bounds = indices[layer_name]
        xmin, ymin, xmax, ymax = explorer.get_patch_coord(patch_box, patch_angle).bounds
        # Include touching boundaries and a tiny outward tolerance. Only
        # provably disjoint objects are removed; exact SDK clipping is kept.
        tolerance = 1e-7
        keep = ((bounds[:, 0] <= xmax + tolerance) & (bounds[:, 2] >= xmin - tolerance)
                & (bounds[:, 1] <= ymax + tolerance) & (bounds[:, 3] >= ymin - tolerance))
        selected = {}
        for (i, token), retain in zip(entries, keep):
            if retain:
                selected.setdefault(i, []).append(token)
        candidates = [dict(records[i], polygon_tokens=tokens) if layer_name == 'drivable_area'
                      else records[i] for i, tokens in selected.items()]
        setattr(map_api, layer_name, candidates)
        try:
            original = original_polygon if polygon_layer else original_line
            return original(patch_box, patch_angle, layer_name)
        finally:
            setattr(map_api, layer_name, records)

    explorer._get_layer_polygon = render_candidates
    explorer._get_layer_line = render_candidates
    map_api._layout_spatial_cache = True
    return map_api


class SpatialBEVLayoutRenderer(BEVLayoutRenderer):
    """Worker-local geometry cache with identical SDK rasterization."""

    def _get_map(self, location):
        return enable_spatial_map_cache(super()._get_map(location))
