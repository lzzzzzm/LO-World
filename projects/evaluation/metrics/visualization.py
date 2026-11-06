import hashlib


def visualization_sample_index(data_samples, get_sample_meta, fallback):
    """Return a stable index used to select metric visualizations."""
    for data_sample in data_samples:
        if data_sample is None:
            continue
        for key in ('sample_index', 'sample_idx'):
            sample_index = get_sample_meta(data_sample, key)
            if sample_index is not None:
                if hasattr(sample_index, 'item'):
                    sample_index = sample_index.item()
                try:
                    return int(sample_index)
                except (TypeError, ValueError):
                    pass

    for data_sample in data_samples:
        if data_sample is None:
            continue
        for key in ('token', 'scene_name'):
            value = get_sample_meta(data_sample, key)
            if value is not None:
                digest = hashlib.sha1(str(value).encode('utf-8')).digest()
                return int.from_bytes(digest[:8], byteorder='big')

    return int(fallback)


def should_visualize(data_samples, get_sample_meta, interval, fallback):
    if interval <= 0:
        return False, int(fallback)
    sample_index = visualization_sample_index(
        data_samples, get_sample_meta, fallback)
    return sample_index % interval == 0, sample_index


def visualization_save_name(data_sample, get_sample_meta, sample_index):
    """Build the canonical 3D/4D metric visualization filename."""
    scene_name = get_sample_meta(data_sample, 'scene_name')
    return '{}_{}'.format(scene_name, str(sample_index).zfill(6))
