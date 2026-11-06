"""Shared-palette GIF output without Open3D or model dependencies."""

from pathlib import Path

from PIL import Image

from tools.genie.occ_generation.scene_runtime import atomic_file


def frame_image(source):
    if isinstance(source, Image.Image):
        return source.copy()
    return Image.open(source) if isinstance(source, (str, Path)) else Image.fromarray(source)


def encode_gif(frame_paths, output, fps, *, palette=None):
    """Encode one palette for the sequence and preserve mean FPS in centiseconds."""
    if len(frame_paths) == 0 or not 1 <= fps <= 100:
        raise ValueError('GIF requires frames and FPS between 1 and 100')
    if palette is None:
        columns = 9
        thumb_size = (288, 129)
        atlas = Image.new('RGB', (columns * thumb_size[0],
                                  ((len(frame_paths) + columns - 1) // columns) *
                                  thumb_size[1]), (255, 255, 255))
        for i, path in enumerate(frame_paths):
            with frame_image(path) as frame:
                atlas.paste(frame.resize(thumb_size, Image.Resampling.LANCZOS),
                            ((i % columns) * thumb_size[0],
                             (i // columns) * thumb_size[1]))
        palette = atlas.quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    def quantized_frames():
        for path in frame_paths:
            with frame_image(path) as frame:
                yield frame.convert('RGB').quantize(palette=palette, dither=Image.Dither.NONE)

    durations = [10 * (round((i + 1) * 100 / fps) - round(i * 100 / fps))
                 for i in range(len(frame_paths))]
    frames = quantized_frames()
    first = next(frames)
    with atomic_file(output) as stream:
        first.save(stream, format='GIF', save_all=True, append_images=frames, loop=0,
                   duration=durations, disposal=1, optimize=False)
    with Image.open(output) as gif:
        if gif.n_frames != len(frame_paths):
            raise RuntimeError(f'GIF contains {gif.n_frames} of {len(frame_paths)} frames')
        total_ms = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            gif.load()
            total_ms += gif.info['duration']
        return dict(frame_count=gif.n_frames, size=list(gif.size),
                    duration_ms=total_ms, target_fps=fps,
                    effective_fps=gif.n_frames * 1000 / total_ms,
                    palette='shared across all frames', loop=gif.info['loop'])
