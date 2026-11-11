# Project website

## Prerequisites

The project website is a static site in `docs/`. It requires no frontend build
or package installation. Use the `loworld` Conda environment for local preview.

The LiDAR comparison includes five nuScenes validation scenes, each with 12
future frames at 2 Hz. It displays ground truth, layout-conditioned LO-World, and
OpenDWM Temporal MaskGIT with a shared camera, crop, and height scale. The
displayed Chamfer distances are single-scene measurements against raw LiDAR,
not validation-set benchmark scores. The methods' input conditions and history
lengths are listed on the page.

The occupancy comparison shows five examples selected for the largest
LO-World minus I²-World future-mean semantic mIoU gaps among 139 eligible
nuScenes validation scenes. Each candidate uses one complete six-second
window with its current frame nearest index 21. Both models use the same
current and future tokens within each example. The selected cases are
scene-0270, scene-0554, scene-0016, scene-0797, and scene-1068; these are
illustrative selections rather than validation-set benchmark scores.
The viewer displays Occ3D ground truth, I²-World, and LO-World with a common 0.4 m grid,
camera, and semantic class palette. The OCC comparison uses the no-layout
LO-World checkpoint; the LiDAR comparison uses the layout-conditioned checkpoint.
I²-World's original three-second checkpoint is rolled out autoregressively
to six seconds. Their history inputs and condition differences are listed
under the occupancy evaluation details.
Thumbnail labels show the difference in mean semantic mIoU across all 12
future frames, in percentage points (pp).

The controlled-generation viewer includes three starting scenes, each with
straight, left-turn, and right-turn ego trajectories. The layout-conditioned
checkpoint generates 81 frames at 12 Hz: the shared current frame and 80
new future frames over 6.67 seconds. Ego poses and instance tracks are
resampled from raw annotations at uniform physical timestamps before
constructing the layouts.
Within each scene, all three routes share the current latent, four-frame
memory, checkpoint, seed, and LiDAR rig. Future ego poses and their matching
BEV layouts change together. The layouts use the same map and future actor
world trajectories, reprojected into each candidate ego frame.

## Commands

From the repository root:

```bash
conda run --no-capture-output -n loworld python tools/serve_project_site.py --port 8000
```

For GitHub Pages, include `docs/index.html`, `docs/.nojekyll`, and `docs/assets/`
when pushing the website to GitHub, including the bundled font files and their
licenses. In the repository's **Settings → Pages**,
select **Deploy from a branch**, branch **main**, folder **/docs**, and save.

## Output

The local preview is available at `http://localhost:8000`. On a remote server,
forward port 8000 to the machine running your browser.

After GitHub Pages finishes deploying, its live URL appears in
**Settings → Pages**. The page includes the paper title, authors, resource
links, a short paper introduction before the visualizations, and a copyable
BibTeX citation. **Results** shows the published paper’s Tables 1–5, with
tabs for occupancy, LiDAR, long-horizon forecasting, Waymo and video.
The quality-and-speed overview above the tables plots the published 1–3 second
averages and throughput. Choose an occupancy metric or a comparison baseline
to update the highlighted gain. Occupancy uses LO-World-T; LiDAR uses
LO-World-TBM, with each model's inputs and future conditions shown. Occupancy
model FPS excludes decoding, while LiDAR FPS-E2E includes decoding. Missing
baseline FPS remains unreported. These inference speeds are separate from the
12 FPS visualization playback rate.
Input conditions, model variants, all table values and units follow the paper.
Each forecasting gallery keeps its selected-scene curves under **Scene
measurements & setup**. LiDAR comparisons support synchronized
playback, frame selection, a draggable OpenDWM/LO-World divider, a 2/4/6-second
grid, enlarged point clouds, and scene measurements. Evaluation settings can
be expanded below the measurements. Select a scene using the five thumbnail
buttons; the point clouds, time grid, and measurements update together.
The **Occupancy** navigation link opens the OCC comparison. Its scene controls
and time axis are independent of the LiDAR viewer. The displayed semantic mIoU
is the unweighted mean of class IoUs for occupied semantic classes 0–16.
Free class 17 is excluded from the class mean, and GT label
255 is ignored. Classes with zero union are omitted; a zero IoU with nonzero
union remains in the mean. Future mean averages the 12 future-frame mIoUs.
These are selected-scene measurements rather than a validation-set benchmark.
The **Control** link opens the trajectory viewer. Select a starting scene,
view **Occupancy + LiDAR** together or inspect **Layout condition**, and use
the shared timeline to play all three routes together at 12 FPS. Each route
shows occupancy above LiDAR at the same timestamp and camera. Lightweight
WebP frame atlases are buffered before playback and drawn using a shared
animation clock; only nearby atlases remain decoded. **Shared start** shows
the identical current reconstruction before the routes diverge. The path
diagram uses the initial ego frame; generated images use each candidate ego
frame with a shared camera. Alternative ego poses do not have matched
ground-truth futures.

**All modes** opens three complete multimodal examples. Six generated RGB
camera views, six semantic camera projections, occupancy and LiDAR share one
12 FPS timeline across 81 frames. LO-World predicts occupancy and LiDAR;
semantic views projected from that occupancy condition downstream GenieDrive
RGB generation. Every displayed modality uses the same prediction source and
physical timestamps. Select a scene, scrub the timeline, or click any view
to enlarge it.

Rendered images live in `docs/assets/lidar/`. Scene entries, image paths,
timestamps, point counts, and measurements live in `docs/assets/comparison.json`.
Additional scenes use the same `scenes` array and twelve-frame structure. Each
entry also supplies a title, a description, and a thumbnail path. Keep full
predictions, datasets, checkpoints, and videos outside the tracked website.

OCC images live in `docs/assets/occupancy/`; their scene entries, occupied-voxel
counts, semantic mIoU values, per-class IoUs, and semantic legend live in
`docs/assets/occupancy.json`.

Controlled-generation images live in `docs/assets/control12hz/`. Route paths,
frame measurements, atlas paths, shared-start images, physical timestamps,
and generation settings live in `docs/assets/control12hz.json`.
Multimodal frame atlases and scene data live in `docs/assets/fullmodal/` and
`docs/assets/fullmodal.json`. Published table cells and paper source links
are also available in `docs/assets/paper-results.json`.
