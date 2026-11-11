"use strict";

const byId = id => document.getElementById(id);
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const viewers = [];

function createForecastViewer(config) {
  const $ = id => byId(config.prefix + id);
  const root = byId(config.rootId);
  const methods = config.methods;
  const metric = config.metric;
  let viewer;
  const preloadImages = new Map();
  let scenes = [];
  let scene;
  let frameIndex = 3;
  let timer = null;
  let view = "timeline";

  function stop() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
    $("play").innerHTML = `${icon("play")}<span>Play</span>`;
    $("play").setAttribute("aria-label", "Play all forecasts");
  }

  function play() {
    if (timer !== null) return stop();
    viewers.filter(instance => instance !== viewer).forEach(instance => instance.stop());
    if (frameIndex === scene.frames.length - 1) frameIndex = 0;
    updateFrame();
    $("play").innerHTML = `${icon("pause")}<span>Pause</span>`;
    $("play").setAttribute("aria-label", "Pause all forecasts");
    timer = window.setInterval(() => {
      const next = (frameIndex + 1) % scene.frames.length;
      if (next === 0 && !$("loop").checked) return stop();
      const ready = Object.values(scene.frames[next].images).every(path => {
        const img = preloadImages.get(path);
        return img?.complete && img.naturalWidth > 0;
      });
      if (!ready) return;
      frameIndex = next;
      updateFrame();
    }, 500);
  }

  function openLightbox(method, frame, sourceButton) {
    viewers.forEach(instance => instance.stop());
    byId("lightbox-image").src = frame.images[method.id];
    byId("lightbox-image").alt = `${method.label} ${config.noun}, ${scene.id}, +${frame.seconds} seconds`;
    byId("lightbox-title").textContent = `${method.label} · ${scene.id} · +${frame.seconds.toFixed(1)} s`;
    byId("lightbox-note").textContent = config.prefix
      ? "Shared camera, voxel grid, and semantic class palette."
      : "Shared camera, ego-coordinate crop, and height scale.";
    byId("lightbox").showModal();
    byId("lightbox").addEventListener("close", () => sourceButton.focus(), {once: true});
  }

  function imageButton(method, frame) {
    const button = document.createElement("button");
    button.className = "image-button";
    button.type = "button";
    button.setAttribute("aria-label", `Enlarge ${method.label}, +${frame.seconds} seconds`);
    const img = document.createElement("img");
    img.src = frame.images[method.id];
    img.alt = `${method.label} ${config.noun} in ${scene.id} at +${frame.seconds} seconds`;
    img.width = 960;
    img.height = 680;
    img.decoding = "async";
    const expand = document.createElement("span");
    expand.className = "expand-icon";
    expand.innerHTML = icon("expand");
    button.append(img, expand);
    button.addEventListener("click", () => openLightbox(method, frame, button));
    return button;
  }

  function renderPanels() {
    $("panels").replaceChildren();
    for (const method of methods) {
      const panel = document.createElement("article");
      panel.className = "cloud-card" + (method.id === "loworld" ? " ours" : "");
      // Markup uses the fixed method definitions above; scene data uses textContent.
      panel.innerHTML = `<div class="cloud-heading"><div><span class="method-dot ${method.dot}"></span><h4>${method.label}</h4></div><span class="${method.id === "loworld" ? "ours-label" : ""}">${method.detail}</span></div><div id="${config.prefix}panel-image-${method.id}"></div><div class="cloud-foot"><span id="${config.prefix}panel-count-${method.id}"></span><strong id="${config.prefix}panel-cd-${method.id}"></strong></div>`;
      $("panels").append(panel);
    }
  }

  function updateFrame() {
    const frame = scene.frames[frameIndex];
    $("time").value = frameIndex;
    $("time").style.setProperty("--progress", `${frameIndex / (scene.frames.length - 1) * 100}%`);
    $("time").setAttribute("aria-valuetext", `${frame.seconds} seconds into the future`);
    $("time-label").value = `+${frame.seconds.toFixed(1)} s`;
    for (const method of methods) {
      $("panel-image-" + method.id).replaceChildren(imageButton(method, frame));
      $("panel-count-" + method.id).textContent = `${frame[config.countKey][method.id].toLocaleString("en-US")} ${config.countLabel}`;
      $("panel-cd-" + method.id).textContent = method.id === "gt" ? "Reference" : `${metric.shortLabel} ${frame[metric.key][method.id].toFixed(metric.decimals)} ${metric.unit}`;
    }
    $("wipe-gt").replaceChildren(imageButton(methods[0], frame));
    $("wipe-before").src = frame.images[methods[1].id];
    $("wipe-after").src = frame.images.loworld;
    $("wipe-before").alt = `${methods[1].label} ${config.noun} at +${frame.seconds} s`;
    $("wipe-after").alt = `LO-World ${config.noun} at +${frame.seconds} s`;
    root.querySelectorAll("[data-seconds]").forEach(button => {
      button.setAttribute("aria-pressed", String(Number(button.dataset.seconds) === frame.seconds));
    });
  }

  function renderOverview() {
    const grid = $("overview-panel");
    grid.replaceChildren(document.createElement("span"));
    const frames = [2, 4, 6].map(time => scene.frames.find(frame => frame.seconds === time));
    for (const frame of frames) {
      const label = document.createElement("div");
      label.className = "overview-time";
      label.textContent = `+${frame.seconds.toFixed(1)} seconds`;
      grid.append(label);
    }
    for (const method of methods) {
      const label = document.createElement("div");
      label.className = "overview-label" + (method.id === "loworld" ? " ours-label" : "");
      label.textContent = method.label;
      grid.append(label);
      for (const frame of frames) grid.append(imageButton(method, frame));
    }
  }

  function renderMetrics() {
    $("metrics-scene-label").textContent = `nuScenes · ${scene.id}`;
    const tbody = $("metrics-body");
    tbody.replaceChildren();
    const forecastMethods = methods.filter(method => method.id !== "gt");
    const rows = [2, 4, 6].map(time => ({label: `+${time}.0 s`, values: scene.frames.find(frame => frame.seconds === time)[metric.key]}));
    rows.push({label: "Future mean", values: Object.fromEntries(forecastMethods.map(({id}) => [id, scene.frames.reduce((sum, frame) => sum + frame[metric.key][id], 0) / scene.frames.length]))});
    for (const row of rows) {
      const tr = document.createElement("tr");
      for (const value of [row.label, ...forecastMethods.map(({id}) => row.values[id].toFixed(metric.decimals))]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.append(cell);
      }
      tbody.append(tr);
    }
    const w = 580, h = 300, left = 38, right = 14, top = 14, bottom = 35;
    const max = Math.max(...scene.frames.flatMap(frame => Object.values(frame[metric.key])));
    const ceiling = metric.ceiling || Math.max(1, Math.ceil(max * 1.08));
    const x = seconds => left + (seconds - .5) / 5.5 * (w - left - right);
    const y = value => h - bottom - value / ceiling * (h - top - bottom);
    const titleId = config.prefix + "chart-title";
    const descriptionId = config.prefix + "chart-desc";
    const gradientId = config.prefix + "area-gradient";
    let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="${titleId} ${descriptionId}"><title id="${titleId}">${metric.accessibleTitle}</title><desc id="${descriptionId}">${metric.description} Exact values for two, four and six seconds and the future mean appear in the adjacent table; all values are in the linked scene data.</desc><defs><linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#416ddd" stop-opacity=".09"/><stop offset="100%" stop-color="#416ddd" stop-opacity="0"/></linearGradient></defs>`;
    for (let tick = 0; tick <= 4; tick++) {
      const value = ceiling * tick / 4;
      svg += `<line x1="${left}" x2="${w-right}" y1="${y(value)}" y2="${y(value)}" stroke="#edf0f6" stroke-dasharray="3 4"/><text x="${left-10}" y="${y(value)+4}" text-anchor="end" fill="#a4aec0" font-size="10">${metric.ceiling ? value.toFixed(0) : value.toFixed(1)}</text>`;
    }
    for (const seconds of [.5, 2, 4, 6]) svg += `<text x="${x(seconds)}" y="${h-12}" text-anchor="middle" fill="#a4aec0" font-size="10">+${seconds} s</text>`;
    const ours = scene.frames.map(frame => `${x(frame.seconds)},${y(frame[metric.key].loworld)}`).join(" ");
    svg += `<polygon points="${x(.5)},${y(0)} ${ours} ${x(6)},${y(0)}" fill="url(#${gradientId})"/>`;
    for (const method of forecastMethods) {
      const color = method.id === "loworld" ? "#416ddd" : "#b795af";
      const points = scene.frames.map(frame => `${x(frame.seconds)},${y(frame[metric.key][method.id])}`).join(" ");
      svg += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>`;
      for (const frame of scene.frames) svg += `<circle cx="${x(frame.seconds)}" cy="${y(frame[metric.key][method.id])}" r="3" fill="white" stroke="${color}" stroke-width="1.7"><title>${method.label}, +${frame.seconds} s: ${frame[metric.key][method.id].toFixed(metric.decimals)} ${metric.unit}</title></circle>`;
    }
    svg += `</svg><div class="chart-legend"><span><i></i>${forecastMethods[0].label}</span><span><i class="loworld-key"></i>LO-World · ${methods[2].detail}</span></div>`;
    $("chart").innerHTML = svg;
  }

  function renderSemanticLegend(data) {
    const legend = $("semantic-legend");
    if (!legend) return;
    legend.replaceChildren();
    for (const semanticClass of data.classes) {
      if (semanticClass.id === data.protocol.free_class) continue;
      const entry = document.createElement("span");
      entry.className = "semantic-key";
      const swatch = document.createElement("i");
      swatch.className = "semantic-swatch";
      if (!/^#[0-9a-f]{6}$/i.test(semanticClass.color)) throw new Error("Invalid semantic color");
      swatch.style.backgroundColor = semanticClass.color;
      const label = document.createElement("span");
      label.textContent = semanticClass.name.replaceAll("_", " ");
      entry.append(swatch, label);
      legend.append(entry);
    }
  }

  function selectView(selected) {
    stop();
    view = selected;
    for (const name of ["timeline", "wipe", "overview"]) {
      $(name + "-panel").hidden = view !== name;
      $(name + "-view").setAttribute("aria-pressed", String(view === name));
    }
    $("playback-controls").hidden = view === "overview";
    $("viewer-help").innerHTML = view === "wipe"
      ? "Drag the divider to compare the two forecasts"
      : `Click any ${config.noun} to enlarge ${icon("expand")}`;
  }

  function selectScene(selected) {
    stop();
    scene = selected;
    frameIndex = 3;
    $("scene-label").textContent = scene.id.replace("scene-", "Scene ");
    const position = scenes.findIndex(item => item.id === scene.id) + 1;
    $("scene-position").textContent = `${String(position).padStart(2, "0")} / ${String(scenes.length).padStart(2, "0")}`;
    $("scene-position").hidden = scenes.length < 2;
    root.querySelectorAll(".scene-tile").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.scene === scene.id));
    });
    $("time").max = scene.frames.length - 1;
    renderPanels();
    updateFrame();
    renderOverview();
    renderMetrics();
    preloadImages.clear();
    for (const frame of scene.frames) for (const path of Object.values(frame.images)) {
      const img = new Image();
      img.decoding = "async";
      img.addEventListener("error", () => {
        if (selected !== scene) return;
        stop();
        $("load-status").textContent = "A comparison image could not be loaded. Please reload to retry.";
        $("load-status").hidden = false;
      });
      img.src = path;
      preloadImages.set(path, img);
    }
  }

  function renderScenePicker() {
    $("scene-picker").replaceChildren();
    $("scene-picker").hidden = scenes.length < 2;
    scenes.forEach((example, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "scene-tile";
      button.dataset.scene = example.id;
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", `${example.id.replace("scene-", "Scene ")} · ${example.title || config.title}`);
      button.title = example.description || example.label;
      const preview = document.createElement("img");
      preview.src = example.preview || example.frames[3].images.loworld;
      preview.alt = "";
      preview.width = 288;
      preview.height = 204;
      preview.decoding = "async";
      const caption = document.createElement("span");
      caption.className = "scene-tile-caption";
      const number = document.createElement("span");
      number.className = "scene-tile-number";
      number.textContent = String(index + 1).padStart(2, "0");
      const label = document.createElement("span");
      label.className = "scene-tile-title";
      label.textContent = example.title || example.id.replace("scene-", "Scene ");
      const sceneId = document.createElement("span");
      sceneId.className = "scene-tile-id";
      sceneId.textContent = example.selection
        ? `${example.id} · +${example.selection.gap_pp.toFixed(1)} pp`
        : example.id;
      if (example.selection) sceneId.title = "LO-World minus I²-World, mean semantic mIoU across all 12 future frames";
      caption.append(number, label, sceneId);
      button.append(preview, caption);
      button.addEventListener("click", () => selectScene(example));
      $("scene-picker").append(button);
    });
  }

  $("play").addEventListener("click", play);
  $("time").addEventListener("input", () => { stop(); frameIndex = Number($("time").value); updateFrame(); });
  root.querySelectorAll("[data-seconds]").forEach(button => button.addEventListener("click", () => {
    stop(); frameIndex = scene.frames.findIndex(frame => frame.seconds === Number(button.dataset.seconds)); updateFrame();
  }));
  for (const selected of ["timeline", "wipe", "overview"]) $(selected + "-view").addEventListener("click", () => selectView(selected));
  $("wipe-range").addEventListener("input", () => $("wipe-stage").style.setProperty("--reveal", `${$("wipe-range").value}%`));

  viewer = {stop};
  viewers.push(viewer);
  fetch(config.dataUrl).then(response => {
    if (!response.ok) throw new Error(`Comparison data: HTTP ${response.status}`);
    return response.json();
  }).then(data => {
    scenes = data.scenes;
    if (!scenes?.length) throw new Error("No comparison scenes available");
    for (const example of scenes) {
      if (example.frames.length !== 12) throw new Error("Incomplete forecast");
      example.frames.forEach((frame, index) => {
        if (frame.seconds !== (index+1)/2) throw new Error("Unexpected forecast timestamps");
        for (const {id} of methods) if (!frame.images[id] || !(frame[config.countKey][id] > 0)) throw new Error("Missing point cloud");
        for (const id of methods.filter(method => method.id !== "gt").map(method => method.id)) if (!Number.isFinite(frame[metric.key][id]) || frame[metric.key][id] < 0) throw new Error("Invalid scene metric");
      });
    }
    if (config.onData) config.onData(data, methods);
    renderSemanticLegend(data);
    renderScenePicker();
    selectScene(scenes[0]);
    $("comparison").hidden = false;
    byId(config.measurementsId || config.prefix + "measurements").hidden = false;
    $("load-status").hidden = true;
    root.hidden = false;
    if (config.navId) byId(config.navId).hidden = false;
  }).catch(error => {
    console.error(error);
    $("load-status").textContent = "The comparison could not be loaded. Please reload the page or view the linked scene data.";
  });

}

createForecastViewer({
  prefix: "", rootId: "forecasting", dataUrl: "assets/comparison.json", title: "LiDAR forecasting", measurementsId: "lidar-scene-measurements",
  noun: "point cloud", countKey: "point_counts", countLabel: "points",
  methods: [
    {id: "gt", label: "Ground truth", detail: "Observation", dot: "gt-dot"},
    {id: "opendwm", label: "OpenDWM", detail: "Temporal MaskGIT", dot: "baseline-dot"},
    {id: "loworld", label: "LO-World", detail: "Layout condition", dot: "ours-dot"},
  ],
  metric: {key: "cd", shortLabel: "CD", unit: "m²", decimals: 3,
    accessibleTitle: "Scene Chamfer distance over the forecast horizon",
    description: "Symmetric mean squared Chamfer distance in square meters. Lower is better."},
});

createForecastViewer({
  prefix: "occ-", rootId: "occ-forecasting", dataUrl: "assets/occupancy.json", navId: "occ-nav", title: "Occupancy forecasting",
  noun: "occupancy volume", countKey: "voxel_counts", countLabel: "occupied voxels",
  methods: [
    {id: "gt", label: "Ground truth", detail: "Occ3D", dot: "gt-dot"},
    {id: "i2world", label: "I²-World", detail: "Intra/inter tokenizer", dot: "baseline-dot"},
    {id: "loworld", label: "LO-World", detail: "No layout", dot: "ours-dot"},
  ],
  metric: {key: "miou", shortLabel: "mIoU", unit: "%", decimals: 1, ceiling: 100,
    accessibleTitle: "Scene semantic mIoU over the forecast horizon",
    description: "Mean intersection over union across occupied semantic classes 0–16, as a percentage. Higher is better. Free class 17 is excluded from the class mean and GT label 255 is ignored. Classes with zero union are omitted; zero IoU with nonzero union is retained."},
  onData(data, methods) {
    if (data.protocol.loworld_variant !== "layout") return;
    methods[2].detail = "Layout condition";
    byId("occ-model-variant").textContent = "12-frame autoregressive, BEVLayout-conditioned";
    byId("occ-loworld-conditions").textContent = "Future BEV layouts and ego poses";
    byId("occ-condition-note").textContent = "LO-World additionally receives future BEV layouts in this comparison; I²-World receives ego-motion conditions. Their history representations differ. I²-World uses its original three-second checkpoint rolled out autoregressively to six seconds.";
    byId("occ-config-link").href = "https://github.com/lzzzzzm/LO-World/blob/main/configs/world_model/bev_world_12f_layout_autoregressive.py";
  },
});

byId("close-lightbox").addEventListener("click", () => byId("lightbox").close());
byId("lightbox").addEventListener("click", event => {
  if (event.target !== byId("lightbox")) return;
  const rect = byId("lightbox").getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) byId("lightbox").close();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) viewers.forEach(viewer => viewer.stop());
});
byId("copy-citation").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(byId("bibtex").textContent);
    byId("copy-citation").innerHTML = `${icon("copy")}<span>Copied</span>`;
    byId("copy-status").textContent = "Citation copied to clipboard.";
    window.setTimeout(() => { byId("copy-citation").innerHTML = `${icon("copy")}<span>Copy</span>`; }, 2000);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(byId("bibtex"));
    const selection = window.getSelection();
    selection.removeAllRanges(); selection.addRange(range);
    byId("copy-status").textContent = "Select and copy the highlighted citation.";
    byId("copy-citation").innerHTML = `${icon("copy")}<span>Select & copy</span>`;
  }
});
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      const target = entry.target.id === "occ-measurements" ? "occ-forecasting" : entry.target.id === "lidar-scene-measurements" ? "forecasting" : entry.target.id;
      document.querySelectorAll(".nav-link").forEach(link => link.classList.toggle("active", link.getAttribute("href") === `#${target}`));
    }
  }, {rootMargin: "-10% 0px -60% 0px", threshold: 0});
  for (const id of ["paper-title", "introduction", "forecasting", "occ-forecasting", "trajectory-control", "fullmodal-generation", "measurements", "lidar-scene-measurements", "occ-measurements", "citation"]) observer.observe(byId(id));
}
