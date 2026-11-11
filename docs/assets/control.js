"use strict";

function createControlledViewer() {
  const root = byId("trajectory-control");
  const $ = name => byId("control-" + name);
  const labels = {occ: "Occupancy", lidar: "LiDAR", layout: "Layout condition"};
  const representations = () => mode === "joint" ? ["occ", "lidar"] : ["layout"];
  const routeLabels = {straight: "Straight", left: "Left turn", right: "Right turn"};
  const svgNS = "http://www.w3.org/2000/svg";
  let scenes, scene, mode = "joint", frameIndex = 24;
  let fps = 12, frameCount = 81, atlas;
  let diagramPositions = {};

  const player = createAtlasPlayback({
    fps:12, frameCount:81,
    onFrame(index) {frameIndex=index;update();},
    onPlayback(playing, buffering) {
      $("play").innerHTML = `${icon(playing ? "pause" : "play")}<span>${playing ? buffering ? "Loading…" : "Pause" : "Play"}</span>`;
      $("play").setAttribute("aria-label", playing ? "Pause all trajectories" : "Play all trajectories");
    },
    onStatus(message) {$("buffer-status").textContent=message;},
    onError(error) {console.error(error);},
  });
  player.loop = () => $("loop").checked;
  function stop() {player.pause();}
  const viewer = {stop};
  viewers.push(viewer);

  function setStreams() {
    return player.setStreams(scene.routes.flatMap(route => representations().map(kind => ({
      id:`${route.id}-${kind}`,urls:route.atlases[kind],atlas,
    }))),frameIndex);
  }

  function enlarge(path, title, button, representation) {
    viewers.forEach(instance => instance.stop());
    byId("lightbox-image").src = path;
    byId("lightbox-image").alt = title;
    byId("lightbox-title").textContent = title;
    byId("lightbox-note").textContent = representation === "layout"
      ? "BEV layout supplied to the model at this candidate ego pose."
      : "LO-World generation in the candidate ego frame; shared camera and spatial extent.";
    byId("lightbox").showModal();
    byId("lightbox").addEventListener("close", () => button.focus(), {once: true});
  }

  function svgElement(name, attributes, text) {
    const node = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text) node.textContent = text;
    return node;
  }

  function renderDiagram() {
    const points = scene.routes.flatMap(route => route.relative_xy.map(([x, y]) => [-y, -x]));
    const xs = points.map(point => point[0]), ys = points.map(point => point[1]);
    const lo = [Math.min(...xs) - 8, Math.min(...ys) - 8];
    const hi = [Math.max(...xs) + 8, Math.max(...ys) + 8];
    const scale = Math.min(300 / (hi[0] - lo[0]), 180 / (hi[1] - lo[1]));
    const project = ([x, y]) => [160 + (-y - (lo[0]+hi[0])/2)*scale, 97 + (-x - (lo[1]+hi[1])/2)*scale];
    const svg = svgElement("svg", {viewBox: "0 0 320 206", role: "img", "aria-label": "Three planned ego paths from the same starting pose"});
    for (let x = 20; x < 320; x += 20) svg.append(svgElement("line", {x1:x, x2:x, y1:8, y2:190, stroke:"#edf0f5"}));
    for (let y = 10; y < 195; y += 20) svg.append(svgElement("line", {x1:10, x2:310, y1:y, y2:y, stroke:"#edf0f5"}));
    diagramPositions = {};
    for (const route of scene.routes) {
      const projected = route.relative_xy.map(project);
      diagramPositions[route.id] = projected;
      svg.append(svgElement("polyline", {points:projected.map(point=>point.join(",")).join(" "), fill:"none", stroke:route.color, "stroke-width":2.5, "stroke-linejoin":"round", "stroke-linecap":"round"}));
      svg.append(svgElement("circle", {id:"control-cursor-"+route.id, r:4, fill:route.color, stroke:"white", "stroke-width":1.8}));
    }
    const origin = project([0, 0]);
    svg.append(svgElement("circle", {cx:origin[0], cy:origin[1], r:3, fill:"#36485f"}));
    svg.append(svgElement("text", {x:origin[0], y:origin[1]+14, "text-anchor":"middle", fill:"#667085", "font-size":9}, "Shared start"));
    svg.append(svgElement("line", {x1:18, x2:18+10*scale, y1:192, y2:192, stroke:"#8090a6", "stroke-width":1.6}));
    svg.append(svgElement("text", {x:18, y:204, fill:"#8090a6", "font-size":9}, "10 m"));
    $("diagram").replaceChildren(svg);
  }

  function renderCards() {
    $("panels").replaceChildren();
    for (const route of scene.routes) {
      const article = document.createElement("article");
      article.className = "cloud-card control-card";
      article.style.setProperty("--route-color", route.color);
      const heading = document.createElement("div");
      heading.className = "cloud-heading";
      const titleWrap = document.createElement("div");
      const dot = document.createElement("span");
      dot.className = "method-dot";
      dot.style.backgroundColor = route.color;
      const title = document.createElement("h4");
      title.textContent = routeLabels[route.id];
      titleWrap.append(dot, title);
      const badge = document.createElement("span");
      badge.id = "control-badge-" + route.id;
      heading.append(titleWrap, badge);
      const views = document.createElement("div");
      views.className = "control-route-views";
      for (const kind of representations()) {
        const section = document.createElement("section");
        section.className = "control-modality";
        section.dataset.representation = kind;
        const caption = document.createElement("div");
        caption.className = "control-modality-heading";
        const name = document.createElement("span");
        name.textContent = labels[kind];
        const count = document.createElement("span");
        count.id = `control-count-${route.id}-${kind}`;
        caption.append(name, count);
        const button = document.createElement("button");
        button.className = "image-button";
        button.type = "button";
        button.id = `control-image-${route.id}-${kind}`;
        const picture = document.createElement("canvas");
        picture.width = atlas.tile_width;
        picture.height = atlas.tile_height;
        picture.setAttribute("role","img");
        picture.dataset.representation = kind;
        button.append(picture);
        button.addEventListener("click", () => enlarge(picture.toDataURL("image/png"), `${routeLabels[route.id]} · ${labels[kind]} · ${scene.id} · +${(frameIndex/fps).toFixed(2)} s`, button, kind));
        section.append(caption, button);
        views.append(section);
      }
      const foot = document.createElement("div");
      foot.className = "cloud-foot";
      const caption = document.createElement("span");
      caption.textContent = mode === "joint" ? "Joint prediction · Same timestamp" : "BEV condition";
      const distance = document.createElement("strong");
      distance.textContent = `${route.length_m.toFixed(1)} m route`;
      foot.append(caption, distance);
      article.append(heading, views, foot);
      $("panels").append(article);
    }
  }

  function renderInitial() {
    $("initial").replaceChildren();
    $("initial").classList.toggle("control-initial-pair", mode === "joint");
    for (const kind of representations()) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "image-button control-initial-item";
      button.setAttribute("aria-label", `Enlarge shared starting ${labels[kind]}`);
      const image = document.createElement("img");
      image.id = `control-initial-${kind}`;
      image.src = scene.initial_images[kind];
      image.alt = `Shared starting ${labels[kind].toLowerCase()} in ${scene.id}`;
      image.width = 960;image.height = 680;
      const label = document.createElement("span");
      label.textContent = labels[kind];
      button.append(image, label);
      button.addEventListener("click", () => enlarge(scene.initial_images[kind], `Shared starting scene · ${labels[kind]} · ${scene.id}`, button, kind));
      $("initial").append(button);
    }
  }

  function update() {
    $("time").value = frameIndex;
    $("time").style.setProperty("--progress", `${frameIndex/(frameCount-1)*100}%`);
    $("time").setAttribute("aria-valuetext", `${(frameIndex/fps).toFixed(2)} seconds into the future`);
    $("time-label").value = `+${(frameIndex/fps).toFixed(2)} s`;
    $("buffer-status").textContent = `Frame ${frameIndex+1} / ${frameCount} · ${fps} Hz`;
    $("output-note").textContent = mode === "layout" ? "Input BEV layouts at the candidate ego poses" : "Occupancy above · LiDAR below · Shared camera and timestamp";
    for (const route of scene.routes) {
      for (const kind of representations()) {
        const button = $("image-"+route.id+"-"+kind);
        const picture = button.querySelector("canvas");
        const description = `${routeLabels[route.id]}, ${labels[kind]}, ${scene.id}, +${(frameIndex/fps).toFixed(2)} seconds`;
        picture.setAttribute("aria-label",description);
        button.setAttribute("aria-label", `Enlarge ${description}`);
        const sourceRoute = frameIndex === 0 ? scene.routes[0] : route;
        player.draw(picture,`${sourceRoute.id}-${kind}`,frameIndex);
        picture.dataset.frameIndex = frameIndex;
        picture.dataset.mode = mode;
        const frame = route.frames[frameIndex];
        $("count-"+route.id+"-"+kind).textContent = kind === "layout" ? "18 channels" : frameIndex === 0 ? "Shared start" : `${(kind === "occ" ? frame.voxel_count : frame.point_count).toLocaleString("en-US")} ${kind === "occ" ? "voxels" : "points"}`;
      }
      $("badge-"+route.id).textContent = mode === "layout" ? "Condition input" : "Layout-conditioned";
      const [x,y] = diagramPositions[route.id][frameIndex];
      $("cursor-"+route.id).setAttribute("cx",x);
      $("cursor-"+route.id).setAttribute("cy",y);
    }
    root.querySelectorAll("[data-control-seconds]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.controlSeconds) === frameIndex/fps)));
    root.querySelectorAll("[data-control-mode]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.controlMode === mode)));
  }

  function selectScene(example) {
    stop();
    scene = example;
    frameIndex = fps*2;
    $("time").max = frameCount-1;
    $("scene-label").textContent = scene.id.replace("scene-", "Scene ");
    $("scene-position").textContent = `${String(scenes.indexOf(scene)+1).padStart(2,"0")} / ${String(scenes.length).padStart(2,"0")}`;
    root.querySelectorAll(".scene-tile").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.scene === scene.id)));
    renderDiagram();
    renderInitial();
    renderCards();
    setStreams();
  }

  $("play").addEventListener("click", () => {
    if (!player.playing) viewers.filter(instance => instance !== viewer).forEach(instance => instance.stop());
    player.play();
  });
  $("time").addEventListener("input", () => {player.seek(Number($("time").value));});
  root.querySelectorAll("[data-control-seconds]").forEach(button => button.addEventListener("click", () => {player.seek(Math.round(Number(button.dataset.controlSeconds)*fps));}));
  root.querySelectorAll("[data-control-mode]").forEach(button => button.addEventListener("click", () => {
    if (mode === button.dataset.controlMode) return;
    stop();mode=button.dataset.controlMode;renderInitial();renderCards();setStreams();
  }));

  fetch("assets/control12hz.json").then(response => {
    if (!response.ok) throw new Error(`Control data: HTTP ${response.status}`);
    return response.json();
  }).then(data => {
    scenes = data.scenes;
    fps=data.protocol.hz;frameCount=data.protocol.frame_count;atlas=data.protocol.atlas;
    if (scenes.length !== 3 || !data.protocol.use_bev_layout || fps!==12 || frameCount!==81) throw new Error("Incomplete controlled generation");
    for (const example of scenes) {
      if (example.routes.length !== 3) throw new Error("Expected three route conditions");
      for (const route of example.routes) {
        if (route.frames.length !== frameCount || route.relative_xy.length !== frameCount) throw new Error("Incomplete route");
        if (Math.hypot(...route.relative_xy[0]) > 1e-6) throw new Error("Route does not share the current pose");
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = "scene-tile";
      button.dataset.scene = example.id;
      button.setAttribute("aria-label", `Choose ${example.id} shared starting scene`);
      const picture = document.createElement("img");
      picture.src = example.preview;
      picture.alt = "";
      picture.width = 288;picture.height = 204;
      const caption = document.createElement("span");
      caption.className = "scene-tile-caption";
      for (const [className,text] of [["number",String(scenes.indexOf(example)+1).padStart(2,"0")],["title",example.title],["id",example.id]]) {
        const span = document.createElement("span");span.className="scene-tile-"+className;span.textContent=text;caption.append(span);
      }
      button.append(picture,caption);
      button.addEventListener("click", () => selectScene(example));
      $("scene-picker").append(button);
    }
    selectScene(scenes[0]);
    $("content").hidden = false;
    $("load-status").hidden = true;
    root.hidden = false;
    byId("control-nav").hidden = false;
  }).catch(error => {
    console.error(error);
    $("load-status").textContent = "The trajectory examples could not be loaded. Please reload the page.";
  });
}

createControlledViewer();
