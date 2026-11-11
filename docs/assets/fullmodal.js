"use strict";

function createFullModalViewer() {
  const $ = name => byId("full-"+name);
  const root = byId("fullmodal-generation");
  const names = {rgb:"Generated RGB",semantic:"Semantic projection",occ:"Occupancy",lidar:"LiDAR"};
  const cameras = {CAM_FRONT_LEFT:"Front left",CAM_FRONT:"Front",CAM_FRONT_RIGHT:"Front right",CAM_BACK_LEFT:"Rear left",CAM_BACK:"Rear",CAM_BACK_RIGHT:"Rear right"};
  let scenes, scene, frameIndex = 24;
  const player = createAtlasPlayback({
    fps:12,frameCount:81,
    onFrame(index) {frameIndex=index;update();},
    onPlayback(playing,buffering) {
      $("play").innerHTML=`${icon(playing ? "pause" : "play")}<span>${playing ? buffering ? "Loading…" : "Pause" : "Play"}</span>`;
      $("play").setAttribute("aria-label",playing ? "Pause all modalities" : "Play all modalities");
    },
    onStatus(message) {$("status").textContent=message;},
    onError(error) {console.error(error);},
  });
  player.loop=() => $("loop").checked;
  const viewer={stop:() => player.pause()};
  viewers.push(viewer);

  function enlarge(stream,canvas,button) {
    viewers.forEach(instance => instance.stop());
    const title=`${stream.camera ? cameras[stream.camera]+" · " : ""}${names[stream.kind]} · ${scene.id} · +${(frameIndex/12).toFixed(2)} s`;
    byId("lightbox-image").src=canvas.toDataURL("image/png");
    byId("lightbox-image").alt=title;
    byId("lightbox-title").textContent=title;
    byId("lightbox-note").textContent=stream.kind === "rgb" ? "Six-camera RGB generated with GenieDrive from predicted-occupancy semantic conditions."
      : stream.kind === "semantic" ? "Semantic camera view projected from the paired occupancy prediction."
      : "Paired LO-World occupancy and LiDAR predictions at the same physical timestamp.";
    byId("lightbox").showModal();
    byId("lightbox").addEventListener("close",() => button.focus(),{once:true});
  }

  function renderViews() {
    $("cameras").replaceChildren();$("semantics").replaceChildren();$("geometry").replaceChildren();
    for (const stream of scene.streams) {
      const article=document.createElement("article");
      article.className=stream.camera ? "full-camera-card" : "cloud-card full-geometry-card";
      const heading=document.createElement("div");
      heading.className="full-view-heading";
      const name=document.createElement("span");
      name.textContent=stream.camera ? cameras[stream.camera] : names[stream.kind];
      heading.append(name);
      if (!stream.camera) {
        const count=document.createElement("span");count.id="full-count-"+stream.id;heading.append(count);
      }
      const button=document.createElement("button");
      button.type="button";button.className="image-button";button.id="full-image-"+stream.id;
      const canvas=document.createElement("canvas");
      canvas.width=stream.atlas.tile_width;canvas.height=stream.atlas.tile_height;
      canvas.dataset.kind=stream.kind;canvas.dataset.stream=stream.id;
      canvas.setAttribute("role","img");
      button.append(canvas);
      button.addEventListener("click",() => enlarge(stream,canvas,button));
      article.append(heading,button);
      $(stream.kind === "rgb" ? "cameras" : stream.kind === "semantic" ? "semantics" : "geometry").append(article);
    }
  }

  function update() {
    $("time").value=frameIndex;
    $("time").style.setProperty("--progress",`${frameIndex/80*100}%`);
    $("time").setAttribute("aria-valuetext",`${(frameIndex/12).toFixed(2)} seconds into the scene`);
    $("time-label").value=`+${(frameIndex/12).toFixed(2)} s`;
    $("status").textContent=`Frame ${frameIndex+1} / 81 · 12 Hz · All modalities synchronized`;
    for (const stream of scene.streams) {
      const button=$("image-"+stream.id),canvas=button.querySelector("canvas");
      const title=`${stream.camera ? cameras[stream.camera]+", " : ""}${names[stream.kind]}, ${scene.id}, +${(frameIndex/12).toFixed(2)} seconds`;
      player.draw(canvas,stream.id,frameIndex);
      canvas.dataset.frameIndex=frameIndex;
      canvas.setAttribute("aria-label",title);button.setAttribute("aria-label","Enlarge "+title);
      if (!stream.camera) {
        const frame=scene.frames[frameIndex];
        $("count-"+stream.id).textContent=stream.kind === "occ" ? `${frame.voxel_count.toLocaleString("en-US")} voxels` : `${frame.point_count.toLocaleString("en-US")} points`;
      }
    }
    root.querySelectorAll("[data-full-seconds]").forEach(button => button.setAttribute("aria-pressed",String(Number(button.dataset.fullSeconds) === frameIndex/12)));
  }

  function select(example) {
    player.pause();scene=example;frameIndex=24;
    $("scene-label").textContent=scene.id.replace("scene-","Scene ");
    $("scene-position").textContent=`${String(scenes.indexOf(scene)+1).padStart(2,"0")} / 03`;
    root.querySelectorAll(".scene-tile").forEach(button => button.setAttribute("aria-pressed",String(button.dataset.scene === scene.id)));
    renderViews();
    player.setStreams(scene.streams,frameIndex);
  }

  $("play").addEventListener("click",() => {
    if (!player.playing) viewers.filter(instance => instance !== viewer).forEach(instance => instance.stop());
    player.play();
  });
  $("time").addEventListener("input",() => player.seek(Number($("time").value)));
  root.querySelectorAll("[data-full-seconds]").forEach(button => button.addEventListener("click",() => player.seek(Math.round(Number(button.dataset.fullSeconds)*12))));

  fetch("assets/fullmodal.json").then(response => {
    if (!response.ok) throw new Error(`Multimodal data: HTTP ${response.status}`);
    return response.json();
  }).then(data => {
    scenes=data.scenes;
    if (scenes.length !== 3 || data.protocol.hz !== 12 || data.protocol.frame_count !== 81) throw new Error("Incomplete multimodal scenes");
    for (const example of scenes) {
      if (example.streams.length !== 14 || example.frames.length !== 81) throw new Error("Incomplete modality timeline");
      const button=document.createElement("button");
      button.type="button";button.className="scene-tile";button.dataset.scene=example.id;
      button.setAttribute("aria-label",`Choose multimodal ${example.id}`);
      const picture=document.createElement("img");picture.src=example.preview;picture.alt="";picture.width=288;picture.height=204;
      const caption=document.createElement("span");caption.className="scene-tile-caption";
      for (const [suffix,text] of [["number",String(scenes.indexOf(example)+1).padStart(2,"0")],["title",example.title],["id",example.id]]) {
        const label=document.createElement("span");label.className="scene-tile-"+suffix;label.textContent=text;caption.append(label);
      }
      button.append(picture,caption);button.addEventListener("click",() => select(example));$("scene-picker").append(button);
    }
    select(scenes[0]);$("content").hidden=false;$("load-status").hidden=true;
    root.hidden=false;byId("full-nav").hidden=false;
  }).catch(error => {console.error(error);$("load-status").textContent="The multimodal examples could not be loaded. Please reload the page.";});
}

createFullModalViewer();
