"use strict";

function createAtlasPlayback({fps, frameCount, onFrame, onPlayback, onStatus, onError}) {
  const decoded = new Map(), encoded = new Map();
  let streams = [], index = 0, timer = null, playing = false;
  let epoch = 0, revision = 0, intent = 0;
  let aborter = new AbortController();
  const key = (stream, chunk) => `${stream.id}:${chunk}`;
  const chunkAt = (stream, frame) => Math.floor(frame/stream.atlas.frames_per_atlas);

  function pause() {
    intent++;
    if (timer !== null) cancelAnimationFrame(timer);
    timer = null;playing = false;
    onPlayback(false, false);
  }

  function report(error, generation) {
    if (generation !== epoch || error.name === "AbortError") return;
    pause();onStatus("A frame could not be loaded. Reload to retry.");
    onError(error);
  }

  function blob(path) {
    if (!encoded.has(path)) encoded.set(path, fetch(path, {signal:aborter.signal}).then(response => {
      if (!response.ok) throw new Error(`Frame data: HTTP ${response.status}`);
      return response.blob();
    }));
    return encoded.get(path);
  }

  function entry(stream, chunk) {
    const id = key(stream, chunk);
    if (decoded.has(id)) return decoded.get(id);
    const generation = epoch;
    const record = {stream, chunk, bitmap:null, promise:null};
    record.promise = blob(stream.urls[chunk]).then(value => createImageBitmap(value)).then(bitmap => {
      if (generation !== epoch || decoded.get(id) !== record) {bitmap.close();return;}
      record.bitmap = bitmap;
    });
    decoded.set(id, record);
    return record;
  }

  const ready = frame => streams.every(stream => decoded.get(key(stream,chunkAt(stream,frame)))?.bitmap);
  const ensure = frame => Promise.all(streams.map(stream => entry(stream,chunkAt(stream,frame)).promise));
  const prefetch = () => Promise.all(streams.flatMap(stream => stream.urls.map(blob)));

  function keepNearby(frame) {
    const generation = epoch;
    for (const [id, record] of decoded) {
      const chunk = chunkAt(record.stream,frame);
      const next = (chunk+1)%record.stream.urls.length;
      if (record.chunk !== chunk && record.chunk !== next) {record.bitmap?.close();decoded.delete(id);}
    }
    Promise.all(streams.map(stream => entry(stream,(chunkAt(stream,frame)+1)%stream.urls.length).promise))
      .catch(error => report(error,generation));
  }

  async function render(frame) {
    const request = ++revision, generation = epoch;
    index = frame;
    if (!ready(frame)) onStatus("Loading frames…");
    try {
      await ensure(frame);
      if (request !== revision || generation !== epoch) return;
      onFrame(frame);keepNearby(frame);
      prefetch().catch(error => report(error,generation));
    } catch (error) {report(error,generation);}
  }

  async function play() {
    if (playing) {pause();return;}
    const request = ++intent, generation = epoch;
    playing = true;onPlayback(true,true);
    if (index === frameCount-1) index = 0;
    await render(index);
    if (request !== intent || generation !== epoch) return;
    try {
      await Promise.all([prefetch(),...streams.map(stream => entry(stream,(chunkAt(stream,index)+1)%stream.urls.length).promise)]);
    } catch (error) {report(error,generation);return;}
    if (request !== intent || generation !== epoch) return;
    onPlayback(true,false);
    let last = performance.now();
    const tick = now => {
      if (request !== intent || generation !== epoch) return;
      if (now-last >= 1000/fps) {
        const next = (index+1)%frameCount;
        if (next === 0 && !api.loop()) {pause();return;}
        if (ready(next)) {
          index = next;onFrame(index);keepNearby(index);
          last += 1000/fps;
          if (now-last > 1000/fps) last = now;
        } else {last = now;ensure(next).catch(error => report(error,generation));}
      }
      timer = requestAnimationFrame(tick);
    };
    timer = requestAnimationFrame(tick);
  }

  const api = {
    pause, play, loop:() => true,
    seek(frame) {pause();return render(Math.max(0,Math.min(frameCount-1,frame)));},
    setStreams(next, frame=0) {
      pause();epoch++;revision++;
      aborter.abort();aborter = new AbortController();
      for (const record of decoded.values()) record.bitmap?.close();
      decoded.clear();encoded.clear();streams = next;
      return render(frame);
    },
    draw(canvas, id, frame=index) {
      const stream = streams.find(item => item.id === id);
      const {atlas} = stream;
      const bitmap = decoded.get(key(stream,chunkAt(stream,frame))).bitmap;
      const within = frame%atlas.frames_per_atlas;
      canvas.getContext("2d").drawImage(bitmap,(within%atlas.columns)*atlas.tile_width,
        Math.floor(within/atlas.columns)*atlas.tile_height,atlas.tile_width,atlas.tile_height,
        0,0,canvas.width,canvas.height);
    },
    get index() {return index;},
    get playing() {return playing;},
  };
  return api;
}
