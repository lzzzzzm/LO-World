"use strict";

byId("architecture-expand").addEventListener("click", event => {
  viewers.forEach(viewer => viewer.stop());
  const thumbnail = byId("architecture-thumbnail");
  const dialog = byId("lightbox");
  byId("lightbox-image").src = thumbnail.currentSrc || thumbnail.src;
  byId("lightbox-image").alt = thumbnail.alt;
  byId("lightbox-title").textContent = "LO-World · Architecture overview";
  byId("lightbox-note").textContent = "Joint BEV encoding → conditional latent generation → occupancy & LiDAR decoding. sg: stop-gradient occupancy guidance.";
  dialog.classList.add("architecture-dialog");
  dialog.showModal();
  const source = event.currentTarget;
  dialog.addEventListener("close", () => {
    dialog.classList.remove("architecture-dialog");
    source.focus();
  }, {once: true});
});
