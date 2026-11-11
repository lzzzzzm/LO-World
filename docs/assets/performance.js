"use strict";

// Read the same published cells used by the complete result tables.
const performanceConfigs = [
  {
    id: "occupancy", title: "Semantic occupancy", variant: "LO-World-T",
    ours: "LO-World-T (Ours)", methods: ["OccWorld-O", "I²-World", "GenieDrive"],
    baseline: "GenieDrive", condition: "Ego trajectory condition", table: 1,
    metrics: [
      {id: "miou", label: "mIoU", column: 5, unit: "%", direction: "up", ceiling: 50},
      {id: "iou", label: "IoU", column: 9, unit: "%", direction: "up", ceiling: 60},
    ],
    speedColumn: 10, speedLabel: "Model FPS", speedCeiling: 50,
    speedNote: "Single RTX 4090 · Model FPS excludes decoding",
  },
  {
    id: "lidar", title: "LiDAR forecasting", variant: "LO-World-TBM",
    ours: "LO-World-TBM (Ours)", methods: ["OpenDWM-DiT", "LaGen"],
    baseline: "LaGen", condition: "Ego trajectory + box + map conditions", table: 2,
    metrics: [{id: "cd", label: "CD", column: 5, unit: "m²", direction: "down", ceiling: 3}],
    speedColumn: 9, speedLabel: "FPS-E2E", speedCeiling: 30,
    speedNote: "Single RTX 4090 · FPS-E2E includes decoding",
  },
];

const performanceEscape = value => String(value).replace(/[&<>"']/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);

function populatePaperHighlights(data) {
  const comparison = id => {
    const config = performanceConfigs.find(item => item.id === id);
    const table = data.tables.find(item => item.id === id);
    const row = name => table.rows.find(item => item[0].text === name);
    const column = config.metrics[0].column;
    return {ours: Number(row(config.ours)[column].text), baseline: Number(row(config.baseline)[column].text),
      fps: Number(row(config.ours)[config.speedColumn].text)};
  };
  const occupancy = comparison("occupancy");
  const lidar = comparison("lidar");
  const values = {
    miou: `+${(occupancy.ours - occupancy.baseline).toFixed(2)}`,
    cd: `−${((lidar.baseline - lidar.ours) / lidar.baseline * 100).toFixed(1)}`,
    fps: lidar.fps.toFixed(2),
  };
  for (const node of document.querySelectorAll("[data-paper-highlight]")) {
    node.textContent = values[node.dataset.paperHighlight];
  }
  byId("paper-highlights").setAttribute("aria-busy", "false");
}

function createPerformanceCard(config, table) {
  const rows = new Map(table.rows.map(row => [row[0].text, row]));
  const ours = rows.get(config.ours);
  const card = document.createElement("article");
  card.className = "performance-card";
  card.id = `performance-${config.id}`;
  card.setAttribute("aria-labelledby", `${card.id}-title`);
  let baseline = config.baseline;
  let metric = config.metrics[0];

  card.innerHTML = `
    <div class="performance-heading"><div><span class="performance-overline">${config.condition}</span><h3 id="${card.id}-title">${config.title}</h3></div><span class="performance-variant">${config.variant}</span></div>
    <div class="performance-controls">
      <label>Metric <select class="performance-metric" aria-label="${config.title} metric">${config.metrics.map(item => `<option value="${item.id}">${item.label} ${item.direction === "up" ? "↑" : "↓"}</option>`).join("")}</select></label>
      <label>Compare with <select class="performance-baseline" aria-label="${config.title} comparison baseline">${config.methods.map(name => `<option${name === baseline ? " selected" : ""}>${performanceEscape(name)}</option>`).join("")}</select></label>
    </div>
    <div class="performance-highlights" aria-live="polite" aria-atomic="true"></div>
    <div class="performance-chart"></div>
    <div class="performance-footer"><span>${config.speedNote}</span><a href="${table.source_url}">Table ${config.table} ↗</a></div>`;

  const numberAt = (row, index) => {
    const text = row[index].text;
    return text === "-" ? null : Number(text);
  };

  function render() {
    const selected = rows.get(baseline);
    const value = numberAt(ours, metric.column);
    const reference = numberAt(selected, metric.column);
    const absolute = value - reference;
    const relative = (metric.direction === "up" ? absolute : -absolute) / reference * 100;
    const speed = numberAt(ours, config.speedColumn);
    const referenceSpeed = numberAt(selected, config.speedColumn);
    const gain = metric.direction === "up" ? `+${absolute.toFixed(2)}` : `−${relative.toFixed(1)}`;
    const gainUnit = metric.direction === "up" ? "pp" : "%";
    const gainDetail = metric.direction === "up" ? `+${relative.toFixed(1)}% relative ${metric.label}` : `${metric.label} reduction`;
    const ratio = referenceSpeed === null ? "Baseline FPS not reported" : `${(speed / referenceSpeed).toFixed(2)}× baseline speed`;

    card.querySelector(".performance-highlights").innerHTML = `
      <div class="performance-gain"><strong data-performance-gain>${gain}<small>${gainUnit}</small></strong><span>${gainDetail} vs ${performanceEscape(baseline)}</span></div>
      <div class="performance-speed"><strong data-performance-fps>${speed.toFixed(2)}<small>${config.speedLabel}</small></strong><span>${ratio}</span></div>`;

    const chart = card.querySelector(".performance-chart");
    chart.setAttribute("aria-label", `${config.title}: average ${metric.label} and ${config.speedLabel}. ${config.speedNote}.`);
    chart.innerHTML = `
      <div class="performance-chart-head"><span>Method / observed input</span><span>Avg. ${metric.label} ${metric.unit} ${metric.direction === "up" ? "↑" : "↓"}</span><span>${config.speedLabel} ↑</span></div>
      ${[...config.methods, config.ours].map(name => {
        const row = rows.get(name);
        const quality = numberAt(row, metric.column);
        const fps = numberAt(row, config.speedColumn);
        const isOurs = name === config.ours;
        const input = ({O: "Occupancy", L: "LiDAR", "O&L": "Occupancy + LiDAR"})[row[1].text.split("/")[0]];
        const style = isOurs ? "is-ours" : name === baseline ? "is-reference" : "";
        return `<div class="performance-row ${style}" data-method="${performanceEscape(name)}">
          <div class="performance-method"><strong>${performanceEscape(isOurs ? config.variant : name)}</strong><span title="${performanceEscape(row[1].text)}">${input}</span></div>
          <div class="performance-value"><strong>${quality.toFixed(2)}<small>${metric.unit}</small></strong><span class="performance-track" aria-hidden="true"><i style="width:${quality / metric.ceiling * 100}%"></i></span></div>
          <div class="performance-value${fps === null ? " is-unreported" : ""}"><strong>${fps === null ? "Not reported" : fps.toFixed(2)}</strong><span class="performance-track" aria-hidden="true">${fps === null ? "" : `<i style="width:${fps / config.speedCeiling * 100}%"></i>`}</span></div>
        </div>`;
      }).join("")}
      <div class="performance-chart-scale"><span>Zero-based bar scales</span><span>0–${metric.ceiling} ${metric.unit}</span><span>0–${config.speedCeiling} FPS</span></div>`;
  }

  card.querySelector(".performance-baseline").addEventListener("change", event => {
    baseline = event.target.value;
    render();
  });
  card.querySelector(".performance-metric").addEventListener("change", event => {
    metric = config.metrics.find(item => item.id === event.target.value);
    render();
  });
  render();
  return card;
}

fetch("assets/paper-results.json").then(response => {
  if (!response.ok) throw new Error(`Paper results: HTTP ${response.status}`);
  return response.json();
}).then(data => {
  populatePaperHighlights(data);
  const container = byId("performance-cards");
  performanceConfigs.forEach(config => container.append(createPerformanceCard(config, data.tables.find(table => table.id === config.id))));
  byId("performance-status").hidden = true;
}).catch(error => {
  byId("paper-highlights").hidden = true;
  byId("performance-status").textContent = "Performance highlights could not load. The complete paper tables are available below.";
  console.error(error);
});
