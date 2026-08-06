/* ==========================================================================
   SIPEMO — chart primitives (SVG murni, tanpa library)

   Spesifikasi mark yang dipatuhi:
     · mark tipis (tinggi bar 14px pada baris 34px)
     · ujung data membulat 4px, pangkal menempel baseline (persegi)
     · celah 2px antar-bar diisi warna surface, bukan garis border
     · grid & sumbu berupa hairline SOLID (tidak putus-putus)
     · label langsung yang selektif — nilai selalu terbaca tanpa hover
     · area hit >= 24px, tooltip hanya memperkaya, tidak menjadi satu-satunya
       cara membaca nilai
   ========================================================================== */

import { dec } from "./format.js";

const NS = "http://www.w3.org/2000/svg";

const BAR_H = 14;
const ROW_H = 34;
const RADIUS = 4;

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) node.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(node);
  return node;
}

/* Bar horizontal: pangkal persegi di baseline, ujung data membulat. */
function barPath(x0, x1, y, h, r = RADIUS) {
  const w = x1 - x0;
  if (w <= 0.5) return `M${x0},${y} h0.5 v${h} h-0.5 Z`;
  if (w <= r) return `M${x0},${y} h${w} v${h} h${-w} Z`;
  return [
    `M${x0},${y}`,
    `H${x1 - r}`,
    `Q${x1},${y} ${x1},${y + r}`,
    `V${y + h - r}`,
    `Q${x1},${y + h} ${x1 - r},${y + h}`,
    `H${x0}`,
    "Z",
  ].join(" ");
}

/* -------------------------------------------------------------------------
   Tooltip tunggal, dipakai bersama semua chart
   ------------------------------------------------------------------------- */
let tipEl = null;

function tooltip() {
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "chart-tooltip";
    tipEl.setAttribute("role", "presentation");
    document.body.appendChild(tipEl);
  }
  return tipEl;
}

function showTip(html, evt) {
  const t = tooltip();
  t.innerHTML = html;
  t.dataset.show = "true";
  const pad = 14;
  const r = t.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${Math.max(8, y)}px`;
}

function hideTip() {
  if (tipEl) tipEl.dataset.show = "false";
}

/* -------------------------------------------------------------------------
   Bar chart horizontal
   ------------------------------------------------------------------------- */
export function hbar(opts) {
  const {
    data,
    max = 1,
    ticks = [0, 0.25, 0.5, 0.75, 1],
    threshold = null,
    thresholdLabel = "",
    format = (v) => dec(v, 3),
    gutterLeft = 104,
    gutterRight = 54,
    tickFormat = (v) => String(v),
    tooltipHtml = null,
  } = opts;

  const W = 640;
  /* Ruang atas harus cukup menampung label threshold secara utuh — bbox teks
     10px kira-kira 13px tinggi, jadi baseline perlu duduk >= 14px dari tepi.
     Label yang terpotong tepi kontainer adalah cacat, bukan gaya. */
  const padTop = threshold !== null ? 34 : 8;
  const padBottom = 24;
  const plotW = W - gutterLeft - gutterRight;
  const H = padTop + data.length * ROW_H + padBottom;

  const svg = el("svg", {
    class: "chart-svg",
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
  });

  const x = (v) => gutterLeft + (Math.max(0, Math.min(v, max)) / max) * plotW;
  const plotBottom = padTop + data.length * ROW_H;

  /* Grid vertikal — hairline solid, resesif */
  for (const t of ticks) {
    if (t === 0) continue;
    el("line", {
      class: "chart-grid-line",
      x1: x(t),
      x2: x(t),
      y1: padTop,
      y2: plotBottom,
    }, svg);
  }

  /* Sumbu baseline di nol */
  el("line", {
    class: "chart-axis-line",
    x1: gutterLeft,
    x2: gutterLeft,
    y1: padTop,
    y2: plotBottom,
  }, svg);

  /* Tick label */
  for (const t of ticks) {
    el("text", {
      class: "chart-tick",
      x: x(t),
      y: plotBottom + 15,
      "text-anchor": t === 0 ? "start" : t === max ? "end" : "middle",
    }, svg).textContent = tickFormat(t);
  }

  /* Baris data */
  data.forEach((d, i) => {
    const y = padTop + i * ROW_H;
    const barY = y + (ROW_H - BAR_H) / 2;
    const g = el("g", { class: "chart-row" }, svg);

    /* Nama kategori — label langsung, identitas tidak pernah warna saja */
    el("text", {
      class: "chart-cat",
      x: gutterLeft - 10,
      y: y + ROW_H / 2 + 4,
      "text-anchor": "end",
    }, g).textContent = d.label;

    el("path", {
      class: "chart-bar",
      d: barPath(gutterLeft, x(d.value), barY, BAR_H),
      fill: d.color || "var(--series-1)",
      "fill-opacity": d.dim ? 0.4 : 1,
    }, g);

    /* Nilai selalu terbaca — memenuhi aturan relief untuk warna kontras rendah */
    el("text", {
      class: "chart-value",
      x: W - gutterRight + 8,
      y: y + ROW_H / 2 + 4,
      "text-anchor": "start",
    }, g).textContent = format(d.value);

    /* Area hit penuh baris (34px > minimum 24px) */
    const hit = el("rect", {
      class: "chart-hit",
      x: 0,
      y,
      width: W,
      height: ROW_H,
    }, g);

    if (tooltipHtml) {
      const html = tooltipHtml(d);
      hit.addEventListener("mousemove", (e) => showTip(html, e));
      hit.addEventListener("mouseleave", hideTip);
    }
  });

  /* Penanda threshold — garis solid + label */
  if (threshold !== null) {
    const tx = x(threshold);
    el("line", { class: "thr-line", x1: tx, x2: tx, y1: padTop - 12, y2: plotBottom }, svg);
    el("text", {
      class: "thr-label",
      x: Math.min(tx + 5, W - gutterRight - 4),
      y: padTop - 18,
      "text-anchor": "start",
    }, svg).textContent = thresholdLabel;
  }

  return svg;
}

/* -------------------------------------------------------------------------
   Line chart mini untuk small multiples (kurva training)
   ------------------------------------------------------------------------- */
export function miniLine(opts) {
  const {
    points,
    color = "var(--series-1)",
    yMin = 0,
    yMax = 1,
    xLabels = [],
    format = (v) => dec(v, 3),
    label = "",
  } = opts;

  const W = 260;
  const H = 108;
  const padL = 34;
  const padR = 12;
  const padT = 10;
  const padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const svg = el("svg", {
    class: "chart-svg",
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
  });

  const x = (i) => padL + (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const y = (v) => padT + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

  /* Grid horizontal hairline */
  for (const frac of [0, 0.5, 1]) {
    const gy = padT + plotH - frac * plotH;
    el("line", { class: "chart-grid-line", x1: padL, x2: W - padR, y1: gy, y2: gy }, svg);
    el("text", {
      class: "chart-tick",
      x: padL - 6,
      y: gy + 3.5,
      "text-anchor": "end",
    }, svg).textContent = dec(yMin + frac * (yMax - yMin), 2);
  }

  el("line", { class: "chart-axis-line", x1: padL, x2: padL, y1: padT, y2: padT + plotH }, svg);

  /* Garis 2px */
  el("path", {
    d: points.map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`).join(" "),
    fill: "none",
    stroke: color,
    "stroke-width": 2,
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
  }, svg);

  points.forEach((v, i) => {
    /* Cincin surface 2px pada marker yang bertumpuk */
    el("circle", { cx: x(i), cy: y(v), r: 4, fill: color, stroke: "var(--surface-1)", "stroke-width": 2 }, svg);

    const hit = el("rect", {
      class: "chart-hit",
      x: x(i) - 14,
      y: padT,
      width: 28,
      height: plotH,
    }, svg);
    const name = xLabels[i] || `titik ${i + 1}`;
    const html = `<div class="tt-title">${label}</div><div class="tt-row">${name}: ${format(v)}</div>`;
    hit.addEventListener("mousemove", (e) => showTip(html, e));
    hit.addEventListener("mouseleave", hideTip);
  });

  /* Label endpoint — label langsung yang selektif.
     Kalau titik akhir dekat tepi atas, label dipindah ke bawah titik supaya
     tidak terpotong kontainer. */
  const last = points.length - 1;
  const lastY = y(points[last]);
  el("text", {
    class: "chart-value",
    x: Math.min(x(last) + 6, W - 4),
    y: lastY > padT + 18 ? lastY - 9 : lastY + 16,
    "text-anchor": "end",
  }, svg).textContent = format(points[last]);

  xLabels.forEach((lab, i) => {
    el("text", {
      class: "chart-tick",
      x: x(i),
      y: H - 6,
      "text-anchor": "middle",
    }, svg).textContent = lab;
  });

  return svg;
}

/* -------------------------------------------------------------------------
   Tabel — kembaran setiap chart (WCAG-clean)
   ------------------------------------------------------------------------- */
export function table(columns, rows, opts = {}) {
  const t = document.createElement("table");
  t.className = "data";

  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  for (const c of columns) {
    const th = document.createElement("th");
    th.textContent = c.label;
    th.scope = "col";
    htr.appendChild(th);
  }
  thead.appendChild(htr);
  t.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const r of rows) {
    const tr = document.createElement("tr");
    if (opts.isHighlighted && opts.isHighlighted(r)) tr.dataset.served = "true";
    columns.forEach((c, i) => {
      const cell = document.createElement(i === 0 ? "th" : "td");
      if (i === 0) cell.scope = "row";
      const v = c.get(r);
      if (v instanceof Node) cell.appendChild(v);
      else cell.innerHTML = v;
      if (c.className) cell.className = c.className;
      tr.appendChild(cell);
    });
    tbody.appendChild(tr);
  }
  t.appendChild(tbody);

  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.appendChild(t);
  return wrap;
}

/* Pasangan chart + tabel dengan tombol alih tampilan. */
export function withTableView(chartNode, tableNode, opts = {}) {
  const { defaultView = "chart" } = opts;
  const container = document.createElement("div");

  const body = document.createElement("div");
  const chartWrap = document.createElement("div");
  chartWrap.appendChild(chartNode);
  const tableWrap = document.createElement("div");
  tableWrap.appendChild(tableNode);
  body.appendChild(chartWrap);
  body.appendChild(tableWrap);

  const toggle = document.createElement("div");
  toggle.className = "view-toggle";
  toggle.setAttribute("role", "group");
  toggle.setAttribute("aria-label", "Alih tampilan");

  const btnChart = document.createElement("button");
  btnChart.textContent = "Grafik";
  const btnTable = document.createElement("button");
  btnTable.textContent = "Tabel";

  function set(view) {
    const isChart = view === "chart";
    chartWrap.hidden = !isChart;
    tableWrap.hidden = isChart;
    btnChart.setAttribute("aria-pressed", String(isChart));
    btnTable.setAttribute("aria-pressed", String(!isChart));
  }
  btnChart.addEventListener("click", () => set("chart"));
  btnTable.addEventListener("click", () => set("table"));
  toggle.append(btnChart, btnTable);
  set(defaultView);

  container.appendChild(body);
  container._toggle = toggle;
  return container;
}

export { hideTip };
