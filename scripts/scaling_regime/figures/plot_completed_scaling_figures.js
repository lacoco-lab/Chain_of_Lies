#!/usr/bin/env node
/*
 * Publication figures for the completed balanced scaling experiments.
 *
 * Deliberately excludes S5 until its balanced 5-mechanism sweep is complete.
 * Inputs are the aggregate tables from the jointly trained Multiplication and
 * Iterated Addition (historically named "knowledge") experiments.
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "../../..");
const OUT = path.join(ROOT, "artifacts", "paper_figures", "scaling_completed_no_s5");
const PANELS = path.join(OUT, "appendix_accuracy_panels");
const INPUTS = {
  multiplication: path.join(ROOT, "artifacts", "multiplication_length_retraining_v1", "aggregate_length_metrics.csv"),
  knowledge: path.join(ROOT, "artifacts", "knowledge_length_retraining_all_but_one_v2", "aggregate_length_metrics.csv"),
};

const METHODS = ["vanilla", "filler", "public_only_cot", "piggyback", "steganography"];
const METHOD = {
  vanilla: { label: "Vanilla", color: "#6F6F6F", dash: "", marker: "circle" },
  filler: { label: "Filler", color: "#A48645", dash: "8 4", marker: "square" },
  public_only_cot: { label: "Public-only CoT", color: "#7F6AA2", dash: "3 3", marker: "triangle" },
  piggyback: { label: "Piggyback", color: "#0072B2", dash: "", marker: "diamond" },
  steganography: { label: "Invisible (steg.)", color: "#D55E00", dash: "9 3", marker: "cross" },
};
const TASK = {
  multiplication: { label: "Multiplication", xLabel: "Digits per operand", ticks: [2, 3, 4], domain: [1.82, 4.18] },
  knowledge: { label: "Iterated addition", xLabel: "Number of facts", ticks: [1, 2, 3, 4, 5], domain: [0.68, 5.32] },
};
const MODEL = { qwen: "Qwen2.5-7B", llama: "Llama-3.1-8B" };

function parseCsv(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const header = lines.shift().split(",");
  return lines.map(line => {
    const values = line.split(",");
    return Object.fromEntries(header.map((key, i) => [key, values[i] ?? ""]));
  });
}

function esc(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function n(value) { return Number(value); }
function clamp(value, lo, hi) { return Math.max(lo, Math.min(hi, value)); }
function fmt(value) { return Number(value.toFixed(2)); }
function linePath(points) { return points.map((p, i) => `${i ? "L" : "M"}${fmt(p[0])},${fmt(p[1])}`).join(" "); }

function markerSvg(kind, x, y, color, size = 5) {
  const common = `fill="white" stroke="${color}" stroke-width="2.2"`;
  if (kind === "circle") return `<circle cx="${fmt(x)}" cy="${fmt(y)}" r="${size}" ${common}/>`;
  if (kind === "square") return `<rect x="${fmt(x-size)}" y="${fmt(y-size)}" width="${2*size}" height="${2*size}" rx="1" ${common}/>`;
  if (kind === "triangle") return `<path d="M${fmt(x)},${fmt(y-size-1)} L${fmt(x+size+1)},${fmt(y+size)} L${fmt(x-size-1)},${fmt(y+size)} Z" ${common}/>`;
  if (kind === "diamond") return `<path d="M${fmt(x)},${fmt(y-size-1)} L${fmt(x+size+1)},${fmt(y)} L${fmt(x)},${fmt(y+size+1)} L${fmt(x-size-1)},${fmt(y)} Z" ${common}/>`;
  return `<path d="M${fmt(x-size)},${fmt(y-size)} L${fmt(x+size)},${fmt(y+size)} M${fmt(x+size)},${fmt(y-size)} L${fmt(x-size)},${fmt(y+size)}" fill="none" stroke="${color}" stroke-width="2.4" stroke-linecap="round"/>`;
}

function validate(rows, task) {
  const expectedLengths = TASK[task].ticks;
  const keySet = new Set(rows.map(r => `${r.model}|${r.condition}|${r.length}`));
  for (const model of Object.keys(MODEL)) {
    for (const method of METHODS) {
      for (const length of expectedLengths) {
        if (!keySet.has(`${model}|${method}|${length}`)) throw new Error(`Missing ${task}/${model}/${method}/${length}`);
      }
    }
  }
  const expected = 2 * METHODS.length * expectedLengths.length;
  if (rows.length !== expected) throw new Error(`${task}: expected ${expected} rows, found ${rows.length}`);
  for (const row of rows) {
    if (n(row.n_seeds) !== 3 || n(row.n_model_outputs) !== 600 || n(row.n_unique_eval_prompts) !== 200) {
      throw new Error(`Unexpected replication in ${task}/${row.model}/${row.condition}/${row.length}`);
    }
    if (Math.abs(n(row.format_rate_mean) - 1) > 1e-12) throw new Error(`Non-unit format rate in ${task}`);
  }
}

function svgStart(widthIn, heightIn, viewW, viewH) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${widthIn}in" height="${heightIn}in" viewBox="0 0 ${viewW} ${viewH}">
  <rect width="100%" height="100%" fill="white"/>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: #222; }
    .tick { font-size: 18px; fill: #444; }
    .label { font-size: 21px; font-weight: 500; }
    .title { font-size: 23px; font-weight: 700; }
    .model { font-size: 18px; font-weight: 700; }
    .legend { font-size: 18px; }
    .note { font-size: 16px; fill: #5A5A5A; }
  </style>`;
}

function axesSvg(panel, task, showYLabels, showXLabel, yDomain, yTicks, yPercent = false) {
  const { x, y, w, h } = panel;
  const cfg = TASK[task];
  const sx = value => x + (value - cfg.domain[0]) / (cfg.domain[1] - cfg.domain[0]) * w;
  const sy = value => y + h - (value - yDomain[0]) / (yDomain[1] - yDomain[0]) * h;
  let s = `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="#FFFFFF" stroke="#D7D7D7" stroke-width="1"/>`;
  for (const tick of yTicks) {
    const yy = sy(tick);
    s += `<line x1="${x}" y1="${fmt(yy)}" x2="${x+w}" y2="${fmt(yy)}" stroke="#E8E8E8" stroke-width="1"/>`;
    if (showYLabels) s += `<text class="tick" x="${x-13}" y="${fmt(yy+6)}" text-anchor="end">${yPercent ? Math.round(100*tick) : tick}</text>`;
  }
  for (const tick of cfg.ticks) {
    const xx = sx(tick);
    s += `<line x1="${fmt(xx)}" y1="${y+h}" x2="${fmt(xx)}" y2="${y+h+6}" stroke="#444" stroke-width="1.2"/>`;
    s += `<text class="tick" x="${fmt(xx)}" y="${y+h+27}" text-anchor="middle">${tick}</text>`;
  }
  s += `<line x1="${x}" y1="${y+h}" x2="${x+w}" y2="${y+h}" stroke="#444" stroke-width="1.2"/>`;
  s += `<line x1="${x}" y1="${y}" x2="${x}" y2="${y+h}" stroke="#444" stroke-width="1.2"/>`;
  if (showXLabel) s += `<text class="label" x="${x+w/2}" y="${y+h+58}" text-anchor="middle">${esc(cfg.xLabel)}</text>`;
  return { svg: s, sx, sy };
}

function legendSvg(x, y, compact = false) {
  const gap = compact ? 190 : 243;
  let s = "";
  METHODS.forEach((method, i) => {
    const m = METHOD[method], xx = x + gap*i;
    s += `<line x1="${xx}" y1="${y}" x2="${xx+35}" y2="${y}" stroke="${m.color}" stroke-width="${method === "piggyback" || method === "steganography" ? 3.2 : 2.5}" ${m.dash ? `stroke-dasharray="${m.dash}"` : ""}/>`;
    s += markerSvg(m.marker, xx+17.5, y, m.color, 4.2);
    s += `<text class="legend" x="${xx+44}" y="${y+6}">${esc(m.label)}</text>`;
  });
  return s;
}

function curvesSvg(rows, panel, task, model, clipId, showYLabels = true) {
  const axis = axesSvg(panel, task, showYLabels, false, [0, 1], [0, .25, .5, .75, 1], true);
  let s = axis.svg;
  s += `<defs><clipPath id="${clipId}"><rect x="${panel.x}" y="${panel.y}" width="${panel.w}" height="${panel.h}"/></clipPath></defs>`;
  const selected = rows.filter(r => r.task === task && r.model === model);
  for (const method of METHODS) {
    const m = METHOD[method];
    const series = selected.filter(r => r.condition === method).sort((a,b) => n(a.length)-n(b.length));
    const upper = series.map(r => [axis.sx(n(r.length)), axis.sy(clamp(n(r.private_exact_rate_mean)+n(r.private_exact_rate_sample_sd), 0, 1))]);
    const lower = [...series].reverse().map(r => [axis.sx(n(r.length)), axis.sy(clamp(n(r.private_exact_rate_mean)-n(r.private_exact_rate_sample_sd), 0, 1))]);
    s += `<path d="${linePath([...upper, ...lower])} Z" fill="${m.color}" fill-opacity="0.10" stroke="none" clip-path="url(#${clipId})"/>`;
  }
  for (const method of METHODS) {
    const m = METHOD[method];
    const series = selected.filter(r => r.condition === method).sort((a,b) => n(a.length)-n(b.length));
    const points = series.map(r => [axis.sx(n(r.length)), axis.sy(n(r.private_exact_rate_mean))]);
    s += `<path d="${linePath(points)}" fill="none" stroke="${m.color}" stroke-width="${method === "piggyback" || method === "steganography" ? 3.3 : 2.4}" ${m.dash ? `stroke-dasharray="${m.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round" clip-path="url(#${clipId})"/>`;
    for (const [xx, yy] of points) s += markerSvg(m.marker, xx, yy, m.color, 5.3);
  }
  return s;
}

function accuracyMainSvg(rows) {
  const W = 1400, H = 720;
  const panels = {
    qwen_multiplication: {x: 96, y: 118, w: 600, h: 215},
    qwen_knowledge: {x: 765, y: 118, w: 600, h: 215},
    llama_multiplication: {x: 96, y: 409, w: 600, h: 215},
    llama_knowledge: {x: 765, y: 409, w: 600, h: 215},
  };
  let s = svgStart(7.0, 3.6, W, H);
  s += legendSvg(190, 36, true);
  s += `<text class="title" x="396" y="91" text-anchor="middle">Multiplication</text>`;
  s += `<text class="title" x="1065" y="91" text-anchor="middle">Iterated addition</text>`;
  s += `<text class="label" transform="translate(25,371) rotate(-90)" text-anchor="middle">Private exact accuracy (%)</text>`;
  for (const [row, model] of [["qwen", "qwen"], ["llama", "llama"]]) {
    for (const task of ["multiplication", "knowledge"]) {
      const panel = panels[`${row}_${task}`];
      s += curvesSvg(rows, panel, task, model, `clip_${row}_${task}`, task === "multiplication");
      s += `<text class="model" x="${panel.x+12}" y="${panel.y+25}">${MODEL[model]}</text>`;
      if (row === "llama") s += `<text class="label" x="${panel.x+panel.w/2}" y="${panel.y+panel.h+58}" text-anchor="middle">${esc(TASK[task].xLabel)}</text>`;
    }
  }
  s += `<text class="note" x="1380" y="702" text-anchor="end">Mean ± 1 SD over 3 seeds; all plotted lengths are in training support.</text>`;
  return s + `</svg>`;
}

function accuracyPanelSvg(rows, task, model) {
  const W = 760, H = 560, panel = {x: 91, y: 145, w: 640, h: 310};
  let s = svgStart(3.35, 2.468, W, H);
  s += `<text class="title" x="${W/2}" y="34" text-anchor="middle">${TASK[task].label} · ${MODEL[model]}</text>`;
  const positions = [[44,70], [270,70], [510,70], [165,105], [420,105]];
  METHODS.forEach((method, i) => {
    const m = METHOD[method], [xx, yy] = positions[i];
    s += `<line x1="${xx}" y1="${yy}" x2="${xx+32}" y2="${yy}" stroke="${m.color}" stroke-width="2.7" ${m.dash ? `stroke-dasharray="${m.dash}"` : ""}/>`;
    s += markerSvg(m.marker, xx+16, yy, m.color, 4);
    s += `<text class="note" x="${xx+40}" y="${yy+5}">${m.label}</text>`;
  });
  s += curvesSvg(rows, panel, task, model, `clip_${task}_${model}`);
  s += `<text class="label" transform="translate(24,288) rotate(-90)" text-anchor="middle">Private exact accuracy (%)</text>`;
  s += `<text class="label" x="${panel.x+panel.w/2}" y="525" text-anchor="middle">${TASK[task].xLabel}</text>`;
  return s + `</svg>`;
}

function chainMainSvg(rows) {
  const W = 1400, H = 440;
  const panels = {
    multiplication: {x: 93, y: 83, w: 592, h: 270},
    knowledge: {x: 775, y: 83, w: 592, h: 270},
  };
  let s = svgStart(7.0, 2.2, W, H);
  const shown = ["piggyback", "steganography"];
  shown.forEach((method, i) => {
    const m = METHOD[method], xx = 465 + i*250;
    s += `<line x1="${xx}" y1="31" x2="${xx+42}" y2="31" stroke="${m.color}" stroke-width="3.3" ${m.dash ? `stroke-dasharray="${m.dash}"` : ""}/>`;
    s += markerSvg(m.marker, xx+21, 31, m.color, 5);
    s += `<text class="legend" x="${xx+51}" y="37">${m.label}</text>`;
  });
  s += `<text class="label" transform="translate(24,220) rotate(-90)" text-anchor="middle">Median visible chain length (words)</text>`;
  for (const task of ["multiplication", "knowledge"]) {
    const p = panels[task];
    const axis = axesSvg(p, task, task === "multiplication", true, [20, 95], [20, 40, 60, 80], false);
    s += axis.svg;
    s += `<text class="title" x="${p.x+p.w/2}" y="70" text-anchor="middle">${TASK[task].label}</text>`;
    for (const method of shown) {
      const byModel = ["qwen", "llama"].map(model => rows.filter(r => r.task === task && r.model === model && r.condition === method).sort((a,b) => n(a.length)-n(b.length)));
      const values = byModel.map(series => series.map(r => `${r.length}:${r.pooled_median_cot_words_format_valid}`).join("|"));
      if (values[0] !== values[1]) throw new Error(`Chain medians differ across models for ${task}/${method}`);
      const m = METHOD[method];
      const points = byModel[0].map(r => [axis.sx(n(r.length)), axis.sy(n(r.pooled_median_cot_words_format_valid))]);
      s += `<path d="${linePath(points)}" fill="none" stroke="${m.color}" stroke-width="3.6" ${m.dash ? `stroke-dasharray="${m.dash}"` : ""} stroke-linecap="round" stroke-linejoin="round"/>`;
      for (const [xx, yy] of points) s += markerSvg(m.marker, xx, yy, m.color, 5.3);
    }
    if (task === "knowledge") s += `<text class="note" x="${p.x+p.w-10}" y="${p.y+24}" text-anchor="end">Piggyback and steg. coincide</text>`;
  }
  s += `<text class="note" x="1380" y="432" text-anchor="end">Qwen and Llama coincide; answer block excluded.</text>`;
  return s + `</svg>`;
}

function toCsv(rows, columns) {
  return [columns.join(","), ...rows.map(r => columns.map(c => r[c]).join(","))].join("\n") + "\n";
}

function sha256(file) { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }

async function writeFigure(stem, svg) {
  const svgPath = `${stem}.svg`, pngPath = `${stem}.png`;
  fs.writeFileSync(svgPath, svg);
  const main = stem.includes("_main_");
  await sharp(Buffer.from(svg), { density: 96 })
    .resize({width: main ? 4200 : 2010})
    .png({ compressionLevel: 9 })
    .withMetadata({density: 600})
    .toFile(pngPath);
  return [svgPath, pngPath];
}

async function main() {
  fs.mkdirSync(PANELS, {recursive: true});
  const rows = [];
  for (const [task, file] of Object.entries(INPUTS)) {
    const taskRows = parseCsv(file);
    validate(taskRows, task);
    rows.push(...taskRows.map(r => ({task, ...r})));
  }

  const outputs = [];
  outputs.push(...await writeFigure(path.join(OUT, "figure_scaling_accuracy_main_no_s5"), accuracyMainSvg(rows)));
  outputs.push(...await writeFigure(path.join(OUT, "figure_scaling_chain_length_main_no_s5"), chainMainSvg(rows)));
  for (const task of Object.keys(TASK)) {
    for (const model of Object.keys(MODEL)) {
      outputs.push(...await writeFigure(path.join(PANELS, `accuracy_${task}_${model}`), accuracyPanelSvg(rows, task, model)));
    }
  }

  const accuracyColumns = ["task", "model", "condition", "length", "n_seeds", "n_unique_eval_prompts", "n_model_outputs", "private_exact_rate_mean", "private_exact_rate_sample_sd", "format_rate_mean"];
  const chainRows = rows.filter(r => ["piggyback", "steganography"].includes(r.condition));
  const chainColumns = ["task", "model", "condition", "length", "n_seeds", "pooled_median_cot_words_format_valid", "mean_seed_median_cot_words", "sample_sd_seed_median_cot_words"];
  const accuracyData = path.join(OUT, "figure_accuracy_data.csv");
  const chainData = path.join(OUT, "figure_chain_length_data.csv");
  fs.writeFileSync(accuracyData, toCsv(rows, accuracyColumns));
  fs.writeFileSync(chainData, toCsv(chainRows, chainColumns));
  outputs.push(accuracyData, chainData);

  const readme = path.join(OUT, "README.md");
  fs.writeFileSync(readme, `# Completed scaling figures (S5 excluded)\n\n` +
    `These figures use only the balanced, jointly trained Multiplication and Iterated Addition experiments. S5 is intentionally omitted because its matching experiment is still in progress.\n\n` +
    `## Suggested captions\n\n` +
    `**Accuracy.** Private-answer exact accuracy as task length increases. Each model–mechanism checkpoint was trained jointly across every displayed length. Points show the mean and bands show ±1 sample SD over three seeds (200 held-out prompts per seed and length). Piggybacking shares the first operand in Multiplication and all but one fact in Iterated Addition.\n\n` +
    `**Visible-chain length.** Median number of words in the monitor-visible chain for Piggybacking and Invisible/Steganographic reasoning, excluding the final answer block. Values coincide across Qwen2.5-7B and Llama-3.1-8B; the two mechanisms also coincide in Iterated Addition.\n\n` +
    `The PNGs are rendered at 600 DPI. SVG companions are retained for vector editing and paper submission.\n`);
  outputs.push(readme);

  const manifestPath = path.join(OUT, "manifest.json");
  const manifest = {
    created_on: new Date().toISOString().slice(0,10),
    status: "complete_for_multiplication_and_iterated_addition; s5_in_progress_and_excluded",
    validation: {
      multiplication_cells: 30,
      iterated_addition_cells: 50,
      seeds_per_cell: 3,
      unique_eval_prompts_per_seed_length: 200,
      outputs_per_aggregate_cell: 600,
      format_rate: 1.0,
      plotted_lengths_are_in_joint_training_support: true,
    },
    sources: Object.values(INPUTS).map(file => ({path: path.relative(ROOT, file), sha256: sha256(file)})),
    outputs: outputs.map(file => ({path: path.relative(ROOT, file), bytes: fs.statSync(file).size, sha256: sha256(file)})),
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  console.log(`Wrote ${outputs.length} files to ${path.relative(ROOT, OUT)}`);
}

main().catch(error => { console.error(error); process.exit(1); });
