"""漫画文字框选网页版：把漫画图拖进网页，自动把所有文字框出来，并列出识别内容。

和 web_ocr.py 的区别：
  - 不再依赖 Docker 里的 VLM 服务，直接用 PaddleOCR (PP-OCRv5) 的检测 + 识别，本地运行
  - 返回每段文字的多边形坐标，在图片上画框；点击框/列表项可互相高亮
  - 网页上可调检测参数（阈值、框扩张比例），框不全或太碎时直接拖滑块重跑

依赖：pip install paddleocr paddlepaddle   (PaddleOCR 3.x)
运行：python manga_ocr.py
然后浏览器打开 http://127.0.0.1:7860（脚本会自动打开）
"""
import argparse
import json
import os
import re
import tempfile
import threading
import traceback
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# 跳过启动时的模型源连通性检查
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAX_BYTES = 50 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>漫画文字框选</title>
<style>
  :root { color-scheme: light dark; --accent: #3b82f6; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; font-family: system-ui, "Microsoft YaHei", sans-serif; }
  h1 { margin: 0 0 16px; font-size: 20px; }
  .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; height: calc(100vh - 100px); min-height: 360px; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; height: auto; } }
  #drop { border: 2px dashed #8888; border-radius: 12px; display: flex; flex-direction: column;
          align-items: center; justify-content: center; text-align: center; padding: 12px;
          cursor: pointer; overflow: auto; min-height: 240px; transition: border-color .15s, background .15s; }
  #drop.over { border-color: var(--accent); background: #3b82f622; }
  #drop p { margin: 0; opacity: .7; line-height: 1.8; }
  #wrap { position: relative; display: none; line-height: 0; }
  #preview { max-width: 100%; max-height: calc(100vh - 150px); display: block; }
  #cv { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
  .right { display: flex; flex-direction: column; min-height: 240px; min-width: 0; }
  .bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; min-height: 32px; gap: 8px; flex-wrap: wrap; }
  #status { opacity: .75; font-size: 14px; }
  button { padding: 6px 14px; border-radius: 8px; border: 1px solid #8886; background: transparent;
           color: inherit; cursor: pointer; font-size: 14px; white-space: nowrap; }
  button:hover { border-color: var(--accent); }
  .kw { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .kw input[type=text] { flex: 1; padding: 8px 12px; border-radius: 8px; border: 1px solid #8886;
                         background: transparent; color: inherit; font-size: 14px; }
  .kw input[type=text]:focus { outline: none; border-color: var(--accent); }
  .kw label { font-size: 13px; opacity: .8; white-space: nowrap; cursor: pointer; }
  details { margin-bottom: 8px; font-size: 13px; }
  summary { cursor: pointer; opacity: .8; }
  .params { display: grid; grid-template-columns: auto 1fr 44px; gap: 6px 10px; align-items: center; margin-top: 8px; }
  .params input[type=range] { width: 100%; }
  .params span.v { text-align: right; opacity: .8; }
  #list { flex: 1; overflow: auto; border: 1px solid #8886; border-radius: 12px; padding: 6px; }
  .item { display: flex; gap: 10px; padding: 6px 8px; border-radius: 8px; cursor: pointer; line-height: 1.6; font-size: 15px; }
  .item:hover { background: #8882; }
  .item.sel { background: #3b82f633; outline: 1px solid var(--accent); }
  .item .n { flex: none; min-width: 26px; text-align: right; opacity: .55; font-variant-numeric: tabular-nums; }
  .item .s { flex: none; font-size: 12px; opacity: .5; align-self: center; }
  .empty { padding: 12px; opacity: .55; font-size: 14px; }
  .spin { display: inline-block; width: 12px; height: 12px; margin-right: 6px; border: 2px solid #8886;
          border-top-color: var(--accent); border-radius: 50%; animation: s .8s linear infinite; vertical-align: -1px; }
  @keyframes s { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<h1>漫画文字框选</h1>
<div class="grid">
  <div id="drop">
    <p id="hint">把漫画图拖到这里<br>或点击选择文件<br>或直接 Ctrl+V 粘贴截图</p>
    <div id="wrap"><img id="preview" alt=""><canvas id="cv"></canvas></div>
  </div>
  <div class="right">
    <div class="bar">
      <span id="status">等待图片</span>
      <span><button id="rerun">重新识别</button> <button id="trans">翻译</button> <button id="copy">复制全部</button> <button id="save">保存当前图</button> <button id="saveFinal">保存译图</button></span>
    </div>
    <div style="margin-bottom:8px"><label style="font-size:14px;cursor:pointer"><input type="checkbox" id="fillWhite"> 框内填充白色（遮住原文字）</label>
      <label style="font-size:14px;cursor:pointer;margin-left:16px"><input type="checkbox" id="showBox" checked> 显示线框和序号</label>
      <label style="font-size:14px;cursor:pointer;margin-left:16px"><input type="checkbox" id="dropRuby" checked> 忽略注音小字（不翻译，导出时擦掉）</label></div>
    <details>
      <summary>检测参数（框不全 / 框太碎时调整，然后点“重新识别”）</summary>
      <div class="params">
        <label for="p_thresh">像素阈值</label><input type="range" id="p_thresh" min="0.05" max="0.6" step="0.05" value="0.2"><span class="v" id="v_thresh"></span>
        <label for="p_box">框置信度</label><input type="range" id="p_box" min="0.1" max="0.9" step="0.05" value="0.5"><span class="v" id="v_box"></span>
        <label for="p_unclip">框扩张比</label><input type="range" id="p_unclip" min="1.0" max="3.5" step="0.1" value="1.8"><span class="v" id="v_unclip"></span>
        <label for="p_side">最长边</label><input type="range" id="p_side" min="640" max="2880" step="160" value="1600"><span class="v" id="v_side"></span>
        <label for="p_ruby">小字比例</label><input type="range" id="p_ruby" min="0.3" max="0.9" step="0.05" value="0.6"><span class="v" id="v_ruby"></span>
      </div>
    </details>
    <div id="list"><div class="empty">识别结果会显示在这里</div></div>
  </div>
</div>
<input type="file" id="file" accept="image/*" hidden>
<script>
const $ = id => document.getElementById(id);
const drop = $("drop"), file = $("file"), preview = $("preview"), wrap = $("wrap"),
      cv = $("cv"), hint = $("hint"), statusEl = $("status"), list = $("list");

let curFile = null, items = [], baseStatus = "", selected = -1;

// ---------- 参数滑块 ----------
const sliders = { thresh: "p_thresh", box: "p_box", unclip: "p_unclip", side: "p_side", ruby: "p_ruby" };
for (const [k, id] of Object.entries(sliders)) {
  const upd = () => { $("v_" + k).textContent = $(id).value; };
  $(id).addEventListener("input", upd); upd();
}
function paramQuery() {
  return new URLSearchParams({
    thresh: $("p_thresh").value, box: $("p_box").value,
    unclip: $("p_unclip").value, side: $("p_side").value,
  }).toString();
}

// ---------- 选择 / 拖拽 / 粘贴 ----------
drop.addEventListener("click", e => { if (e.target === cv) return; if (!curFile) file.click(); });
file.addEventListener("change", () => { if (file.files[0]) run(file.files[0]); file.value = ""; });
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) run(f); });
["dragover", "drop"].forEach(ev => window.addEventListener(ev, e => e.preventDefault()));
document.addEventListener("paste", e => {
  for (const it of e.clipboardData.items) {
    if (it.kind === "file" && it.type.startsWith("image/")) { run(it.getAsFile()); break; }
  }
});
// 已有图时，双击左侧区域可换图
drop.addEventListener("dblclick", () => file.click());

$("rerun").addEventListener("click", () => { if (curFile) run(curFile, true); });
$("fillWhite").addEventListener("change", draw);
$("showBox").addEventListener("change", draw);
const reRuby = () => { markRuby(); draw(); renderList(); };
$("dropRuby").addEventListener("change", reRuby);
$("p_ruby").addEventListener("input", reRuby);

$("trans").addEventListener("click", async () => {
  if (!items.length) { statusEl.textContent = "还没有识别结果"; return; }
  statusEl.innerHTML = '<span class="spin"></span>翻译中…';
  try {
    const r = await fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: items.map(it => isRuby(it) ? "" : it.text) }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    j.texts.forEach((t, i) => { if (items[i]) items[i].tr = t; });
    $("fillWhite").checked = true;   // 翻译完直接显示在白底框里
    statusEl.textContent = baseStatus + "，翻译完成";
    draw(); renderList();
  } catch (err) {
    statusEl.textContent = "翻译出错：" + err.message;
  }
});

// ---------- 注音小字（ふりがな）----------
// 判定：框的短边小于旁边某个大框短边的 ratio 倍，并且中心落在那个大框附近 → 当作注音
function isRuby(it) { return it.ruby && $("dropRuby").checked; }

function markRuby() {
  const ratio = parseFloat($("p_ruby").value);
  const dist = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
  const info = items.map(it => {
    const P = it.poly, xs = P.map(p => p[0]), ys = P.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    let s = Math.min(x1 - x0, y1 - y0);
    if (P.length >= 4) {
      s = Math.min((dist(P[0], P[1]) + dist(P[3], P[2])) / 2, (dist(P[1], P[2]) + dist(P[0], P[3])) / 2);
    }
    return { s, x0, x1, y0, y1 };
  });
  items.forEach((it, i) => {
    const a = info[i], cx = (a.x0 + a.x1) / 2, cy = (a.y0 + a.y1) / 2;
    it.ruby = info.some((b, j) => j !== i && a.s < b.s * ratio
      && cx > b.x0 - b.s * 0.8 && cx < b.x1 + b.s * 0.8
      && cy > b.y0 - b.s * 0.8 && cy < b.y1 + b.s * 0.8);
  });
}

function visibleItems() { return items.filter(it => !isRuby(it)); }

// ---------- 绘制 ----------
function colorOf(i) { return `hsl(${(i * 47) % 360} 85% 50%)`; }

// ---------- 译文排版 ----------
const NO_LINE_START = "，。、！？；：）」』】》…,.!?;:)]}";
function wrapText(ctx, text, maxW) {
  const lines = []; let cur = "";
  for (const ch of text) {
    if (ctx.measureText(cur + ch).width > maxW && cur) {
      if (NO_LINE_START.includes(ch)) { cur += ch; continue; }   // 避头：标点挤在上一行末尾
      lines.push(cur); cur = ch;
    } else cur += ch;
  }
  if (cur) lines.push(cur);
  return lines;
}

function drawTranslations(ctx) {
  ctx.fillStyle = "#000";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const it of items) {
    if (!it.tr || isRuby(it)) continue;
    // 用四边形本身的边长和倾角，而不是外接的水平矩形：斜框里的译文跟着一起旋转
    const P = it.poly;
    if (P.length < 4) continue;
    const dist = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
    const w = (dist(P[0], P[1]) + dist(P[3], P[2])) / 2 - 4;   // 上下边平均长度
    const h = (dist(P[1], P[2]) + dist(P[0], P[3])) / 2 - 4;   // 左右边平均长度
    if (w <= 0 || h <= 0) continue;
    const ox = (P[0][0] + P[1][0] + P[2][0] + P[3][0]) / 4;
    const oy = (P[0][1] + P[1][1] + P[2][1] + P[3][1]) / 4;
    let ang = Math.atan2(P[1][1] - P[0][1], P[1][0] - P[0][0]);  // 上边的倾角
    if (ang > Math.PI / 2) ang -= Math.PI; else if (ang < -Math.PI / 2) ang += Math.PI;
    ctx.save();
    ctx.translate(ox, oy);
    ctx.rotate(ang);
    const cx = 0, cy = 0;   // 旋转后，框中心就是原点，下面的排版代码不用改
    const text = it.tr.replace(/\\n/g, "");
    if (h > w * 1.5) {
      // 窄长的框：竖排，从右到左分列，每列从上到下
      const chars = Array.from(text);
      let size = Math.min(w, 60), per = 1, cols = 1;
      for (; size >= 10; size--) {
        per = Math.max(1, Math.floor(h / size));
        cols = Math.ceil(chars.length / per);
        if (cols * size * 1.1 <= w) break;
      }
      size = Math.max(size, 10);
      per = Math.max(1, Math.floor(h / size));
      cols = Math.ceil(chars.length / per);
      ctx.font = `bold ${size}px "Microsoft YaHei", sans-serif`;
      const colW = size * 1.1, totalW = cols * colW;
      for (let c = 0; c < cols; c++) {
        const part = chars.slice(c * per, (c + 1) * per);
        const x = cx + totalW / 2 - (c + 0.5) * colW;
        const top = cy - part.length * size / 2;
        part.forEach((ch, j) => ctx.fillText(ch, x, top + (j + 0.5) * size));
      }
    } else {
      // 其他框：横排，自动找刚好放得下的最大字号
      let size = Math.min(h, 60), lines = [];
      for (; size >= 10; size--) {
        ctx.font = `bold ${size}px "Microsoft YaHei", sans-serif`;
        lines = wrapText(ctx, text, w);
        if (lines.length * size * 1.15 <= h) break;
      }
      if (size < 10) {
        size = 10;
        ctx.font = `bold ${size}px "Microsoft YaHei", sans-serif`;
        lines = wrapText(ctx, text, w);
      }
      const lh = size * 1.15, top = cy - lines.length * lh / 2;
      lines.forEach((ln, j) => ctx.fillText(ln, cx, top + (j + 0.5) * lh));
    }
    ctx.restore();
  }
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
}

function draw() {
  paint(cv.getContext("2d"), { fill: $("fillWhite").checked, boxes: $("showBox").checked, sel: selected });
}

// 把当前结果画到任意 canvas 上
//   o.fill   框内铺白底并写上译文    o.onlyTr 只给已有译文的框铺白底（导出成品时用）
//   o.boxes  彩色线框和序号          o.sel    选中的框序号（-1 表示没有）
function paint(ctx, o) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  const lw = Math.max(2, Math.round(W / 400));
  const fs = Math.max(14, Math.round(W / 45));
  ctx.font = `bold ${fs}px sans-serif`;
  ctx.textBaseline = "top";
  // 先统一铺一层白色填充（再画边框，避免相邻框互相盖住边框和序号）
  if (o.fill) {
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = lw * 2;   // 白色描边略微外扩，盖住抗锯齿的残边
    ctx.lineJoin = "round";
    for (const it of items) {
      if (o.onlyTr && !it.tr && !isRuby(it)) continue;   // 注音小字也要擦掉，但不写译文
      ctx.beginPath();
      it.poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  }
  for (const it of items) {
    if (!o.boxes) break;
    if (isRuby(it)) {   // 注音小字：灰色虚线，不编号
      ctx.setLineDash([lw * 3, lw * 2]);
      ctx.lineWidth = lw;
      ctx.strokeStyle = "#888";
      ctx.beginPath();
      it.poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      continue;
    }
    const isSel = it.id === o.sel;
    ctx.lineWidth = isSel ? lw * 2 : lw;
    ctx.strokeStyle = isSel ? "#ef4444" : colorOf(it.id);
    ctx.fillStyle = isSel ? "#ef444433" : "transparent";
    ctx.beginPath();
    it.poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    if (isSel) ctx.fill();
    ctx.stroke();
    // 序号小标签
    const minX = Math.min(...it.poly.map(p => p[0])), minY = Math.min(...it.poly.map(p => p[1]));
    const label = String(it.id + 1), w = ctx.measureText(label).width + 8;
    ctx.fillStyle = isSel ? "#ef4444" : colorOf(it.id);
    ctx.fillRect(minX, minY - fs - 4 < 0 ? minY : minY - fs - 4, w, fs + 4);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, minX + 4, (minY - fs - 4 < 0 ? minY : minY - fs - 4) + 2);
  }
  if (o.fill) drawTranslations(ctx);
}

function renderList() {
  const vis = visibleItems();
  list.innerHTML = "";
  if (!items.length) { list.innerHTML = '<div class="empty">没有识别到文字</div>'; return; }
  if (!vis.length) { list.innerHTML = '<div class="empty">没有匹配的文字</div>'; return; }
  for (const it of vis) {
    const d = document.createElement("div");
    d.className = "item" + (it.id === selected ? " sel" : "");
    d.dataset.id = it.id;
    d.innerHTML = '<span class="n"></span><span class="t"></span><span class="s"></span>';
    d.children[0].textContent = it.id + 1;
    d.children[1].textContent = (it.text || "（未识别）") + (it.tr ? "  →  " + it.tr : "");
    d.children[2].textContent = it.score.toFixed(2);
    d.addEventListener("click", () => select(it.id, false));
    list.appendChild(d);
  }
}

function select(id, scroll) {
  selected = selected === id ? -1 : id;
  draw();
  list.querySelectorAll(".item").forEach(el => {
    const on = Number(el.dataset.id) === selected;
    el.classList.toggle("sel", on);
    if (on && scroll) el.scrollIntoView({ block: "nearest" });
  });
}

function inPoly(x, y, poly) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}
cv.addEventListener("click", e => {
  e.stopPropagation();
  const r = cv.getBoundingClientRect();
  const x = (e.clientX - r.left) * cv.width / r.width, y = (e.clientY - r.top) * cv.height / r.height;
  const hit = visibleItems().filter(it => inPoly(x, y, it.poly)).pop();
  select(hit ? hit.id : -1, true);
  if (!hit) { selected = -1; draw(); renderList(); }
});

$("copy").addEventListener("click", async () => {
  const text = visibleItems().map(it => it.text).join("\\n");
  if (!text) return;
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta);
    ta.select(); document.execCommand("copy"); ta.remove();
  }
  const old = statusEl.textContent;
  statusEl.textContent = "已复制";
  setTimeout(() => { statusEl.textContent = old; }, 1200);
});

function saveCanvas(c, name) {
  c.toBlob(b => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
}

// 保存当前图：原图 + 画布上现在看到的内容（线框、选中高亮都会带上）
$("save").addEventListener("click", () => {
  if (!preview.naturalWidth) return;
  const c = document.createElement("canvas");
  c.width = preview.naturalWidth; c.height = preview.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(preview, 0, 0);
  ctx.drawImage(cv, 0, 0);
  saveCanvas(c, "manga_view.png");
});

// 保存译图：不带线框和序号，只有原图 + 白底 + 译文，不受上面两个复选框影响
$("saveFinal").addEventListener("click", () => {
  if (!preview.naturalWidth) return;
  if (!items.some(it => it.tr)) { statusEl.textContent = "还没有译文，请先点“翻译”"; return; }
  const c = document.createElement("canvas");
  c.width = preview.naturalWidth; c.height = preview.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(preview, 0, 0);
  const layer = document.createElement("canvas");
  layer.width = c.width; layer.height = c.height;
  paint(layer.getContext("2d"), { fill: true, onlyTr: true, boxes: false, sel: -1 });
  ctx.drawImage(layer, 0, 0);
  saveCanvas(c, "manga_translated.png");
});

function extOf(f) {
  const m = /\\.([A-Za-z0-9]+)$/.exec(f.name || "");
  if (m) return "." + m[1].toLowerCase();
  const t = (f.type || "").split("/")[1];
  return t ? "." + t.toLowerCase() : ".png";
}

function showImage(f) {
  return new Promise(resolve => {
    preview.onload = () => {
      cv.width = preview.naturalWidth; cv.height = preview.naturalHeight;
      resolve();
    };
    preview.src = URL.createObjectURL(f);
    wrap.style.display = "inline-block";
    hint.style.display = "none";
  });
}

async function run(f, again) {
  if (!f.type.startsWith("image/")) { statusEl.textContent = "只支持图片文件"; return; }
  curFile = f;
  if (!again) await showImage(f);
  items = []; selected = -1; draw();
  list.innerHTML = "";
  statusEl.innerHTML = '<span class="spin"></span>识别中，第一次可能比较慢…';
  const t0 = performance.now();
  try {
    const r = await fetch("/ocr?" + paramQuery(), { method: "POST", headers: { "X-Ext": extOf(f) }, body: f });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    const s = ((performance.now() - t0) / 1000).toFixed(1);
    items = (j.items || []).map((it, i) => ({ id: i, poly: it.poly, text: it.text, score: it.score }));
    markRuby();
    baseStatus = items.length ? `完成，用时 ${s} 秒，共 ${items.length} 处文字` : `完成，用时 ${s} 秒`;
    statusEl.textContent = baseStatus;
    draw(); renderList();
  } catch (err) {
    statusEl.textContent = "出错：" + err.message;
  }
}
</script>
</body>
</html>
"""


def _to_list(v):
    return v.tolist() if hasattr(v, "tolist") else v


def extract_items(res):
    """把 PaddleOCR 3.x 的单页结果整理成 [{poly, text, score}]。"""
    data = res.json if hasattr(res, "json") else res
    data = data.get("res", data)
    texts = _to_list(data.get("rec_texts")) or []
    scores = _to_list(data.get("rec_scores")) or []
    polys = _to_list(data.get("rec_polys")) or []
    if len(polys) != len(texts):  # 兜底：用检测框
        polys = _to_list(data.get("dt_polys")) or []
    items = []
    for i, poly in enumerate(polys):
        items.append({
            "poly": [[float(x), float(y)] for x, y in poly],
            "text": texts[i] if i < len(texts) else "",
            "score": float(scores[i]) if i < len(scores) else 0.0,
        })
    return items


SYSTEM_PROMPT = "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"


def _chat(url, model, text, strict=False):
    """调 LM Studio（OpenAI 兼容接口）翻译一段文字。strict=True 时强调必须输出中文。"""
    instruction = "将下面的日文文本翻译成中文，片假名外来语也要译成中文，不要输出英文："
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (instruction if strict else "将下面的日文文本翻译成中文：") + text},
        ],
        "temperature": 0.1,
        "top_p": 0.3,
        "max_tokens": 1024,
        "frequency_penalty": 0.1,
    }
    if model:
        body["model"] = model
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def translate_texts(texts, url, model):
    """整页一次翻译（一行一句，模型能看到上下文）；行数对不上就退回逐句翻译。"""
    idx = [i for i, t in enumerate(texts) if t and t.strip()]
    out = [""] * len(texts)
    if not idx:
        return out
    joined = "\n".join(texts[i].replace("\n", " ") for i in idx)
    lines = [ln.strip() for ln in _chat(url, model, joined).split("\n") if ln.strip()]
    if len(lines) == len(idx):
        for i, ln in zip(idx, lines):
            out[i] = ln
    else:
        for i in idx:
            out[i] = _chat(url, model, texts[i])
    # 原文有假名、译文却一个汉字都没有（比如把片假名外来语还原成了英文）：强调中文后重译一次
    kana, han = re.compile(r"[぀-ヿ]"), re.compile(r"[一-鿿]")
    for i in idx:
        if kana.search(texts[i]) and not han.search(out[i]):
            retry = _chat(url, model, texts[i], strict=True)
            if han.search(retry):
                out[i] = retry
    return out


class Handler(BaseHTTPRequestHandler):
    ocr = None
    llm_url = "http://127.0.0.1:1234/v1"
    llm_model = ""
    lock = threading.Lock()  # 同一时间只处理一张图

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "Not Found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/translate":
            return self._translate()
        if url.path != "/ocr":
            return self._json(404, {"error": "Not Found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BYTES:
            return self._json(400, {"error": "文件为空或超过 50MB"})
        data = self.rfile.read(length)
        ext = (self.headers.get("X-Ext") or ".png").lower()
        if ext not in EXTS:
            ext = ".png"

        q = parse_qs(url.query)

        def num(key, default, cast=float):
            try:
                return cast(q[key][0])
            except (KeyError, ValueError, IndexError):
                return default

        kwargs = {
            "text_det_thresh": num("thresh", 0.2),
            "text_det_box_thresh": num("box", 0.5),
            "text_det_unclip_ratio": num("unclip", 1.8),
            "text_det_limit_side_len": num("side", 1600, int),
            "text_det_limit_type": "max",
        }

        try:
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / f"input{ext}"
                src.write_bytes(data)
                with self.lock:
                    results = list(self.ocr.predict(str(src), **kwargs))
            items = []
            for res in results:
                items.extend(extract_items(res))
            self._json(200, {"items": items})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def _translate(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 5 * 1024 * 1024:
            return self._json(400, {"error": "请求为空或过大"})
        try:
            texts = json.loads(self.rfile.read(length))["texts"]
            self._json(200, {"texts": translate_texts(texts, self.llm_url, self.llm_model)})
        except urllib.error.URLError as e:
            self._json(502, {"error": f"连不上 LM Studio（{self.llm_url}），请确认已启动本地服务器：{e}"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass  # 不刷屏


def main():
    ap = argparse.ArgumentParser(description="漫画文字框选网页版 (PaddleOCR)")
    ap.add_argument("-p", "--port", type=int, default=7860, help="网页端口，默认 7860")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，默认只允许本机访问")
    ap.add_argument("--lang", default="ch", help="识别语言，默认 ch（中文，含日英数字）；日文漫画可试 japan")
    ap.add_argument("--llm-url", default="http://127.0.0.1:1234/v1", help="LM Studio 本地服务器地址")
    ap.add_argument("--llm-model", default="", help="LM Studio 里的模型名；只加载了一个模型时可留空")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    print("正在加载 OCR 模型，请稍等…")
    Handler.ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,   # 漫画页不需要整页方向分类
        use_doc_unwarping=False,              # 不做文档矫正，保证框坐标和原图一致
        use_textline_orientation=False,
        enable_mkldnn=False,                  # 关闭 oneDNN，规避 Windows 上的 ConvertPirAttribute2RuntimeAttribute 报错
        text_rec_score_thresh=0.0,            # 不丢弃低分结果，保证“全部框出来”
    )

    Handler.llm_url = args.llm_url
    Handler.llm_model = args.llm_model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"就绪：{url}  （Ctrl+C 退出）")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("已退出")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
