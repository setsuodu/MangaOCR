"""PaddleOCR-VL-1.6 网页版：把图片拖进网页，显示识别出的文字。

前提：Docker 里的 paddleocr-vlm 容器已经在 8080 端口运行。
运行：python web_ocr.py
然后浏览器打开 http://127.0.0.1:7860（脚本会自动打开）
"""
import argparse
import json
import os
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 跳过启动时的模型源连通性检查
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCRVL

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAX_BYTES = 50 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>图片识字</title>
<style>
  :root { color-scheme: light dark; --accent: #3b82f6; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; font-family: system-ui, "Microsoft YaHei", sans-serif; }
  h1 { margin: 0 0 16px; font-size: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: calc(100vh - 100px); min-height: 360px; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; height: auto; } }
  #drop { border: 2px dashed #8888; border-radius: 12px; display: flex; flex-direction: column;
          align-items: center; justify-content: center; text-align: center; padding: 16px;
          cursor: pointer; overflow: hidden; min-height: 240px; transition: border-color .15s, background .15s; }
  #drop.over { border-color: var(--accent); background: #3b82f622; }
  #drop p { margin: 0; opacity: .7; line-height: 1.8; }
  #preview { max-width: 100%; max-height: 100%; object-fit: contain; display: none; }
  .right { display: flex; flex-direction: column; min-height: 240px; }
  .bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; min-height: 32px; }
  #status { opacity: .75; font-size: 14px; }
  button { padding: 6px 14px; border-radius: 8px; border: 1px solid #8886; background: transparent;
           color: inherit; cursor: pointer; font-size: 14px; }
  button:hover { border-color: var(--accent); }
  textarea { flex: 1; width: 100%; resize: none; padding: 12px; border-radius: 12px; border: 1px solid #8886;
             background: transparent; color: inherit; font: 15px/1.7 "Microsoft YaHei", system-ui, sans-serif; }
  .kw { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .kw input[type=text] { flex: 1; padding: 8px 12px; border-radius: 8px; border: 1px solid #8886;
                         background: transparent; color: inherit; font-size: 14px; }
  .kw input[type=text]:focus { outline: none; border-color: var(--accent); }
  .kw label { font-size: 13px; opacity: .8; white-space: nowrap; cursor: pointer; }
  .spin { display: inline-block; width: 12px; height: 12px; margin-right: 6px; border: 2px solid #8886;
          border-top-color: var(--accent); border-radius: 50%; animation: s .8s linear infinite; vertical-align: -1px; }
  @keyframes s { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<h1>图片识字</h1>
<div class="grid">
  <div id="drop">
    <p id="hint">把图片拖到这里<br>或点击选择文件<br>或直接 Ctrl+V 粘贴截图</p>
    <img id="preview" alt="">
  </div>
  <div class="right">
    <div class="bar"><span id="status">等待图片</span><button id="copy">复制全部</button></div>
    <div class="kw">
      <input type="text" id="kw" placeholder="关键词，多个用空格或逗号分隔；留空显示全部">
      <label><input type="checkbox" id="all"> 必须全部包含</label>
    </div>
    <textarea id="out" readonly placeholder="识别结果会显示在这里"></textarea>
  </div>
</div>
<input type="file" id="file" accept="image/*" hidden>
<script>
const $ = id => document.getElementById(id);
const drop = $("drop"), file = $("file"), preview = $("preview"),
      hint = $("hint"), statusEl = $("status"), out = $("out");

drop.addEventListener("click", () => file.click());
file.addEventListener("change", () => { if (file.files[0]) run(file.files[0]); file.value = ""; });

["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add("over");
}));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove("over");
}));
drop.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (f) run(f);
});
// 防止拖到页面其它地方时浏览器直接打开图片
["dragover", "drop"].forEach(ev => window.addEventListener(ev, e => e.preventDefault()));

document.addEventListener("paste", e => {
  for (const it of e.clipboardData.items) {
    if (it.kind === "file" && it.type.startsWith("image/")) { run(it.getAsFile()); break; }
  }
});

$("copy").addEventListener("click", async () => {
  if (!out.value) return;
  try { await navigator.clipboard.writeText(out.value); }
  catch { out.select(); document.execCommand("copy"); }
  const old = statusEl.textContent;
  statusEl.textContent = "已复制";
  setTimeout(() => { statusEl.textContent = old; }, 1200);
});

// 关键词筛选：只保留包含关键词的行（不区分大小写）
let fullText = "", baseStatus = "";

function keywords() {
  return $("kw").value.split(/[\\s,，、;；]+/).map(s => s.toLowerCase()).filter(Boolean);
}

function render() {
  if (!fullText) return;
  const ks = keywords();
  if (!ks.length) {
    out.value = fullText;
    statusEl.textContent = baseStatus;
    return;
  }
  const needAll = $("all").checked;
  const lines = fullText.split(/\\r?\\n/).filter(l => {
    const t = l.toLowerCase();
    if (!t.trim()) return false;
    return needAll ? ks.every(k => t.includes(k)) : ks.some(k => t.includes(k));
  });
  out.value = lines.join("\\n");
  statusEl.textContent = lines.length ? `${baseStatus}，匹配 ${lines.length} 行` : `${baseStatus}，没有匹配的行`;
}
$("kw").addEventListener("input", render);
$("all").addEventListener("change", render);

function extOf(f) {
  const m = /\\.([A-Za-z0-9]+)$/.exec(f.name || "");
  if (m) return "." + m[1].toLowerCase();
  const t = (f.type || "").split("/")[1];
  return t ? "." + t.toLowerCase() : ".png";
}

async function run(f) {
  if (!f.type.startsWith("image/")) { statusEl.textContent = "只支持图片文件"; return; }
  preview.src = URL.createObjectURL(f);
  preview.style.display = "block";
  hint.style.display = "none";
  fullText = ""; baseStatus = "";
  out.value = "";
  statusEl.innerHTML = '<span class="spin"></span>识别中，第一次可能比较慢…';
  const t0 = performance.now();
  try {
    const r = await fetch("/ocr", { method: "POST", headers: { "X-Ext": extOf(f) }, body: f });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    const s = ((performance.now() - t0) / 1000).toFixed(1);
    fullText = j.text || "";
    if (!fullText) {
      out.value = "";
      statusEl.textContent = "没有识别到文字";
    } else {
      baseStatus = `完成，用时 ${s} 秒`;
      statusEl.textContent = baseStatus;
      render();
    }
  } catch (err) {
    statusEl.textContent = "出错：" + err.message;
  }
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    pipeline = None
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
        if self.path != "/ocr":
            return self._json(404, {"error": "Not Found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BYTES:
            return self._json(400, {"error": "文件为空或超过 50MB"})
        data = self.rfile.read(length)
        ext = (self.headers.get("X-Ext") or ".png").lower()
        if ext not in EXTS:
            ext = ".png"

        try:
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / f"input{ext}"
                src.write_bytes(data)
                out_dir = Path(tmp) / "out"
                out_dir.mkdir()
                with self.lock:
                    for res in self.pipeline.predict(str(src)):
                        res.save_to_markdown(save_path=str(out_dir))
                parts = [p.read_text(encoding="utf-8").strip() for p in sorted(out_dir.rglob("*.md"))]
            self._json(200, {"text": "\n\n".join(x for x in parts if x)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass  # 不刷屏


def main():
    ap = argparse.ArgumentParser(description="PaddleOCR-VL-1.6 网页版")
    ap.add_argument("-s", "--server", default="http://127.0.0.1:8080/v1", help="VLM 服务地址")
    ap.add_argument("-p", "--port", type=int, default=7860, help="网页端口，默认 7860")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，默认只允许本机访问")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    print("正在加载模型管线（版面检测等），请稍等…")
    Handler.pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="vllm-server",
        vl_rec_server_url=args.server,
    )

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
