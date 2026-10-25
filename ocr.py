import argparse
import os
import sys
from pathlib import Path

# 跳过启动时的模型源连通性检查（日志里提示过的开关）
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCRVL

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".pdf"}


def collect(inputs):
    files = []
    for p in inputs:
        path = Path(p)
        if path.is_dir():
            files += sorted(f for f in path.iterdir() if f.suffix.lower() in EXTS)
        elif path.is_file():
            files.append(path)
        else:
            print(f"跳过（找不到）: {p}", file=sys.stderr)
    return files


def main():
    ap = argparse.ArgumentParser(description="PaddleOCR-VL-1.6 通用识别脚本")
    ap.add_argument("inputs", nargs="+", help="图片路径，可多个，也可以是文件夹")
    ap.add_argument("-o", "--out", default="output", help="输出目录，默认 output")
    ap.add_argument("-s", "--server", default="http://127.0.0.1:8080/v1",
                    help="VLM 服务地址，默认 http://127.0.0.1:8080/v1")
    ap.add_argument("--json", action="store_true", help="同时保存 JSON")
    args = ap.parse_args()

    files = collect(args.inputs)
    if not files:
        sys.exit("没有可处理的文件")

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="vllm-server",
        vl_rec_server_url=args.server,
    )

    for f in files:
        print(f"\n===== {f} =====")
        save_dir = Path(args.out) / f.stem
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            for res in pipeline.predict(str(f)):
                res.print()
                res.save_to_markdown(save_path=str(save_dir))
                if args.json:
                    res.save_to_json(save_path=str(save_dir))
        except Exception as e:
            print(f"处理失败 {f}: {e}", file=sys.stderr)

    print(f"\n完成，结果在: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()