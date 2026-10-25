python ocr.py a.png
python ocr.py a.png b.jpg c.png
python ocr.py D:\scans                  # 整个文件夹
python ocr.py a.png -o result --json    # 指定输出目录并保存 JSON
python ocr.py a.png -s http://127.0.0.1:8080/v1

# 带翻译启动（需要 LMStudio启动服务器）
```cmd
python manga_ocr.py --lang japan --llm-model sakura-14b-qwen2.5-v1.0-q4km
```
