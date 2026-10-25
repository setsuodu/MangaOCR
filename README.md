# AI漫画翻译

## 环境要求

- Nvidia 8G+ VRAM

## 安装模型

1. Python3.10
2. PaddleOCR-VL-1.6
```
pip install paddlepaddle paddleocr
```
3. LMStudio
- 翻译：Sakura-14B-Qwen2.5-v1.0-GGUF
```
hf download SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF sakura-14b-qwen2.5-v1.0-q4km.gguf
```
- 备选：Hy-MT2

## 部署步骤

1. 打开 LM Studio，加载 Sakura。
2. 到 Model Server 页面，把服务器状态切到 Running，确认地址是 http://127.0.0.1:1234。
3. CMD运行：
```
python manga_ocr.py --lang japan --llm-model sakura-14b-qwen2.5-v1.0-q4km
```
4. 浏览器会自动打开 http://127.0.0.1:7860，把日文漫画拖进去，等文字框出来，点"翻译"，再点"保存译图"。

> 输入
>![input](assets\manga_input.jpg)
>
> 输出
>![output](assets\manga_view.png)