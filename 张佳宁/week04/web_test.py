"""本地模型测试页面：python web_test.py，然后打开 http://127.0.0.1:8000。"""

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


from inference import (
    DEFAULT_CHECKPOINT, generate_text, load_checkpoint,
    next_character_candidates, select_device,
)


class ModelService:
    def __init__(self, checkpoint, device):
        self.path = Path(checkpoint).resolve()
        self.device = device
        self.lock = threading.Lock()
        self.model = None
        self.signature = None

    def load(self):
        # 训练完成或权重更新后，下一个请求自动重新加载。
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            raise ValueError(f"找不到模型 {self.path.name}。请先训练，或启动时用 --checkpoint 指定 TransformerEncoder 权重。") from None
        signature = (stat.st_mtime_ns, stat.st_size)
        if self.model is None or signature != self.signature:
            model, chars, reverse = load_checkpoint(self.path, self.device)
            self.model, self.chars, self.reverse = model, chars, reverse
            self.signature = signature

    def status(self):
        with self.lock:
            try:
                self.load()
            except (ValueError, RuntimeError, OSError, KeyError) as error:
                return {"ready": False, "checkpoint": self.path.name, "device": str(self.device), "error": str(error)}
            return {
                "ready": True, "checkpoint": self.path.name, "device": str(self.device),
                "vocab_size": len(self.chars), "context_length": self.model.position_embed.num_embeddings,
                "layers": len(self.model.tf_ers.layers),
                "parameters": sum(p.numel() for p in self.model.parameters()),
            }

    def generate(self, data):
        if not isinstance(data, dict):
            raise ValueError("请求内容必须是 JSON 对象。")
        prompt = data.get("prompt", "")
        length = data.get("length", 100)
        temperature = data.get("temperature", 0.8)
        top_k = data.get("top_k", 20)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
            raise ValueError("请输入 1～4000 个字符，不能全部为空白。")
        if type(length) is not int or not 1 <= length <= 500:
            raise ValueError("生成长度必须是 1～500 的整数。")
        if type(top_k) is not int or not 0 <= top_k <= 1000:
            raise ValueError("Top-K 必须是 0～1000 的整数。")
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("温度必须在 0～2 之间。")
        with self.lock:
            self.load()
            started = time.perf_counter()
            result = generate_text(self.model, self.chars, self.reverse, prompt, length, temperature, top_k)
            candidates = next_character_candidates(
                self.model, self.chars, self.reverse, prompt,
            )
            return {"prompt": prompt, "continuation": result[len(prompt):], "text": result,
                    "seconds": round(time.perf_counter() - started, 3), "length": length,
                    "truncated": len(prompt) > self.model.position_embed.num_embeddings,
                    "candidates": candidates}


def make_handler(service):
    page = Path(__file__).resolve().with_name("model_test.html")

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, value, content_type="application/json; charset=utf-8"):
            payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path in ("/", "/model_test.html"):
                self.respond(200, page.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self.respond(200, service.status())
            else:
                self.respond(404, {"error": "页面不存在。"})

        def do_POST(self):
            if self.path != "/api/generate":
                self.respond(404, {"error": "接口不存在。"})
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.respond(403, {"error": "请从本地测试页面发起请求。"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32768:
                    raise ValueError("请求大小无效。")
                data = json.loads(self.rfile.read(size))
                self.respond(200, service.generate(data))
            except (ValueError, UnicodeDecodeError) as error:
                self.respond(400, {"error": str(error)})
            except Exception as error:
                print(f"生成失败：{error}", flush=True)
                self.respond(500, {"error": "模型运行失败，请检查终端中的错误信息。"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port 必须在 1～65535 之间。")
    device = select_device(args.device)
    service = ModelService(args.checkpoint, device)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    except OSError as error:
        parser.error(f"启动失败：{error}；可以用 --port 更换端口。")
    print(f"测试页面：http://127.0.0.1:{args.port}\n模型：{service.path}\n设备：{device}\n按 Ctrl+C 停止服务。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
