"""
本地展示服务：python server.py 后浏览器打开 http://localhost:8765。
纯标准库实现 —— 静态托管 web/，并把 eval_report/ 的 JSON 经 /api/* 提供给页面。
"""

import argparse
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
REPORT_DIR = os.path.join(ROOT, "eval_report")


class DashboardHandler(SimpleHTTPRequestHandler):
    """静态页 + 两个只读 JSON API（/api/eval、/api/attention）。"""

    def do_GET(self):
        if self.path == "/api/eval":
            return self._serve_json(os.path.join(REPORT_DIR, "report.json"))
        if self.path == "/api/attention":
            return self._serve_json(os.path.join(REPORT_DIR, "attention.json"))
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def _serve_json(self, path):
        if not os.path.exists(path):
            # 数据未生成时给出可操作的提示，而不是裸 404
            body = {"error": "未找到评估数据，请先运行: python evaluate.py"}
            return self._send(200, json.dumps(body, ensure_ascii=False), "application/json")
        with open(path, encoding="utf-8") as f:
            return self._send(200, f.read(), "application/json")

    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    handler = partial(DashboardHandler, directory=WEB_DIR)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"展示服务已启动: http://localhost:{args.port}")
        print("Ctrl+C 停止")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")


if __name__ == "__main__":
    main()
