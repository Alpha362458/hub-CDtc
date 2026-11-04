"""在任意工作目录运行：python <本文件路径>。无需联网下载数据。"""
import argparse
import sys
import traceback

from src.common import ROOT, now


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="训练配置 JSON")
    parser.add_argument("--report-only", action="store_true", help="从已有结果重新生成图表及 README")
    args = parser.parse_args()
    if args.report_only:
        from src.report import build_report
        build_report()
        return
    from src.train import train_all
    from src.evaluate import evaluate_final
    from src.report import build_report
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs" / "run.log").open("w", encoding="utf-8") as log:
        original = sys.stdout
        sys.stdout = Tee(original, log)
        try:
            print(f"Run started {now()}")
            train_all(args.config)
            evaluate_final()
            build_report()
            print(f"Run finished {now()}")
        except Exception:
            traceback.print_exc(file=log)
            raise
        finally:
            sys.stdout = original


if __name__ == "__main__":
    main()
