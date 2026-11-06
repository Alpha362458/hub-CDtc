"""数据处理、模型训练、评估和报告的一键入口。"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--evaluate-only", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.evaluate_only and not args.report_only:
        from train import train
        train()
    if not args.report_only:
        from evaluate import evaluate
        evaluate()
    from build_readme import build_readme
    build_readme()


if __name__ == "__main__":
    main()
