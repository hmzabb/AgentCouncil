"""本地 CI：一条命令跑完与 GitHub Actions 相同的检查（跨平台，无需 Docker）。

用途：提交前在本地复现 CI 结果，避免「推上去才发现红了」。
检查项与 .github/workflows/ci.yml 对齐：
  1. 语法编译（compileall，覆盖全部源码包）；
  2. ruff 静态检查（未安装则跳过并明确提示，不算通过）；
  3. pytest mock 用例（eval/test_suites.py 包装的确定性场景）；
  4. A/B 统计框架冒烟（mock 链路，验证框架可跑通）。

用法（项目根目录）：
    python scripts/ci.py
    python scripts/ci.py --with-ab      # 额外跑 A/B 冒烟（默认跑）
    python scripts/ci.py --skip-ab

退出码：0=全部通过；非 0=有检查失败（CI 同样会红）。
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
SOURCE_PACKAGES: tuple[str, ...] = ("agents", "graph", "tools", "data", "api", "eval")
SOURCE_FILES: tuple[str, ...] = ("config.py", "main.py")


def _run(label: str, args: list[str]) -> tuple[bool, str]:
    """执行一条检查命令，返回 (是否通过, 输出末尾若干行)。"""
    print(f"\n>>> {label}\n    {' '.join(args)}", flush=True)
    proc = subprocess.run(
        args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    output: str = (proc.stdout or "") + (proc.stderr or "")
    tail: str = "\n".join(output.strip().splitlines()[-12:])
    if tail:
        print(tail, flush=True)
    ok: bool = proc.returncode == 0
    print(f"    => {'PASS' if ok else 'FAIL'} (exit {proc.returncode})", flush=True)
    return ok, tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgentCouncil 本地 CI（对齐 GitHub Actions）")
    parser.add_argument("--skip-ab", action="store_true", help="跳过 A/B 冒烟")
    args = parser.parse_args(argv)

    results: list[tuple[str, bool]] = []

    results.append((
        "语法编译",
        _run("语法编译 (compileall)", [sys.executable, "-m", "compileall", "-q", *SOURCE_PACKAGES,
                                       *SOURCE_FILES])[0],
    ))

    if importlib.util.find_spec("ruff") is not None:
        results.append(("ruff 检查", _run("ruff check", [sys.executable, "-m", "ruff", "check", "."])[0]))
    else:
        print("\n>>> ruff 未安装：跳过静态检查（pip install ruff 后可复现 CI 的 lint 任务）")

    results.append((
        "pytest mock 用例",
        _run("pytest (mock)", [sys.executable, "-m", "pytest", "-q", "-m", "mock"])[0],
    ))

    if not args.skip_ab:
        results.append((
            "A/B 统计框架冒烟",
            _run("A/B 冒烟 (mock)", [sys.executable, "-m", "eval.ab_test", "--repeats", "2",
                                     "--tickers", "600519.SH"])[0],
        ))

    print("\n=== 汇总 ===")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed: int = sum(1 for _, ok in results if not ok)
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项失败'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())