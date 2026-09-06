# -*- coding: utf-8 -*-
"""figs/*.png 를 data URI 로 심어 결과 리포트 HTML 을 완성한다.

템플릿의 {{FIG:파일명}} 토큰을 base64 이미지로 치환한다. 그림과 문서가 따로 놀지 않도록
학습을 다시 돌린 뒤에는 이 스크립트를 다시 실행하면 된다.
"""
import base64
import re
import sys
from pathlib import Path

from ..common.paths import FIGS, ROOT

TEMPLATE = ROOT / "src" / "viz" / "report_template.html"
OUT = ROOT / "outputs" / "results_report.html"


def inject(template: Path, out: Path):
    html = template.read_text(encoding="utf-8")

    missing = []

    def sub(m):
        name = m.group(1)
        p = FIGS / name
        if not p.exists():
            missing.append(name)
            return ""
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    html = re.sub(r"\{\{FIG:([^}]+)\}\}", sub, html)
    if missing:
        print(f"[report] 누락된 그림: {missing}", file=sys.stderr)
    out.write_text(html, encoding="utf-8")
    print(f"[report] {out}  ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    inject(TEMPLATE, OUT)
