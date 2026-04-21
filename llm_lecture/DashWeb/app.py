"""
app.py - Insurance Dashboard 메인 애플리케이션
layout.py에서 레이아웃을 임포트하여 사용
"""

import dash
from dash import Input, Output
from datetime import datetime

# ── layout.py에서 레이아웃 임포트 ────────────────────────────
from layout import layout

# ── 앱 초기화 ─────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Insurance Dashboard",
    update_title=None,
)

# 임포트한 레이아웃 적용
app.layout = layout


# ── 콜백: 실시간 시계 ─────────────────────────────────────────
@app.callback(
    Output("live-clock", "children"),
    Input("interval-clock", "n_intervals"),
)
def update_clock(_):
    now = datetime.now()
    return now.strftime("%Y-%m-%d  %H:%M:%S")


# ── 실행 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
