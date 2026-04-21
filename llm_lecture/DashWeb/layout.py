"""
layout.py - Insurance Dashboard 레이아웃 정의
"""

from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── 공통 색상 팔레트 ─────────────────────────────────────────
COLORS = {
    "bg_dark":    "#040d1e",
    "bg_card":    "#071228",
    "bg_card2":   "#0a1a35",
    "border":     "#0e2a55",
    "accent":     "#00aaff",
    "accent2":    "#00d4ff",
    "text_main":  "#e8f4ff",
    "text_sub":   "#7ab3d4",
    "gold":       "#ffc107",
    "red":        "#ff4444",
    "green":      "#00e676",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text_sub"], size=10),
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
)

# ── 헬퍼 함수 ────────────────────────────────────────────────

def make_donut(value, label, color_main, color_sub="#1a3a5c"):
    fig = go.Figure(go.Pie(
        values=[value, 100 - value],
        hole=0.72,
        marker_colors=[color_main, color_sub],
        textinfo="none",
        hoverinfo="none",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        annotations=[dict(
            text=f"<b>{value}%</b>",
            x=0.5, y=0.5,
            font=dict(size=13, color=color_main),
            showarrow=False,
        )],
        height=110,
    )
    return fig


def make_trend_chart():
    x = list(range(1, 10))
    y = [2100, 3200, 2800, 3800, 4200, 5100, 5600, 6200, 7100]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill="tozeroy",
        fillcolor="rgba(0,170,255,0.15)",
        line=dict(color=COLORS["accent"], width=2),
        mode="lines+markers",
        marker=dict(size=6, color=COLORS["accent2"],
                    line=dict(width=1, color="white")),
    ))
    # 강조점(빨간 점)
    fig.add_trace(go.Scatter(
        x=[4], y=[3800],
        mode="markers",
        marker=dict(size=10, color=COLORS["red"],
                    line=dict(width=2, color="white")),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=170,
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=COLORS["text_sub"]),
                   tickvals=x),
        yaxis=dict(showgrid=True, gridcolor="#0e2a55",
                   tickfont=dict(size=9, color=COLORS["text_sub"])),
    )
    return fig


def make_sop_bar():
    projects = [f"P{i}" for i in range(1, 9)]
    vals     = [6.2, 7.8, 3.1, 2.0, 5.5, 4.8, 6.0, 5.2]
    colors   = [COLORS["accent"]] * 8
    colors[3] = COLORS["red"]   # P4 강조
    fig = go.Figure(go.Bar(
        x=projects, y=vals,
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=130,
        bargap=0.3,
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=COLORS["text_sub"])),
        yaxis=dict(showgrid=False, visible=False),
    )
    return fig


def make_funnel():
    labels = ["Direct", "Agency", "Broker", "Online", "Partner"]
    values = [100, 82, 64, 45, 28]
    colors_funnel = ["#003f88","#005cbf","#0077e6","#009aff","#33b8ff"]
    fig = go.Figure(go.Funnel(
        y=labels,
        x=values,
        marker_color=colors_funnel,
        textinfo="none",
        connector=dict(line=dict(color="rgba(0,0,0,0)")),
    ))
    fig.update_layout(**CHART_LAYOUT, height=130)
    return fig


# ── 테이블 데이터 ─────────────────────────────────────────────
svm_data = [
    {"Date": "2018-4-13", "Dept": "SVM", "Agent": "SVM", "Brand": "SVM", "State": "AL", "Company": "XX Insurance Co."},
    {"Date": "2018-4-13", "Dept": "SVM", "Agent": "SVM", "Brand": "SVM", "State": "HB", "Company": "XX Insurance Co."},
    {"Date": "2018-4-12", "Dept": "SVM", "Agent": "SVM", "Brand": "SVM", "State": "AL", "Company": "XX Insurance Co."},
    {"Date": "2018-4-12", "Dept": "SVM", "Agent": "SVM", "Brand": "SVM", "State": "HB", "Company": "XX Insurance Co."},
    {"Date": "2018-4-11", "Dept": "SVM", "Agent": "SVM", "Brand": "SVM", "State": "HB", "Company": "XX Insurance Co."},
]


def make_table(data):
    header_style = {
        "background": "#0e2a55",
        "color": COLORS["accent2"],
        "padding": "6px 8px",
        "fontSize": "10px",
        "fontWeight": "600",
        "letterSpacing": "0.5px",
        "borderBottom": f"1px solid {COLORS['border']}",
    }
    cell_style = {
        "padding": "5px 8px",
        "fontSize": "10px",
        "color": COLORS["text_sub"],
        "borderBottom": f"1px solid {COLORS['border']}",
        "whiteSpace": "nowrap",
    }
    headers = ["Date", "Dept", "Agent", "Brand", "State", "Company"]
    keys    = ["Date", "Dept", "Agent", "Brand", "State", "Company"]

    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(html.Tr([
                html.Th(h, style=header_style) for h in headers
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row[k], style={**cell_style,
                            "color": COLORS["text_main"] if k == "Date" else COLORS["text_sub"]})
                    for k in keys
                ]) for row in data
            ]),
        ],
    )


# ── 카드 래퍼 ─────────────────────────────────────────────────
def card(children, style=None):
    base = {
        "background": COLORS["bg_card"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "6px",
        "padding": "12px",
        "boxSizing": "border-box",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def section_title(text):
    return html.Div(text, style={
        "color": COLORS["text_main"],
        "fontSize": "12px",
        "fontWeight": "700",
        "letterSpacing": "0.8px",
        "marginBottom": "8px",
        "textTransform": "uppercase",
    })


def metric_label(text):
    return html.Div(text, style={
        "fontSize": "9px",
        "color": COLORS["text_sub"],
        "letterSpacing": "0.4px",
    })


def metric_value(text, big=False, color=None):
    return html.Div(text, style={
        "fontSize": "18px" if big else "13px",
        "fontWeight": "700",
        "color": color or COLORS["text_main"],
        "lineHeight": "1.1",
    })


# ── 최종 레이아웃 ──────────────────────────────────────────────
layout = html.Div(
    style={
        "background": COLORS["bg_dark"],
        "minHeight": "100vh",
        "fontFamily": "'Rajdhani', 'Orbitron', 'Segoe UI', sans-serif",
        "padding": "16px",
        "boxSizing": "border-box",
    },
    children=[

        # ── Google Font 로드 ──────────────────────────────
        html.Link(rel="stylesheet",
                  href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&display=swap"),

        # ── 타이틀 바 ─────────────────────────────────────
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "14px",
            },
            children=[
                html.Div(style={"width": "120px"}),
                html.Div("Insurance Dashboard", style={
                    "color": COLORS["text_main"],
                    "fontSize": "22px",
                    "fontWeight": "700",
                    "letterSpacing": "2px",
                    "textAlign": "center",
                    "textShadow": f"0 0 20px {COLORS['accent']}55",
                }),
                html.Div(id="live-clock", style={
                    "color": COLORS["text_sub"],
                    "fontSize": "11px",
                    "textAlign": "right",
                }),
            ],
        ),

        # ── 메인 그리드 ───────────────────────────────────
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "200px 1fr 200px",
                "gridTemplateRows": "auto auto auto",
                "gap": "10px",
            },
            children=[

                # ── 왼쪽 열 ──────────────────────────────

                # 차량 보험
                card(
                    style={"gridColumn": "1", "gridRow": "1"},
                    children=[
                        section_title("Vehicle Insurance"),
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "8px"},
                            children=[
                                html.Div(
                                    dcc.Graph(figure=make_donut(72, "", "#00aaff"), config={"displayModeBar": False}),
                                    style={"flex": "1"},
                                ),
                                html.Div([
                                    metric_label("Vehicles"),
                                    metric_value("18,600"),
                                    html.Div(style={"height": "6px"}),
                                    metric_label("Premium"),
                                    metric_value("16,280,600"),
                                ]),
                            ],
                        ),
                    ],
                ),

                # 기타 보험
                card(
                    style={"gridColumn": "1", "gridRow": "2"},
                    children=[
                        section_title("Other Insurance"),
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "8px"},
                            children=[
                                html.Div(
                                    dcc.Graph(figure=make_donut(55, "", "#ffc107"), config={"displayModeBar": False}),
                                    style={"flex": "1"},
                                ),
                                html.Div([
                                    metric_label("Number"),
                                    metric_value("18,600"),
                                    html.Div(style={"height": "6px"}),
                                    metric_label("Premium"),
                                    metric_value("16,280,600"),
                                ]),
                            ],
                        ),
                    ],
                ),

                # 월별 갱신 트렌드
                card(
                    style={"gridColumn": "1", "gridRow": "3"},
                    children=[
                        section_title("Monthly Renewal Trend"),
                        dcc.Graph(
                            figure=make_trend_chart(),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),

                # ── 중앙 열 ──────────────────────────────

                # 일일 통계
                card(
                    style={"gridColumn": "2", "gridRow": "1"},
                    children=[
                        section_title("Daily Statistics"),
                        html.Div(
                            style={"display": "flex", "gap": "30px"},
                            children=[
                                html.Div([
                                    html.Div("🚗 Vehicle Insurance", style={"fontSize": "10px", "color": COLORS["text_sub"]}),
                                    metric_value("186,000", big=True, color=COLORS["accent"]),
                                    html.Div("¥ 1,216,280,600", style={"fontSize": "9px", "color": COLORS["text_sub"], "marginTop": "4px"}),
                                ]),
                                html.Div([
                                    html.Div("🛡️ Other Insurance", style={"fontSize": "10px", "color": COLORS["text_sub"]}),
                                    metric_value("65,000", big=True, color=COLORS["gold"]),
                                    html.Div("¥ 15,280,600", style={"fontSize": "9px", "color": COLORS["text_sub"], "marginTop": "4px"}),
                                ]),
                            ],
                        ),
                    ],
                ),

                # SVM 부서 도넛
                card(
                    style={"gridColumn": "2", "gridRow": "2"},
                    children=[
                        section_title("Department of SVM Business"),
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-around"},
                            children=[
                                html.Div([
                                    dcc.Graph(figure=make_donut(68, "", "#00aaff"), config={"displayModeBar": False}),
                                    html.Div("Proportion of Agents", style={"fontSize": "9px", "color": COLORS["text_sub"], "textAlign": "center"}),
                                ], style={"flex": "1"}),
                                html.Div([
                                    dcc.Graph(figure=make_donut(45, "", "#00e676"), config={"displayModeBar": False}),
                                    html.Div("Proportion of Direct", style={"fontSize": "9px", "color": COLORS["text_sub"], "textAlign": "center"}),
                                ], style={"flex": "1"}),
                                html.Div([
                                    dcc.Graph(figure=make_donut(82, "", "#ffc107"), config={"displayModeBar": False}),
                                    html.Div("Proportion of Brokers", style={"fontSize": "9px", "color": COLORS["text_sub"], "textAlign": "center"}),
                                ], style={"flex": "1"}),
                            ],
                        ),
                    ],
                ),

                # 대형 수치
                html.Div(
                    style={
                        "gridColumn": "2",
                        "gridRow": "3",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                    },
                    children=[
                        html.Div("1,044,000", style={
                            "fontSize": "52px",
                            "fontWeight": "700",
                            "color": COLORS["accent"],
                            "textShadow": f"0 0 30px {COLORS['accent']}88, 0 0 60px {COLORS['accent']}44",
                            "letterSpacing": "3px",
                        }),
                    ],
                ),

                # ── 오른쪽 열 ─────────────────────────────

                # SOP Project Team 바 차트
                card(
                    style={"gridColumn": "3", "gridRow": "1"},
                    children=[
                        section_title("SOP Project Team"),
                        dcc.Graph(figure=make_sop_bar(), config={"displayModeBar": False}),
                    ],
                ),

                # SVP Sale Plan 퍼널
                card(
                    style={"gridColumn": "3", "gridRow": "2"},
                    children=[
                        section_title("SVP Sale Plan"),
                        dcc.Graph(figure=make_funnel(), config={"displayModeBar": False}),
                    ],
                ),

                # SVM 테이블
                card(
                    style={"gridColumn": "3", "gridRow": "3", "overflowX": "auto"},
                    children=[
                        section_title("SVM"),
                        make_table(svm_data),
                    ],
                ),
            ],
        ),

        # ── 실시간 클럭 인터벌 ────────────────────────────
        dcc.Interval(id="interval-clock", interval=1000, n_intervals=0),
    ],
)
