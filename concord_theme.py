"""Concord's aurora presentation layer. No analysis or application state lives here."""
from html import escape, unescape
import base64
from math import cos, sin, radians, isfinite
from uuid import uuid4
from math import floor, log, tan, pi, asinh

import streamlit as st
import plotly.graph_objects as go
import requests


CSS = """
:root {
  --bg-0:#0F1535;--bg-1:#151B45;--surface:rgba(255,255,255,.07);
  --surface-2:rgba(255,255,255,.07);--surface-hover:rgba(255,255,255,.11);
  --text:#F5F7FF;--text-muted:#B9BFD3;--text-faint:#AAB1CA;
  --accent:#FF8F5A;--accent-ink:#281509;--accent-2:#F5B79B;--warn:#F0C66C;--alert:#FF7A7A;--series-4:#79B9EC;
  --radius-card:16px;--radius-inner:16px;--radius-pill:16px;--space:8px;
  --concord-font:"Segoe UI Variable","Segoe UI",system-ui,Inter,sans-serif;--heading:var(--concord-font);
}
html {scroll-padding-top:64px;}
body,.stApp,[data-testid="stAppViewContainer"] {color:var(--text);font-family:var(--concord-font);background:linear-gradient(180deg,var(--bg-0),var(--bg-1));background-attachment:fixed;}
[data-testid="stHeader"],[data-testid="stToolbar"] {background:transparent;}
[data-testid="stMain"] {scroll-padding-top:64px;}
footer {display:none;}
[data-testid="stMainBlockContainer"] {max-width:1520px;padding:3.5rem 32px 32px;}
h1,h2,h3,h4,h5,h6,[data-testid="stMetricValue"] {font-family:var(--concord-font)!important;color:var(--text);letter-spacing:normal!important;}
h1 {font-size:2rem!important;font-weight:600!important;}
h2,h3,.section-heading h2 {font-size:1.25rem!important;font-weight:600!important;}
p,li,label,button,input,textarea {font-family:var(--concord-font);line-height:1.5;}
[data-testid="stCaptionContainer"] {opacity:1!important;}
[data-testid="stCaptionContainer"] p,.stCaption p {color:var(--text-muted)!important;font-size:.875rem;}
code,.mono {font-family:var(--concord-font)!important;font-size:.875rem;}
code {background:transparent;color:var(--text);padding:0;}
a {color:var(--accent);}
a:focus-visible,button:focus-visible,input:focus-visible,textarea:focus-visible,summary:focus-visible,[tabindex]:focus-visible {outline:2px solid var(--accent)!important;outline-offset:3px!important;}
[data-testid="stSidebar"] {background:#101630;border:0;}
[data-testid="stSidebar"] h3 {font-size:1.25rem!important;}
[data-testid="stWidgetLabel"] p {font-size:.875rem;font-weight:400;color:var(--text-muted);}
[data-baseweb="input"],[data-baseweb="select"]>div,textarea {background:var(--surface)!important;color:var(--text)!important;border:0!important;border-radius:16px!important;min-height:44px;box-shadow:none;}
[data-baseweb="base-input"] {border:0!important;background:transparent!important;}
input,textarea {font-family:var(--concord-font)!important;}
[data-baseweb="select"] span {color:var(--text);}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="menu"] {background:#242C47!important;border:0;border-radius:16px;}
[data-baseweb="tag"] {background:var(--surface-hover)!important;color:var(--text)!important;border-radius:16px!important;}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"] button {min-height:44px;border-radius:16px!important;background:var(--accent)!important;color:var(--accent-ink)!important;border:0!important;font-family:var(--concord-font);font-weight:600;padding:8px 16px;box-shadow:none!important;white-space:normal!important;}
.stButton>button p,.stDownloadButton>button p {color:inherit!important;font-weight:600;font-size:.875rem;line-height:1.4;white-space:normal!important;}
.stButton>button:hover {filter:brightness(1.08);}
.stDownloadButton>button,[class*="st-key-load_watch_"] button,[data-testid="stSidebar"] .stButton>button {color:var(--text)!important;background:var(--surface)!important;}
.stDownloadButton>button::before {content:'↓';font-size:1rem;}
.stButton>button:disabled {opacity:.55;}
[data-testid="stTabs"] [role="tablist"] {gap:8px;background:transparent;border:0;flex-wrap:wrap;height:auto;}
[data-testid="stTabs"] [role="tab"] {border-radius:16px;min-height:44px;padding:8px 16px;background:var(--surface);color:var(--text-muted);border:0;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {background:var(--accent);color:var(--accent-ink);}
[data-testid="stTabs"] [role="tab"] p {font-weight:600;font-size:.875rem;color:inherit;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],[data-testid="stTabs"] [data-baseweb="tab-border"] {display:none;}
[data-testid="stRadio"] [role="radiogroup"] {display:flex;flex-wrap:wrap;gap:8px;}
[data-testid="stRadio"] [role="radiogroup"] label {margin:0!important;min-height:44px;border:0;border-radius:16px;padding:8px 12px;background:var(--surface);}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {background:var(--accent);color:var(--accent-ink);}
[data-testid="stRadio"] label:has(input:focus-visible) {outline:2px solid var(--accent);outline-offset:3px;}
[data-testid="stRadio"] label>div:first-child {position:absolute;opacity:0;width:1px;}
[data-testid="stRadio"] label p {font-size:.875rem;font-weight:500;color:inherit;}
[data-testid="stMetric"],[data-testid="stFileUploader"],[data-testid="stPlotlyChart"],.var-card,.location-chip,.empty-state,.hero-card,.stat-tile,.led-tile,.st-key-simulator-controls,.table-wrap {background:var(--surface);border:0;border-radius:16px;box-shadow:none;}
[data-testid="stExpander"] details {background:none;border:0!important;border-radius:0!important;box-shadow:none;}
[data-testid="stExpander"] summary {min-height:44px;padding:0;color:var(--text-muted);}
[data-testid="stExpander"] summary p {font-size:.875rem;}
[data-testid="stPlotlyChart"] {padding:0;overflow:hidden;}
[data-testid="stMetric"] {padding:20px;min-height:148px;}
[data-testid="stMetricLabel"] p {font-size:.875rem;color:var(--text-muted);}
[data-testid="stMetricValue"] {font-size:1.5rem!important;white-space:normal;overflow-wrap:anywhere;}
[data-testid="stMetricDelta"] {font-family:var(--concord-font);font-size:.875rem;color:var(--accent);}
[data-testid="stFileUploader"] {padding:12px;}
[data-testid="stFileUploaderDropzone"] {background:transparent;border:0;border-radius:16px;}
[data-testid="stSlider"] [role="slider"] {background:var(--accent)!important;box-shadow:none;}
.st-key-simulator-controls {padding:20px;}
.st-key-user_location input {padding-left:40px!important;}
.st-key-user_location [data-baseweb="input"] {position:relative;}
.st-key-user_location [data-baseweb="input"]::before {content:'';position:absolute;left:16px;top:14px;width:12px;height:12px;border:1.75px solid var(--text-muted);border-radius:50%;}
.st-key-user_location [data-baseweb="input"]::after {content:'';position:absolute;left:27px;top:27px;width:6px;border-top:1.75px solid var(--text-muted);transform:rotate(45deg);}
hr {display:none;}
.concord-nav {display:flex;align-items:center;gap:24px;padding:0;margin:0 0 20px;}
.wordmark {display:inline-flex;align-items:center;gap:8px;font:600 1rem var(--concord-font);letter-spacing:normal;color:var(--text);text-decoration:none;flex-shrink:0;min-height:44px;}
.wordmark img {width:24px;height:24px;}
.nav-links {display:flex;align-items:center;gap:4px;overflow-x:auto;scrollbar-width:thin;min-width:0;}
.nav-links a {font-size:.875rem;color:var(--text-muted);padding:12px 8px;text-decoration:none;white-space:nowrap;min-height:44px;}
.nav-links a:hover {color:var(--text);}
.section-heading {scroll-margin-top:64px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:24px 0 8px;}
.section-heading h2 {font:600 1.25rem var(--concord-font);letter-spacing:normal;margin:0;}
#observatory {margin:0;}#observatory h2 {font-size:1.25rem!important;}
.info-note {position:relative;display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;cursor:help;vertical-align:middle;flex-shrink:0;color:var(--text-muted);}
.info-symbol {border:1px solid currentColor;border-radius:50%;width:16px;height:16px;font:12px/16px var(--concord-font);text-align:center;}
.info-copy {display:none;position:absolute;z-index:100;top:100%;right:0;width:320px;max-width:75vw;padding:16px;background:#29334F;color:var(--text);border-radius:16px;font:400 .875rem/1.6 var(--concord-font);letter-spacing:normal;}
.info-note:hover .info-copy,.info-note:focus .info-copy,.info-note:focus-within .info-copy {display:block;}
.plain-summary,.telemetry-banner {background:none;border:0;border-radius:0;box-shadow:none;padding:8px 0;color:var(--text-muted);font:.875rem/1.6 var(--concord-font);overflow-wrap:anywhere;}
.plain-summary strong,.telemetry-banner strong {font-weight:500;color:var(--text);}
.statistics-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;align-items:stretch;}
.var-card {padding:20px;display:flex;flex-direction:column;gap:12px;min-width:0;overflow-wrap:anywhere;}
.var-card .plain-summary {margin-top:auto;font-size:.875rem;padding:0;}
.var-role,.var-meta,.var-plain-what {font:.875rem/1.5 var(--concord-font);color:var(--text-muted);margin-bottom:8px;}
.var-title {font:600 1rem/1.5 var(--concord-font);margin-bottom:8px;}
.var-card details {color:var(--text-muted);font-size:.875rem;line-height:1.5;}
details summary {min-height:44px;cursor:pointer;}
.var-card details summary {color:var(--text-muted);display:flex;align-items:center;gap:8px;}
.var-card details summary::after {content:'+';}
.var-card details[open] summary::after {content:'−';}
.var-metric-box {padding:0;margin:16px 0;background:none;border:0;}
.var-metric-label,.tile-label {font:400 .875rem/1.5 var(--concord-font);color:var(--text-muted);}
.var-metric-val {font:400 1.5rem/1.3 var(--concord-font);letter-spacing:normal;margin:8px 0;color:var(--text);}
.var-metric-ci,.var-stats-grid {font:.875rem/1.6 var(--concord-font);color:var(--text-muted);}
.var-stats-grid {margin:12px 0;}
.status-chip,.var-verdict-sig,.var-verdict-emerging,.var-verdict-noise {display:inline-flex;align-items:center;gap:6px;font:500 .875rem/1.5 var(--concord-font);border:0;background:none;color:var(--text-muted);max-width:100%;overflow-wrap:anywhere;}
.status-chip.significant,.var-verdict-sig {color:#8DD09B;}
.status-chip.emerging,.var-verdict-emerging {color:var(--warn);}
.status-chip img {width:18px;height:18px;flex-shrink:0;}
.tutorial-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:24px;}
.tutorial-step {padding:0;min-width:0;background:none;}
.tutorial-step-num {font:.875rem var(--concord-font);color:var(--text-muted);margin-bottom:8px;}
.tutorial-step-title {font:600 1rem var(--concord-font);margin-bottom:8px;}
.tutorial-step .plain-summary {padding:0;}
.location-chip {padding:16px;min-height:0;}
.location-chip.selected {background:var(--surface-hover);}
.location-name {display:flex;align-items:center;gap:8px;font:500 .875rem/1.5 var(--concord-font);}
.location-name img {width:16px;min-width:16px;}
.location-value {font:.875rem var(--concord-font);margin-top:8px;}
.location-chip details {font:.875rem/1.5 var(--concord-font);color:var(--text-muted);}
.location-chip summary {min-height:44px;display:flex;align-items:center;cursor:pointer;}
.hero-card {padding:24px;margin:8px 0 16px;scroll-margin-top:64px;}
.hero-top {display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:16px;flex-wrap:wrap;}
.hero-location {font:600 1.25rem var(--concord-font);display:flex;align-items:center;gap:8px;}
.hero-location img {width:24px;height:24px;flex-shrink:0;}
.hero-zones {display:grid;grid-template-columns:1fr 1.2fr .9fr;gap:24px;align-items:center;}
.hero-value {font:300 5rem/1.1 var(--concord-font);letter-spacing:normal;margin:8px 0;color:var(--text);}
.hero-unit {font:.875rem/1.5 var(--concord-font);color:var(--text-muted);}
.hero-sparkline {min-width:0;}
.hero-sparkline img,.mini-geo img,.tile-sparkline img {display:block;width:100%;height:auto;}
.spark-caption {display:flex;justify-content:space-between;font:.875rem var(--concord-font);color:var(--text-muted);gap:8px;}
.mini-geo {border:0;padding:0;background:none;}
.hero-stats {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:16px;margin-top:24px;}
.hero-stats .stat-tile {background:none;padding:0;}
.stat-tile {padding:20px;min-width:0;}
.stat-value {font:600 1rem/1.5 var(--concord-font);color:var(--text);margin-top:8px;overflow-wrap:anywhere;}
.stat-note {font:.875rem/1.5 var(--concord-font);color:var(--text-muted);margin-top:8px;}
.empty-state {padding:24px;display:flex;align-items:center;gap:16px;}
.empty-state img {width:32px;height:32px;flex-shrink:0;}
.empty-state h2 {font:600 1.25rem var(--concord-font);margin:0 0 8px;}
.empty-state p {color:var(--text-muted);font-size:.875rem;margin:0;}
.table-wrap {padding:8px 16px;overflow-x:auto;}
.concord-table {width:100%;border-collapse:collapse;font:.875rem/1.5 var(--concord-font);color:var(--text-muted);text-align:left;}
.concord-table caption {text-align:left;padding:12px 0;color:var(--text);font-weight:600;}
.concord-table th,.concord-table td {font-size:.875rem;font-weight:400;padding:12px 8px;border:0;border-bottom:1px solid rgba(255,255,255,.07);}
.concord-table tr:last-child td {border:0;}
.arc-gauge {display:block;width:120px;max-width:100%;margin:0 auto;}
.gauge-tile {display:flex;align-items:center;gap:16px;min-height:148px;}
.gauge-tile .arc-gauge {width:110px;flex-shrink:0;}
.led-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:16px;}
.led-tile {padding:20px;min-width:0;}
.led-dot {width:12px;height:12px;display:block;border-radius:50%;border:2px solid var(--led-color);margin-bottom:16px;}
.led-dot.on {background:var(--led-color);}
.led-dot.off {border-color:#77819C;}
.led-title {font:600 .875rem var(--concord-font);color:var(--text);margin-bottom:8px;}
.led-state,.led-trigger {font:.875rem/1.5 var(--concord-font);color:var(--text-muted);margin-bottom:8px;}
.device-link {display:flex;align-items:flex-start;gap:8px;font:.875rem/1.5 var(--concord-font);color:var(--text-muted);padding:16px 0;overflow-wrap:anywhere;}
.device-link img {width:18px;height:18px;flex-shrink:0;}
.concord-footer {display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;padding:24px 0;margin-top:32px;color:var(--text-muted);font:.875rem/1.5 var(--concord-font);}
@media(max-width:1200px) {
  .statistics-grid,.led-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
  .hero-zones {grid-template-columns:1fr 1fr;}.mini-geo {grid-column:1/-1;max-width:360px;width:100%;justify-self:center;}
  .tutorial-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
  .concord-nav {gap:16px;}
}
@media(max-width:640px) {
  [data-testid="stMainBlockContainer"] {padding:3.5rem 16px 24px;}
  .statistics-grid,.led-grid,.tutorial-grid,.hero-zones {grid-template-columns:minmax(0,1fr);}
  .hero-card {padding:20px;}.hero-stats {grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;}
  .hero-value {font-size:4rem;}
  .concord-nav {flex-direction:column;align-items:stretch;gap:0;margin-bottom:12px;}
  .nav-links a {padding:12px 8px;}
  .empty-state {padding:20px;align-items:flex-start;}
  .var-card {padding:20px;}.section-heading h2 {font-size:1.25rem;}
  .info-copy {position:fixed;top:auto;left:16px;right:16px;width:auto;max-width:none;}
  .table-wrap {padding:8px;}
}
@media(prefers-reduced-motion:reduce) {html {scroll-behavior:auto;}*,*::before,*::after {animation:none!important;transition:none!important;}}

.var-title {display:flex;align-items:center;gap:10px;}
.variable-icon {width:32px;height:32px;flex-shrink:0;}
.status-icon {width:18px;height:18px;flex-shrink:0;}

.nav-title {display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.product-title {font-size:1rem;color:var(--text-muted);white-space:nowrap;}
.concord-nav {flex-wrap:wrap;justify-content:space-between;gap:0 16px;}
.context-stamp {font-size:.875rem;color:var(--text-muted);margin:0 0 24px;}
[class*="st-key-watchlist-chip-"] {background:var(--surface);border-radius:16px;padding:12px 16px;position:relative;max-width:100%;}
[class*="st-key-watchlist-chip-"]:has(.selected) {background:var(--surface-hover);}
[class*="st-key-watchlist-chip-"] .location-chip {background:none;padding:0 64px 0 0;}
[class*="st-key-watchlist-chip-"] [class*="st-key-load_watch_"] {position:absolute;top:12px;right:12px;width:64px!important;}
.location-name>span:last-child {max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.watch-dot {width:8px;height:8px;border-radius:50%;background:#ADB5CE;flex-shrink:0;}
.watch-dot.significant {background:#90B879;}.watch-dot.emerging {background:#F0C66C;}
.location-chip details[open] {min-width:280px;}
.location-chip .location-value {white-space:nowrap;font-size:.875rem;}
.location-chip summary {font-size:.875rem;}
.chart-facts {padding:8px 0;}.chart-facts .stat-tile {background:none;padding:12px 0;}
[data-testid="stExpander"] summary {width:fit-content;}
.stButton,.stDownloadButton {max-width:100%;}
.stButton>button,.stDownloadButton>button {width:auto!important;max-width:100%;text-align:left;}


.st-key-current-result {background:var(--surface);border-radius:16px;padding:24px;}
.st-key-current-result .hero-card {background:none;padding:0;margin:0;}
.st-key-current-result [data-testid="stPlotlyChart"] {background:none;}
.hero-zones {grid-template-columns:1fr 1fr;}
.map-source {font-size:.875rem;color:var(--text-muted);}.map-source a {color:inherit;}
@media(max-width:640px) {.st-key-current-result {padding:20px;}.hero-zones {grid-template-columns:1fr;}.hero-stats {grid-template-columns:1fr 1fr;}}


.series-key {font-size:.875rem;color:var(--text-muted);margin:0;}
.series-key summary {min-height:40px;display:flex;align-items:center;cursor:pointer;gap:8px;}
.series-key summary::after {content:'+';}.series-key[open] summary::after {content:'−';}
.series-key>div {display:flex;gap:8px 16px;flex-wrap:wrap;}
.series-item {display:inline-flex;align-items:center;gap:8px;}
.series-swatch {width:16px;height:3px;flex-shrink:0;}


/* Existing verdict changes only a small amount of the navy background. */
.stApp:has(.hero-card[data-verdict="significant"]) {--bg-0:#111A35;}
.stApp:has(.hero-card[data-verdict="emerging"]) {--bg-0:#1A1837;}


.st-key-current-result {container-type:inline-size;}
@container(max-width:800px) {
  .st-key-current-result>[data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
  .st-key-current-result>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"] {width:100%!important;flex:1 1 100%!important;min-width:0!important;}
}
@media(max-width:1200px) {.hero-stats {grid-template-columns:repeat(3,minmax(0,1fr));}}
@media(max-width:640px) {.hero-stats {grid-template-columns:repeat(2,minmax(0,1fr));}}

"""


def esc(value):
    return escape(str(value), quote=True)


def safe_prose(value):
    """Escape dynamic prose, retaining only the engine's strong/code emphasis."""
    value = esc(unescape(str(value)))
    for tag in ("strong", "code"):
        value = value.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    return value


def inject_theme():
    st.html("<style>" + CSS + "</style>")


def inline_svg(svg, label="", class_name=""):
    """Keep SVGs inline as image data: st.html's HTML-only sanitizer drops svg tags."""
    svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    payload = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="{esc(class_name)}" alt="{esc(label)}" src="data:image/svg+xml;base64,{payload}" />'


def icon(name="orbit", color="#FF8F5A"):
    paths = {
        "orbit": '<ellipse cx="12" cy="12" rx="10" ry="4.8" transform="rotate(-35 12 12)"/><circle cx="12" cy="12" r="6.5"/><path d="M8 13l3-4 5 6"/>',
        "pin": '<path d="M19 10c0 5-7 11-7 11S5 15 5 10a7 7 0 1 1 14 0Z"/><circle cx="12" cy="10" r="2.5"/>',
        "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
        "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
        "settings": '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2"/><circle cx="15" cy="12" r="2"/><circle cx="8" cy="18" r="2"/>',
        "link": '<path d="m10 14 4-4m-7 6-1 1a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0m2 2 1-1a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0" transform="translate(2 0) scale(.9)"/>',
        "upload": '<path d="M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6"/>',
    }
    return inline_svg('<svg viewBox="0 0 24 24" fill="none" stroke="' + esc(color) + '" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">' + paths.get(name, paths["orbit"]) + '</svg>')


def nav_bar(description=""):
    links = [("inquiry", "Inquiry"), ("significance", "Significance"), ("time-series", "Time-series"),
             ("coupling", "Coupling"), ("simulator", "Simulator"), ("telemetry", "Telemetry")]
    return '<nav class="concord-nav" aria-label="Dashboard"><div class="nav-title"><a class="wordmark" id="observatory" href="#observatory">' + icon() + 'CONCORD</a><span class="product-title">Earth System Detective</span>' + info_note(description) + '</div><div class="nav-links">' + ''.join(f'<a href="#{esc(key)}">{esc(label)}</a>' for key, label in links) + '</div></nav>'


def section_header(chapter, title, anchor):
    return f'<div class="section-heading" id="{esc(anchor)}"><h2>{esc(title)}</h2></div>'


def stat_tile(label, value, note=""):
    return f'<div class="stat-tile"><div class="tile-label">{esc(label)}</div><div class="stat-value">{esc(value)}</div>' + (f'<div class="stat-note">{esc(note)}</div>' if note else '') + '</div>'


def status_chip(label, state="unconfirmed"):
    state = state if state in {"significant", "emerging", "unconfirmed"} else "unconfirmed"
    return f'<span class="status-chip {state}">' + status_icon(state) + esc(label) + '</span>'


def arc_gauge_svg(percent):
    """270-degree display arc; the supplied percentage is already computed."""
    value = float(percent)
    value = min(100., max(0., value)) if isfinite(value) else 0.
    uid = "arc" + uuid4().hex
    ticks = []
    for i in range(11):
        angle = radians(135 + 270 * i / 10)
        ticks.append(f'<path d="M{60+45*cos(angle):.2f} {60+45*sin(angle):.2f}L{60+49*cos(angle):.2f} {60+49*sin(angle):.2f}"/>')
    svg = (f'<svg viewBox="0 0 120 112" role="img" aria-label="Feature importance {value:.1f} percent">'
            f'<defs><linearGradient id="{uid}"><stop stop-color="#FF8F5A"/><stop offset="1" stop-color="#F5B79B"/></linearGradient></defs>'
            '<g fill="none" stroke="rgba(235,240,255,.42)" stroke-width="1">' + ''.join(ticks) + '</g>'
            '<path d="M33.13 86.87 A38 38 0 1 1 86.87 86.87" fill="none" stroke="rgba(255,255,255,.09)" stroke-width="7" stroke-linecap="round"/>'
            f'<path d="M33.13 86.87 A38 38 0 1 1 86.87 86.87" fill="none" stroke="url(#{uid})" stroke-width="7" stroke-linecap="round" pathLength="100" stroke-dasharray="{value:.2f} 100"/>'
            f'<text x="60" y="64" text-anchor="middle" fill="#F5F7FF" font-family="Segoe UI Variable,Segoe UI,system-ui,Inter,sans-serif" font-size="20">{value:.1f}%</text>'
            '<text x="60" y="82" text-anchor="middle" fill="#AEBBD3" font-family="Segoe UI Variable,Segoe UI,system-ui,Inter,sans-serif" font-size="8">IMPORTANCE</text></svg>')
    return inline_svg(svg, f"Feature importance {value:.1f} percent", "arc-gauge")


def sparkline_svg(values, label="Annual observations", color="#FF8F5A"):
    """Map each observation to SVG coordinates, with straight segments and gaps."""
    vals = [float(v) for v in values]
    valid = [v for v in vals if isfinite(v)]
    if not valid:
        return '<span class="tile-label">No finite observations to draw.</span>'
    lo, hi = min(valid), max(valid)
    extent = hi - lo or 1
    commands, last = [], None
    gap = True
    for i, value in enumerate(vals):
        if not isfinite(value):
            gap = True
            continue
        x, y = 8 + i * 304 / max(1, len(vals) - 1), 104 - (value - lo) * 88 / extent
        commands.append(f'{"M" if gap else "L"}{x:.2f},{y:.2f}')
        last, gap = (x, y), False
    uid = "spark" + uuid4().hex
    path = ' '.join(commands)
    area = f'{path} L{last[0]:.2f},120 L8,120Z' if len(valid) == len(vals) else ''
    svg = (f'<svg viewBox="0 0 320 128" role="img" aria-label="{esc(label)}">'
            f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="{esc(color)}" stop-opacity=".25"/><stop offset="1" stop-color="{esc(color)}" stop-opacity="0"/></linearGradient></defs>'
            '<path d="M8 40H312M8 80H312M8 120H312" stroke="rgba(255,255,255,.06)" stroke-width="1"/>'
            f'<path d="{area}" fill="url(#{uid})"/><path d="{path}" fill="none" stroke="{esc(color)}" stroke-width="2" stroke-linejoin="round"/>'
            f'<circle cx="{last[0]:.2f}" cy="{last[1]:.2f}" r="4" fill="{esc(color)}"/></svg>')
    return inline_svg(svg, label)


def mini_geo_svg(lat, lon):
    """A schematic coordinate map, with no fetched or hotlinked map assets."""
    x, y = 20 + (float(lon) + 180) / 360 * 280, 15 + (90 - float(lat)) / 180 * 140
    svg = ('<svg viewBox="0 0 320 174" role="img" aria-label="Schematic latitude and longitude map">'
            '<rect x="20" y="15" width="280" height="140" rx="16" fill="#102B40"/>'
            '<path d="M20 50H300M20 85H300M20 120H300M90 15V155M160 15V155M230 15V155" fill="none" stroke="#294255" stroke-width="1"/>'
            '<path d="m44 43 22-15 34 8 12 20-23 13-10 19-18-16Zm49 46 17-4 20 25-12 34-14-12Zm45-48 26-11 13 13-11 14-24 5Zm16 22 22 4 11 24-20 30-13-25Zm25-29 37-11 58 21-9 26-32 8-22-16-28 4Zm63 77 25-9 17 20-22 10Z" fill="#294958" stroke="#426275" stroke-width="1"/>'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="12" fill="#FF8F5A" opacity=".15"/><circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#FF8F5A"/>'
            '</svg>')
    return inline_svg(svg, "Schematic latitude and longitude map")


def hero_card(meta, intel, variables):
    target = intel["target_var"]
    stat = intel["var_stats"][target]
    annual = intel["annual_df"]
    driver = intel["strongest_driver"]
    state = "significant" if stat["significant"] else "emerging" if stat["emerging"] else "unconfirmed"
    tiles = [('Mann–Kendall p', f'{stat["mk_p"]:.4f}'), ('Inflection year', intel['tipping_year']),
             ('Strongest driver r', f'{intel["driver_couplings"][driver]["r_annual"]:+.2f} · {driver}'),
             ('Years analysed', intel['n_years']), ('ESP32 bitmask', intel['led_bitmask'])]
    return (f'<section class="hero-card" data-verdict="{state}" aria-label="Current investigation"><div class="hero-top"><div class="hero-location">' + variable_icon(target) + esc(meta['region_name']) + '</div>'
            + status_chip(state.capitalize() + ' trend', state) + '</div><div class="context-stamp">' + esc('External CSV' if str(meta.get('router_source', '')).startswith('EXTERNAL') else 'NASA POWER') + ' · ' + esc(meta['start_year']) + '–' + esc(meta['end_year']) + ' · ' + esc(intel['n_years']) + ' annual means</div><div class="hero-zones"><div><div class="tile-label">' + esc(variables[target]['name']) + '</div>'
            f'<div class="hero-value">{stat["slope_dec"]:+.3f}</div><div class="hero-unit">{esc(variables[target]["unit"])} / decade · {esc(target)}</div>'
            '<p class="tile-label">Deseasonalized rate of change</p></div><div class="hero-sparkline"><div class="tile-label">Annual observed record</div>'
            + sparkline_svg(annual[target], 'Annual observed ' + target, variable_color(target)) + f'<div class="spark-caption"><span>{esc(annual.year.iloc[0])}</span><span>Straight segments · {esc(variables[target]["unit"])}</span><span>{esc(annual.year.iloc[-1])}</span></div></div>'
            '</div>'
            '<div class="hero-stats">' + ''.join(stat_tile(k, v) for k, v in tiles) + '</div></section>')


def led_tile(title, active, trigger, color):
    allowed = {"blue": "#60A5FA", "orange": "#FF9B42", "red": "#FF7A7A", "white": "#F5F7FF", "green": "#4ADE80"}
    ink = allowed.get(color, "#F5F7FF")
    return f'<div class="led-tile" style="--led-color:{ink}"><span class="led-dot {"on" if active else "off"}" aria-hidden="true"></span><div class="led-title">{esc(title)}</div><div class="led-state">{"ACTIVE [1]" if active else "OFF [0]"}</div><div class="led-trigger">{esc(trigger)}</div></div>'


def html_table(rows, caption="Driver lag analysis"):
    if not rows:
        return ''
    headers = list(rows[0])
    head = ''.join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = ''
    for row in rows:
        cells = ''.join('<td>' + (status_chip(row[h], 'unconfirmed') if row[h] == 'INVERSE PARADOX' else esc(row[h])) + '</td>' for h in headers)
        body += '<tr>' + cells + '</tr>'
    return f'<div class="table-wrap" role="region" aria-label="{esc(caption)}" tabindex="0"><table class="concord-table"><caption>{esc(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def footer():
    return '<div class="concord-footer"><span class="wordmark">' + icon() + 'CONCORD</span><span>NASA POWER / MERRA-2 / CERES</span><span>Exploratory statistics, not causal inference.</span></div>'


def info_note(description):
    """Focus and hover reveal the full original section explanation."""
    return '<span class="info-note" tabindex="0" role="button" aria-label="' + esc(description) + '"><span class="info-symbol" aria-hidden="true">i</span><span class="info-copy" role="tooltip">' + esc(description) + '</span></span>'


def with_info(markup, description):
    position = markup.rfind('</div>')
    return markup[:position] + info_note(description) + markup[position:]


VARIABLE_COLORS = {
    "T2M": "#FF8F5A", "GWETTOP": "#90B879", "PRECTOTCORR": "#79B9EC",
    "ALLSKY_SFC_SW_DWN": "#F0C66C", "QV2M": "#87CCD2", "WS2M": "#ADBFD4",
}


def variable_color(code):
    return VARIABLE_COLORS.get(str(code), "#C8CDD9")


def variable_icon(code):
    """Small original filled pictograms; color consistently identifies the variable."""
    uid = "variable" + uuid4().hex
    fill = f'url(#{uid})'
    shapes = {
        "T2M": f'<path d="M10 5a4 4 0 0 1 8 0v12a7 7 0 1 1-8 0Z" fill="{fill}"/><rect x="13" y="5" width="2" height="17" rx="1" fill="#FFF1DF"/><circle cx="14" cy="23" r="3" fill="#FFF1DF"/>',
        "GWETTOP": f'<path d="M16 2C13 7 7 12 7 17a9 9 0 0 0 18 0c0-5-6-10-9-15Z" fill="{fill}"/><path d="M3 25h8l4 3h14v3H3Z" fill="#A48662"/><path d="M11 16c-1 4 1 6 4 6" fill="none" stroke="#E6F3D1" stroke-width="2" stroke-linecap="round"/>',
        "PRECTOTCORR": f'<path d="M8 19a6 6 0 1 1 1-12 8 8 0 0 1 15 3 5 5 0 1 1 0 10H8Z" fill="{fill}"/><path d="m9 24-2 5m10-5-2 5m10-5-2 5" stroke="#79B9EC" stroke-width="3" stroke-linecap="round"/>',
        "ALLSKY_SFC_SW_DWN": f'<circle cx="16" cy="16" r="8" fill="{fill}"/><path d="M16 2v3m0 22v3M2 16h3m22 0h3M6 6l2 2m16 16 2 2M6 26l2-2M24 8l2-2" stroke="#F0C66C" stroke-width="3" stroke-linecap="round"/>',
        "QV2M": f'<path d="M12 3C9 8 4 12 4 17a8 8 0 0 0 16 0c0-5-5-9-8-14Zm13 9c-2 4-5 7-5 11a5 5 0 0 0 10 0c0-4-3-7-5-11Z" fill="{fill}"/>',
        "WS2M": f'<path d="M3 9h17a4 4 0 1 0-4-4h3a1 1 0 1 1 1 1H3Zm0 7h23a5 5 0 1 0-5-5h3a2 2 0 1 1 2 2H3Zm0 7h14a2 2 0 1 1-2 2h-3a5 5 0 1 0 5-5H3Z" fill="{fill}"/>',
    }
    shape = shapes.get(code, f'<path d="M4 15h5v14H4Zm9-8h5v22h-5Zm9-5h5v27h-5Z" fill="{fill}"/>')
    svg = f'<svg viewBox="0 0 32 32"><defs><linearGradient id="{uid}" x2=".7" y2="1"><stop stop-color="{variable_color(code)}"/><stop offset="1" stop-color="{variable_color(code)}" stop-opacity=".55"/></linearGradient></defs>{shape}</svg>'
    return inline_svg(svg, str(code), "variable-icon")


def status_icon(state):
    colors = {"significant": "#90B879", "emerging": "#F0C66C", "unconfirmed": "#ADB5CE"}
    color = colors.get(state, colors["unconfirmed"])
    marks = {
        "significant": '<path d="m6 12 4 4 8-9" fill="none" stroke="#152619" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>',
        "emerging": '<path d="m7 14 5-6 5 6h-4v5h-2v-5Z" fill="#35280D"/>',
        "unconfirmed": '<path d="M9 8a3 3 0 0 1 6 1c0 2-3 2-3 5m0 3v1" fill="none" stroke="#1C2340" stroke-width="2" stroke-linecap="round"/>',
    }
    return inline_svg(f'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="{color}"/>{marks.get(state, marks["unconfirmed"])}</svg>', state, "status-icon")


@st.cache_data(ttl=300, show_spinner=False)
def _tile_available(url):
    """Check the actual raster provider without touching scientific acquisition."""
    try:
        response = requests.get(url, timeout=(2, 3))
        return response.status_code == 200 and response.content.startswith(b'\x89PNG')
    except requests.RequestException:
        return False


def outline_map(meta, height=330):
    fig = go.Figure(go.Scattergeo(lat=[meta['lat']], lon=[meta['lon']],
                                 text=[esc(meta['region_name'])], marker=dict(size=10, color='#FF8F5A')))
    fig.update_geos(showland=True, landcolor='#333C57', showocean=True, oceancolor='#18203F',
                    bgcolor='rgba(0,0,0,0)', coastlinecolor='#69718A', showcountries=True,
                    countrycolor='#525B76', projection_type='natural earth')
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Segoe UI, system-ui, sans-serif', color='#F5F7FF'),
                      margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


def basemap_figure(meta, height=330):
    """Token-free Esri dark-gray tiles in Scattermap; outline on failure."""
    lat, lon = float(meta['lat']), float(meta['lon'])
    zoom = 5
    map_lat = max(-85.05, min(85.05, lat))
    x = floor((lon + 180) / 360 * 2**zoom) % 2**zoom
    y = floor((1 - asinh(tan(map_lat * pi / 180)) / pi) / 2 * 2**zoom)
    endpoint = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile'
    if not _tile_available(f'{endpoint}/{zoom}/{y}/{x}'):
        return outline_map(meta, height), False
    # A raster style avoids token, glyph, and sprite dependencies. Attribution
    # stays visible in MapLibre and is repeated as an accessible HTML link.
    style = {'version': 8, 'sources': {'basemap': {'type': 'raster', 'tileSize': 256, 'maxzoom': 16,
             'tiles': [endpoint + '/{z}/{y}/{x}'],
             'attribution': 'Esri, HERE, Garmin, © OpenStreetMap contributors, GIS user community'}},
             'layers': [{'id': 'basemap', 'type': 'raster', 'source': 'basemap'}]}
    cls = go.Scattermap if hasattr(go, 'Scattermap') else go.Scattermapbox
    fig = go.Figure(cls(lat=[lat], lon=[lon], mode='markers',
                       marker=dict(size=13, color='#FF8F5A'),
                       text=[esc(meta['region_name'])], hovertemplate='%{text}<br>%{lat:.2f}, %{lon:.2f}<extra></extra>'))
    map_key = 'map' if hasattr(go, 'Scattermap') else 'mapbox'
    fig.update_layout(**{map_key: dict(style=style, center=dict(lat=map_lat, lon=lon), zoom=zoom)},
                      height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Segoe UI, system-ui, sans-serif', color='#F5F7FF'))
    return fig, True


def render_map(meta, height=330, context='telemetry'):
    fig, tiles_available = basemap_figure(meta, height)
    st.plotly_chart(fig, width='stretch', theme=None, key=f'{context}-map', config={'displayModeBar': False})
    if tiles_available:
        st.html('<div class="map-source"><a href="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer">Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community</a></div>')
        with st.expander('Map not loading? Show outline map', on_change='rerun') as fallback:
            if fallback.open:
                st.plotly_chart(outline_map(meta, height), width='stretch', theme=None,
                                key=f'{context}-outline', config={'displayModeBar': False})
    else:
        st.caption('Basemap tiles unavailable · showing the outline map.')


def render_hero(meta, intel, variables):
    with st.container(key='current-result'):
        result, location = st.columns([2, 1], gap='large')
        with result:
            st.html(hero_card(meta, intel, variables))
        with location:
            st.caption('Sampling grid cell')
            render_map(meta, height=280, context='hero')


def label_chart(fig, unit_label):
    """Direct labels are display geometry only; observed coordinates stay intact."""
    import textwrap
    if unit_label:
        fig.add_annotation(x=0, y=1.03, xref='paper', yref='paper', xanchor='left', yanchor='bottom',
                           text='<br>'.join(esc(line) for line in textwrap.wrap(unit_label, 34)),
                           showarrow=False, font=dict(size=12, color='#B9BFD3'), align='left')
    endpoints = []
    for trace in fig.data:
        if trace.type != 'scatter' or trace.fill == 'toself' or 'lines' not in (trace.mode or ''):
            continue
        if trace.y is None or trace.x is None:
            continue
        finite = [(x, float(y)) for x, y in zip(trace.x, trace.y) if y is not None and isfinite(float(y))]
        if not finite:
            continue
        name = str(trace.name or '')
        if 'OLS' in name:
            label = 'OLS'
        elif 'Forecast' in name:
            label = 'Forecast'
        elif 'Pre-' in name:
            label = 'Before split'
        elif 'Post-' in name:
            label = 'After split'
        elif 'Raw Monthly' in name:
            label = 'Raw'
        elif 'Deseasonalized' in name:
            label = 'Anomaly'
        elif 'Observed' in name or 'Historical' in name:
            label = 'Observed'
        else:
            label = next((short for code, short in [('ALLSKY_SFC_SW_DWN', 'Solar'), ('PRECTOTCORR', 'Rain'), ('GWETTOP', 'Soil'), ('T2M', 'T2M'), ('QV2M', 'Humidity'), ('WS2M', 'Wind')] if code in name), 'Series')
        endpoints.append((finite[-1], label, trace.line.color or '#F5F7FF'))
    # Offset labels vertically where adjacent endpoints would collide.
    if endpoints:
        values = [float(y) for trace in fig.data if trace.type == 'scatter' and trace.y is not None for y in trace.y if y is not None and isfinite(float(y))]
        span = max(values) - min(values) or 1
        last_position = -float('inf')
        plot_height = max(150, (fig.layout.height or 430) - 110)
        for (x, y), label, color in sorted(endpoints, key=lambda item: item[0][1]):
            position = (y - min(values)) / span * plot_height
            adjusted = max(position, last_position + 18)
            shift = adjusted - position
            last_position = adjusted
            fig.add_annotation(x=x, y=y, text=esc(label), showarrow=False, xanchor='left',
                               xshift=8, yshift=shift, font=dict(size=11, color=color),
                               bgcolor='rgba(0,0,0,0)')


def chart_key(fig):
    """Retain every original legend name and statistic in an accessible key."""
    rows = []
    for trace in fig.data:
        if not trace.name:
            continue
        color = getattr(getattr(trace, 'line', None), 'color', None) or '#B9BFD3'
        if color == 'rgba(255,255,255,0)':
            color = '#FF8F5A'
        rows.append('<span class="series-item"><span class="series-swatch" style="background:' + esc(color) + '"></span>' + esc(trace.name) + '</span>')
    return '<details class="series-key"><summary>Series</summary><div>' + ''.join(rows) + '</div></details>'
