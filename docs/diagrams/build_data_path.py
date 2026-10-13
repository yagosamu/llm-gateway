"""Writes docs/diagrams/data-path.html and data-path-dark.html, the README's data path, drawn with the
diagram-design skill (architecture type, a faithful redraw of the former Mermaid block, doc-wide 1280x720).

Usage: uv run python docs/diagrams/build_data_path.py docs/diagrams
The PNGs are screenshots of the <svg> alone at 2x (2560x1440), taken with headless Chrome."""
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
FONTS = ("https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600"
         "&family=Geist+Mono:wght@400;500;600&display=swap")
SANS, MONO = "'Geist', system-ui, sans-serif", "'Geist Mono', ui-monospace, monospace"

SKINS = {
    "light": dict(paper="#f5f5f5", ink="#2d3142", ink_rgb="45,49,66", muted="#4f5d75", muted_rgb="79,93,117",
                  soft="#5a6580", soft_rgb="90,101,128", accent="#bf4520", accent_rgb="191,69,32",
                  accent_tint="rgba(191,69,32,0.08)", link="#2e5aa8", backend_fill="#ffffff"),
    "dark": dict(paper="#2d3142", ink="#f5f5f5", ink_rgb="245,245,245", muted="#bfc0c0", muted_rgb="191,192,192",
                 soft="#949eb2", soft_rgb="148,158,178", accent="#f08a59", accent_rgb="240,138,89",
                 accent_tint="rgba(240,138,89,0.10)", link="#739fdf", backend_fill="#2d3142"),
}

# id: (x, y, w, h, treatment, tag, name, sublabels)
NODES = {
    "T": (56, 96, 160, 64, "input", "IN", "240 Dolly prompts", ["data/traffic.jsonl"]),
    "R": (264, 96, 160, 64, "backend", "JOB", "Recorder", ["harness.record"]),
    "X": (264, 208, 160, 64, "focal", "FILE", "Recordings", ["recordings/matrix.jsonl"]),
    "J": (56, 208, 160, 64, "backend", "LLM", "Two judges", ["harness.routing.judge"]),
    "VD": (56, 304, 160, 64, "store", "FILE", "Verdicts", ["judgments/verdicts.jsonl"]),
    "EV": (56, 456, 160, 64, "backend", "JOB", "Evaluation", ["results/slice2.json"]),
    "RP": (264, 456, 160, 64, "backend", "SIM", "ReplayProvider", ["harness.replay"]),
    "G": (616, 280, 160, 144, "focal", "API", "Gateway", ["/v1/chat/completions"]),
    "RT": (880, 96, 160, 64, "backend", "YAML", "Router", ["routing.yaml"]),
    "AD": (880, 192, 160, 64, "backend", "SDK", "Adapters", ["OpenAI · Anthropic · Groq"]),
    "P": (1072, 192, 160, 64, "external", "EXT", "Providers", []),
    "L": (880, 312, 160, 64, "store", "DB", "Request log", ["+ verifications · SQLite"]),
    "V": (1072, 312, 160, 64, "backend", "LLM", "Verifier", ["high tier + judge"]),
    "M": (880, 400, 160, 64, "external", "OBS", "Metrics", ["Prometheus · Grafana"]),
    "RD": (880, 496, 160, 64, "store", "KV", "Redis", ["health · circuits", "job queue · cache"]),
    "WK": (1072, 496, 160, 64, "backend", "JOB", "Worker", ["deferred jobs"]),
}

# (path, style) with style in default | link | dashed | bidir
EDGES = [
    ("M 216,128 H 264", "default"),                                                   # T -> R
    ("M 424,128 H 496 Q 504,128 504,136 V 304 Q 504,312 512,312 H 616", "default"),  # R -> G
    ("M 344,160 V 208", "default"),                                                   # R -> X
    ("M 264,240 H 216", "default"),                                                   # X -> J
    ("M 136,272 V 304", "default"),                                                   # J -> VD
    ("M 136,368 V 456", "default"),                                                   # VD -> EV
    ("M 344,272 V 456", "default"),                                                   # X -> RP
    ("M 424,488 H 528 Q 536,488 536,480 V 408 Q 536,400 544,400 H 616", "default"),  # RP -> G
    ("M 696,280 V 136 Q 696,128 704,128 H 880", "default"),                           # G -> RT
    ("M 744,280 V 232 Q 744,224 752,224 H 880", "default"),                           # G -> AD
    ("M 1040,224 H 1072", "link"),                                                    # AD -> P
    ("M 776,288 H 1144 Q 1152,288 1152,296 V 312", "dashed"),                         # G -> V, sampled
    ("M 776,344 H 880", "default"),                                                   # G -> L
    ("M 1072,344 H 1040", "default"),                                                 # V -> L
    ("M 776,412 H 880", "default"),                                                   # G -> M
    ("M 736,424 V 520 Q 736,528 744,528 H 880", "bidir"),                              # G <-> Redis
    ("M 1040,528 H 1072", "default"),                                                 # Redis -> Worker
    ("M 1152,560 V 584 Q 1152,592 1144,592 H 704 Q 696,592 696,584 V 424", "default"),  # Worker -> G
]
# (center x, connector y, text): a mask 8px above a horizontal segment
LABELS = [(564, 312, "WITH 3 HEADERS"), (828, 288, "SAMPLED, LATER")]
ZONES = [(32, 64, 520, 328, "RECORD ONCE, LIVE"), (32, 424, 520, 120, "OFFLINE, IN CI"),
         (576, 64, 672, 552, "SERVED PATH")]


def treatment(kind: str, s: dict) -> tuple[str, str, str]:
    """(fill, stroke, tag color)"""
    ink, muted = s["ink_rgb"], s["muted_rgb"]
    return {"focal": (s["accent_tint"], s["accent"], s["accent"]),
            "backend": (s["backend_fill"], s["ink"], s["ink"]),
            "store": (f"rgba({ink},0.05)", s["muted"], s["muted"]),
            "external": (f"rgba({ink},0.03)", f"rgba({ink},0.30)", s["soft"]),
            "input": (f"rgba({muted},0.10)", s["soft"], s["soft"])}[kind]


def svg(skin: str, slug: str) -> str:
    s = SKINS[skin]
    out = [f'<svg viewBox="0 0 1280 720" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-labelledby="{slug}-title {slug}-desc">',
           f'<title id="{slug}-title">llm-gateway data path</title>',
           f'<desc id="{slug}-desc">Dolly prompts are recorded once through the gateway and judged once; the gateway routes, '
           f'calls providers, logs, verifies a sample and shares health, circuits, jobs and cached answers in Redis, '
           f'while CI replays the recordings offline to recompute the results.</desc>',
           "<defs>",
           f'<marker id="{slug}-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto-start-reverse">'
           f'<polygon points="0 0, 8 3, 0 6" fill="{s["muted"]}"/></marker>',
           f'<marker id="{slug}-arrow-link" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
           f'<polygon points="0 0, 8 3, 0 6" fill="{s["link"]}"/></marker>',
           "</defs>",
           f'<rect width="100%" height="100%" fill="{s["paper"]}"/>']
    for x, y, w, h, label in ZONES:  # zones first
        lw = round(len(label) * 7 * 0.62 + 16) // 4 * 4 + 4
        out += [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="rgba({s["ink_rgb"]},0.02)" '
                f'stroke="rgba({s["ink_rgb"]},0.10)" stroke-width="0.8"/>',
                f'<rect x="{x + 16}" y="{y + 4}" width="{lw}" height="12" rx="2" fill="{s["paper"]}"/>',
                f'<text x="{x + 16 + lw / 2}" y="{y + 13}" fill="{s["soft"]}" font-size="7" font-family="{MONO}" '
                f'text-anchor="middle" letter-spacing="0.14em">{label}</text>']
    for path, style in EDGES:  # arrows before nodes
        stroke = s["link"] if style == "link" else s["muted"]
        marker = f"{slug}-arrow-link" if style == "link" else f"{slug}-arrow"
        extra = ' stroke-dasharray="4,3"' if style == "dashed" else ""
        start = f' marker-start="url(#{marker})"' if style == "bidir" else ""
        width = "1" if style == "dashed" else "1.2"
        out.append(f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="{width}"{extra}{start} '
                   f'marker-end="url(#{marker})"/>')
    for cx, y, text in LABELS:
        w = round(len(text) * 8 * 0.62 + 12) // 4 * 4 + 4
        out += [f'<rect x="{cx - w / 2}" y="{y - 20}" width="{w}" height="12" rx="2" fill="{s["paper"]}"/>',
                f'<text x="{cx}" y="{y - 11}" fill="{s["soft"]}" font-size="8" font-family="{MONO}" '
                f'text-anchor="middle" letter-spacing="0.06em">{text}</text>']
    for x, y, w, h, kind, tag, name, subs in NODES.values():
        fill, stroke, tag_color = treatment(kind, s)
        tw = round(len(tag) * 7 * 0.62 + 8) // 4 * 4 + 4
        cx, top = x + w / 2, y + h / 2 - (6 * len(subs)) + 4
        out += [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{s["paper"]}"/>',
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
                f'<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="transparent" stroke="{tag_color}" '
                f'stroke-opacity="0.45" stroke-width="0.8"/>',
                f'<text x="{x + 8 + tw / 2}" y="{y + 17}" fill="{tag_color}" font-size="7" font-family="{MONO}" '
                f'text-anchor="middle" letter-spacing="0.08em">{tag}</text>',
                f'<text x="{cx}" y="{top + 8}" fill="{s["ink"]}" font-size="12" font-weight="600" font-family="{SANS}" '
                f'text-anchor="middle">{name}</text>']
        for i, sub in enumerate(subs):
            out.append(f'<text x="{cx}" y="{top + 23 + 12 * i}" fill="{s["muted"]}" font-size="9" font-family="{MONO}" '
                       f'text-anchor="middle">{sub}</text>')
    # Legend strip
    out += [f'<line x1="32" y1="640" x2="1248" y2="640" stroke="rgba({s["ink_rgb"]},0.10)" stroke-width="0.8"/>',
            f'<text x="32" y="660" fill="{s["muted"]}" font-size="8" font-family="{MONO}" letter-spacing="0.18em">LEGEND</text>']
    x = 112
    for kind, label in (("focal", "Focal"), ("backend", "Job / service"), ("store", "Store / file"),
                        ("input", "Input"), ("external", "External")):
        fill, stroke, _ = treatment(kind, s)
        out += [f'<rect x="{x}" y="652" width="14" height="10" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
                f'<text x="{x + 20}" y="660" fill="{s["muted"]}" font-size="9" font-family="{SANS}">{label}</text>']
        x += 104
    for style, label, w in (("default", "Call or data", 112), ("link", "HTTP to a provider", 148),
                            ("dashed", "Sampled, after the response", 196), ("bidir", "Read and write", 120)):
        stroke = s["link"] if style == "link" else s["muted"]
        marker = f"{slug}-arrow-link" if style == "link" else f"{slug}-arrow"
        extra = ' stroke-dasharray="4,3"' if style == "dashed" else ""
        start = f' marker-start="url(#{marker})"' if style == "bidir" else ""
        out += [f'<line x1="{x + 4 if style == "bidir" else x}" y1="657" x2="{x + 28}" y2="657" stroke="{stroke}" '
                f'stroke-width="1.2"{extra}{start} marker-end="url(#{marker})"/>',
                f'<text x="{x + 36}" y="660" fill="{s["muted"]}" font-size="9" font-family="{SANS}">{label}</text>']
        x += w
    out.append("</svg>")
    return "\n".join(out)


def page(skin: str, slug: str) -> str:
    s = SKINS[skin]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llm-gateway data path</title>
<link href="{FONTS}" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: {SANS}; background: {s["paper"]}; color: {s["ink"]}; padding: 3rem 2rem; }}
  .frame {{ max-width: 1280px; margin: 0 auto; }}
  .diagram-container {{ width: 100%; overflow-x: auto; }}
  .eyebrow {{ font-family: {MONO}; font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em;
             text-transform: uppercase; color: {s["muted"]}; margin-bottom: 0.5rem; }}
  h1 {{ font-family: 'Instrument Serif', serif; font-size: 2rem; font-weight: 400; letter-spacing: -0.02em;
       margin-bottom: 1.5rem; }}
  svg {{ width: 100%; min-width: 1280px; display: block; }}
  @media print {{ .diagram-container {{ overflow-x: visible; }} svg {{ min-width: 0; }} }}
</style>
</head>
<body>
<div class="frame">
<p class="eyebrow">Architecture · llm-gateway</p>
<h1>Data path</h1>
<div class="diagram-container">
{svg(skin, slug)}
</div>
</div>
</body>
</html>
"""


OUT.mkdir(parents=True, exist_ok=True)
for skin, slug in (("light", "data-path"), ("dark", "data-path-dark")):
    (OUT / f"{slug}.html").write_text(page(skin, slug), encoding="utf-8", newline="\n")
    print(OUT / f"{slug}.html")
