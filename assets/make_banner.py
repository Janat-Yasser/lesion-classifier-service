# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Draws the animated banner at the top of the README (light and dark).

    python assets/make_banner.py   ->  assets/banner-light.svg, assets/banner-dark.svg

The request and response are a real call to this service with the shipped model; the numbers on the
right come from model/model_card.json and the load test in the README.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svgkit import Figure, d  # noqa: E402

HERE = Path(__file__).resolve().parent
FACES = {"Mo": "ibm-plex-mono-latin-400-normal.woff2", "Mo6": "ibm-plex-mono-latin-600-normal.woff2",
         "Sg": "space-grotesk-latin-600-normal.woff2", "SgR": "space-grotesk-latin-400-normal.woff2"}
TERM = dict(term="#0b1020", tbar="#151b2e", tx="#c9d1e3", dim="#6b7590", key="#7dd3fc", str="#a7f3d0",
            num="#fcd34d", ok="#4ade80", prompt="#f472b6")
THEMES = {
    "light": dict(bg="#f3f5f9", ink="#0f172a", muted="#475569", faint="#94a3b8", line="#dde3ec", tile="#ffffff", acc="#2563eb", **TERM),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9aa7b8", faint="#5f6b7c", line="#232b36", tile="#131a24", acc="#60a5fa", **TERM),
}

# (indent, [(text, colour key)]) -- one line of the terminal each
LINES = [
    (0, [("$ ", "prompt"), ("curl -F file=@lesion.png localhost:8000/v1/predict", "tx")]),
    (0, [("{", "dim")]),
    (1, [('"label"', "key"), (": ", "dim"), ('"NV"', "str"), (",  ", "dim"), ('"name"', "key"), (": ", "dim"),
         ('"Melanocytic nevus"', "str"), (",", "dim")]),
    (1, [('"probability"', "key"), (": ", "dim"), ("0.7527", "num"), (",", "dim")]),
    (1, [('"melanoma"', "key"), (": {", "dim"), ('"probability"', "key"), (": ", "dim"), ("0.0027", "num"), (", ", "dim"),
         ('"threshold"', "key"), (": ", "dim"), ("0.2646", "num"), (", ", "dim"), ('"flag"', "key"), (": ", "dim"),
         ("false", "num"), ("},", "dim")]),
    (1, [('"model_version"', "key"), (": ", "dim"), ('"ham10000-effnetb0-lesion-20261108"', "str"), (",", "dim")]),
    (1, [('"inference_ms"', "key"), (": ", "dim"), ("20.4", "num")]),
    (0, [("}", "dim")]),
    (0, [("$ ", "prompt"), ("curl localhost:8000/health", "tx")]),
    (0, [("{", "dim"), ('"status"', "key"), (": ", "dim"), ('"ok"', "str"), (", ", "dim"), ('"model_loaded"', "key"),
         (": ", "dim"), ("true", "num"), ("}", "dim")]),
]
CW = 6.62   # IBM Plex Mono advance at 11px: 0.6 em


def banner(t, card):
    W, H = 880, 340
    m = card["test_metrics"]
    f = Figure(W, H, t, "lesion-classifier-service: a terminal sends a skin-lesion photo to the API and gets back a "
               "JSON prediction with calibrated probabilities and a melanoma flag. Median latency 17 ms on 2 CPUs, "
               "about 58 requests per second, 21 tests, drift monitored.", FACES)
    f.add(f'<rect x="0.75" y="0.75" width="{W-1.5}" height="{H-1.5}" rx="14" fill="{t["bg"]}" stroke="{t["line"]}" stroke-width="1.5"/>')
    # left column
    f.text(32, 50, "SERVICE · v1", "Mo6", 10.5, "acc", spacing=1.5, cls="fade")
    f.text(32, 92, "lesion-classifier", "Sg", 30, "ink", cls="rise", style=d(0.1))
    f.text(32, 126, "-service", "Sg", 30, "ink", cls="rise", style=d(0.15))
    f.text(32, 156, "My HAM10000 model behind a real API.", "SgR", 13.5, "muted", cls="fade", style=d(0.3))
    f.text(32, 175, "ONNX Runtime · FastAPI · Docker", "SgR", 13.5, "muted", cls="fade", style=d(0.35))
    tiles = [("p50 latency", "17 ms", "2 CPUs, one image"), ("throughput", "58 /s", "no errors at 4x"),
             ("tests", "21", "+ CI load test"), ("melanoma AUC", f'{m["melanoma_auc"]:.3f}', "official test set")]
    for i, (k, v, sub) in enumerate(tiles):
        x, y = 32 + (i % 2) * 132, 200 + (i // 2) * 64
        f.add(f'<rect x="{x}" y="{y}" width="122" height="56" rx="8" fill="{t["tile"]}" stroke="{t["line"]}" '
              f'class="rise" style="{d(0.5 + i * 0.12)}"/>')
        f.text(x + 10, y + 17, k.upper(), "Mo", 8.5, "faint", spacing=0.8, cls="rise", style=d(0.5 + i * 0.12))
        f.text(x + 10, y + 38, v, "Sg", 18, "ink", cls="rise", style=d(0.5 + i * 0.12))
        f.text(x + 10, y + 50, sub, "SgR", 8.5, "muted", cls="rise", style=d(0.5 + i * 0.12))

    # terminal
    tx, ty, tw, th = 312, 28, 538, 284
    f.add(f'<rect x="{tx}" y="{ty}" width="{tw}" height="{th}" rx="10" fill="{t["term"]}"/>'
          f'<path d="M{tx} {ty+10} a10 10 0 0 1 10 -10 h{tw-20} a10 10 0 0 1 10 10 v18 h-{tw} z" fill="{t["tbar"]}"/>')
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        f.add(f'<circle cx="{tx + 16 + i * 16}" cy="{ty + 14}" r="4.5" fill="{c}"/>')
    f.text(tx + tw / 2, ty + 18, "janat@lab: ~/lesion-classifier-service", "Mo", 10, "dim", "middle")
    f.css.append(".type{clip-path:inset(0 100% 0 0);animation:type var(--t) steps(var(--n), jump-start) both}"
                 "@keyframes type{to{clip-path:inset(0 0 0 0)}}"
                 ".cur{animation:blink 1s steps(1) infinite}@keyframes blink{50%{opacity:0}}"
                 "@media (prefers-reduced-motion: reduce){.type{clip-path:none}}")
    y, at = ty + 52, 0.6
    for n, (indent, segs) in enumerate(LINES):
        x = tx + 18 + indent * 2 * CW
        text = "".join(s for s, _ in segs)
        typed = segs[0][1] == "prompt"
        dur = len(text) * 0.035 if typed else 0.18
        tsp = "".join(f'<tspan fill="{t[c]}">{s.replace("&", "&amp;").replace("<", "&lt;")}</tspan>' for s, c in segs)
        for s, _ in segs:
            f.chars["Mo"].update(s)
        f.add(f'<text x="{x}" y="{y}" font-family="Mo" font-size="11" class="type" xml:space="preserve" '
              f'style="--t:{dur:.2f}s;--n:{max(len(text), 1)};{d(at)}">{tsp}</text>')
        at += dur + (0.5 if typed else 0.05)
        y += 21 if n in (0, 7) else 19
    f.add(f'<rect x="{tx + 18}" y="{y - 10}" width="7" height="13" fill="{t["tx"]}" class="cur"/>')
    f.chars["Mo"].update("$ ")
    return f


def main():
    card = json.loads((HERE.parent / "model" / "model_card.json").read_text())
    for name, t in THEMES.items():
        (HERE / f"banner-{name}.svg").write_text(banner(t, card).render(), encoding="utf-8")
    print("banner written")


if __name__ == "__main__":
    main()
