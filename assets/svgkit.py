# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Tiny helper for the animated README figures: SVG with CSS animation and subset, embedded fonts.

GitHub shows README images without JavaScript, so motion is CSS only; under "reduce motion"
every animation is off and the figure shows its final frame.
"""
import base64
import io
from pathlib import Path
from xml.sax.saxutils import escape

from fontTools import subset
from fontTools.ttLib import TTFont

FONTS = Path(__file__).resolve().parent / "fonts"


def font_uri(file, chars):
    font = TTFont(FONTS / file)
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["*"]          # keep Arabic joining and ligatures
    opts.name_IDs = []
    s = subset.Subsetter(opts)
    s.populate(text="".join(sorted(set(chars) | {" "})))
    s.subset(font)
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    return "data:font/woff2;base64," + base64.b64encode(buf.getvalue()).decode()


class Figure:
    def __init__(self, w, h, theme, label, faces):
        self.w, self.h, self.t, self.label, self.faces = w, h, theme, label, faces
        self.parts, self.css = [], []
        self.chars = {k: set() for k in faces}

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, face, size, fill="ink", anchor="start", cls="", style="", spacing=0, extra=""):
        self.chars[face].update(s)
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        k = f' class="{cls}"' if cls else ""
        st = f' style="{style}"' if style else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{face}" font-size="{size}" fill="{self.t.get(fill, fill)}" '
                 f'text-anchor="{anchor}"{ls}{k}{st}{extra}>{escape(s)}</text>')

    def render(self):
        faces = "".join(f'@font-face{{font-family:{k};src:url({font_uri(f, self.chars[k])}) format("woff2");}}'
                        for k, f in self.faces.items() if self.chars[k])
        base = (".fade{animation:fade .8s ease-out both}@keyframes fade{from{opacity:0}to{opacity:1}}"
                ".rise{animation:rise .9s cubic-bezier(.2,.7,.2,1) both}"
                "@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}"
                "@media (prefers-reduced-motion: reduce){*{animation:none!important}}")
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'role="img" aria-label="{escape(self.label)}"><title>{escape(self.label)}</title>'
                f'<style>{faces}{base}{"".join(self.css)}</style>{"".join(self.parts)}</svg>')


def d(s):
    return f"animation-delay:{s}s"
