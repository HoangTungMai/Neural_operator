#!/usr/bin/env python3
"""Render the project summary report (docs/bao_cao_tong_ket_phase3-6.md) to PDF.

A small generic Markdown->PDF renderer (fpdf2) so the Markdown stays the single
source of truth: headings, paragraphs, bullets, GFM tables, fenced code, and
blockquotes. Reuses the visual style of make_pdf.py.

  python -m novbts.report.make_summary_pdf
"""
import re
from pathlib import Path

from fpdf import FPDF

from novbts.paths import DOCS, ensure

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
INK = (20, 20, 20)
MUT = (90, 90, 90)
ACC = (30, 70, 140)

SRC = DOCS / "bao_cao_tong_ket_phase3-6.md"
OUT = DOCS / "bao_cao_tong_ket_phase3-6.pdf"
TITLE = "Tổng kết dự án — Neural-Operator surrogate cho cảm biến VBTS"


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("D", "", 7); self.set_text_color(*MUT)
        self.cell(0, 6, TITLE, align="L")
        self.cell(0, 6, f"tr. {self.page_no()}", align="R"); self.ln(8)
        self.set_text_color(*INK)


def setup():
    pdf = PDF(format="A4")
    pdf.add_font("D", "", FONT); pdf.add_font("D", "B", FONTB); pdf.add_font("M", "", FONTM)
    # No oblique TTF on this system: map italic/bold-italic to regular/bold so
    # markdown *italics* render (upright) instead of raising "Undefined font: dI".
    pdf.add_font("D", "I", FONT); pdf.add_font("D", "BI", FONTB)
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(18, 16, 18)
    return pdf


def _clean(t):
    """Strip inline code backticks + markdown links -> text (fpdf markdown keeps **bold**)."""
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    return t


def _mc(pdf, h, t, md=True):
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, h, _clean(t), new_x="LMARGIN", new_y="NEXT", markdown=md)


def title(pdf, t):
    pdf.set_font("D", "B", 17); pdf.set_text_color(*ACC)
    _mc(pdf, 9, t, md=False); pdf.set_text_color(*INK); pdf.ln(2)


def h1(pdf, t):
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()
    pdf.ln(2); pdf.set_font("D", "B", 13.5); pdf.set_text_color(*ACC)
    _mc(pdf, 7, t, md=False); pdf.set_text_color(*INK); pdf.ln(1)


def h2(pdf, t):
    pdf.ln(1); pdf.set_font("D", "B", 10.5); pdf.set_text_color(*INK)
    _mc(pdf, 6, t, md=False); pdf.ln(0.5)


def body(pdf, t):
    pdf.set_font("D", "", 9.5); pdf.set_text_color(*INK)
    _mc(pdf, 5, t); pdf.ln(0.5)


def quote(pdf, t):
    pdf.set_font("D", "", 9); pdf.set_text_color(*MUT)
    pdf.set_x(pdf.l_margin + 3)
    pdf.multi_cell(0, 5, _clean(t), new_x="LMARGIN", new_y="NEXT", markdown=True)
    pdf.set_text_color(*INK); pdf.ln(0.5)


def bullet(pdf, t, indent=0):
    pdf.set_font("D", "", 9.5); pdf.set_text_color(*INK)
    pdf.set_x(pdf.l_margin + indent)
    pdf.cell(4, 5, "–" if indent else "•")
    pdf.multi_cell(0, 5, _clean(t), new_x="LMARGIN", new_y="NEXT", markdown=True)


def code(pdf, lines):
    pdf.set_font("M", "", 7.8); pdf.set_fill_color(244, 244, 244); pdf.set_text_color(*INK)
    for line in lines:
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 4.4, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def table(pdf, headers, rows):
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    ncol = len(headers)
    # column weights from longest cell text
    maxlen = [len(_clean(headers[c])) for c in range(ncol)]
    for r in rows:
        for c in range(ncol):
            maxlen[c] = max(maxlen[c], len(_clean(r[c])) if c < len(r) else 0)
    tot = sum(maxlen) or 1
    widths = [max(usable * 0.10, usable * m / tot) for m in maxlen]
    sc = usable / sum(widths)
    widths = [w * sc for w in widths]

    if pdf.get_y() > pdf.h - 30:
        pdf.add_page()
    pdf.set_font("D", "B", 8.3); pdf.set_fill_color(*ACC); pdf.set_text_color(255, 255, 255)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, _clean(h), border=0, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(*INK)
    for i, row in enumerate(rows):
        fill = i % 2 == 1
        pdf.set_fill_color(238, 242, 248)
        pdf.set_font("D", "", 8.1)
        cells = [_clean(row[c]) if c < len(row) else "" for c in range(ncol)]
        heights = []
        for c, w in zip(cells, widths):
            heights.append(len(pdf.multi_cell(w, 4.4, c, dry_run=True, output="LINES", markdown=True)))
        rh = max(heights) * 4.4
        if pdf.get_y() + rh > pdf.h - pdf.b_margin:
            pdf.add_page()
        y0 = pdf.get_y(); x0 = pdf.l_margin
        for c, w in zip(cells, widths):
            x, y = pdf.get_x(), pdf.get_y()
            pdf.multi_cell(w, rh, "", border=0, fill=fill, new_x="RIGHT", new_y="TOP")
            pdf.set_xy(x, y)
            pdf.multi_cell(w, 4.4, c, border=0, align="L", new_x="RIGHT", new_y="TOP", markdown=True)
            pdf.set_xy(x + w, y)
        pdf.set_xy(x0, y0 + rh)
    pdf.ln(2)


def split_row(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def render(pdf, md):
    lines = md.split("\n")
    i = 0
    in_code = False
    code_buf = []
    while i < len(lines):
        ln = lines[i]
        # fenced code
        if ln.strip().startswith("```"):
            if in_code:
                code(pdf, code_buf); code_buf = []; in_code = False
            else:
                in_code = True
            i += 1; continue
        if in_code:
            code_buf.append(ln); i += 1; continue

        s = ln.strip()
        if not s or s == "---":
            i += 1; continue

        # GFM table: header row then a |---| separator
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            headers = split_row(s)
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i])); i += 1
            table(pdf, headers, rows); continue

        if s.startswith("# "):
            title(pdf, s[2:].strip())
        elif s.startswith("## "):
            h1(pdf, s[3:].strip())
        elif s.startswith("### "):
            h2(pdf, s[4:].strip())
        elif s.startswith("#### "):
            h2(pdf, s[5:].strip())
        elif s.startswith("> "):
            quote(pdf, s[2:].strip())
        elif re.match(r"^[-*] ", s):
            bullet(pdf, s[2:].strip())
        elif re.match(r"^\d+\. ", s):
            bullet(pdf, re.sub(r"^\d+\.\s*", "", s))
        elif ln.startswith("  - ") or ln.startswith("    - "):
            bullet(pdf, s[2:].strip(), indent=4)
        elif s.startswith("*") and s.endswith("*") and len(s) > 2:
            quote(pdf, s.strip("*"))
        else:
            body(pdf, s)
        i += 1
    if code_buf:
        code(pdf, code_buf)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Render a project Markdown report to PDF.")
    ap.add_argument("--src", default=str(SRC), help="input Markdown path")
    ap.add_argument("--out", default=None, help="output PDF path (default: src with .pdf)")
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out) if a.out else src.with_suffix(".pdf")
    md = src.read_text(encoding="utf-8")
    pdf = setup(); pdf.add_page()
    render(pdf, md)
    ensure(out.parent)
    pdf.output(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
