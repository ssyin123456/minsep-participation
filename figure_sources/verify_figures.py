# -*- coding: utf-8 -*-
"""Verify the eight submission figures.

The released figures are 600 dpi JPEGs whose physical size is 6.0 x 3.2 in
(3600 x 1920 px).  pdflatex embeds them 1:1 at width=\textwidth, so the 9.5 pt
type set in figstyle.py keeps its physical size.  Text-size compliance and
legend placement are asserted by redraw_all.py / figstyle.audit at generation
time; this script checks the released rasters themselves.
"""
from pathlib import Path
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
W, H, DPI = 3600, 1920, 600
ok = True
for i in range(1, 9):
    p = ROOT / "figs" / f"fig{i}.jpg"
    if not p.exists():
        print(f"  fig{i}.jpg MISSING")
        ok = False
        continue
    b = p.read_bytes()
    jpeg = b[:3] == b"\xff\xd8\xff"
    try:
        with Image.open(p) as im:
            im.load()
            w, h = im.size
            fmt = im.format
            dpi = im.info.get("dpi", (0, 0))
    except Exception as e:
        print(f"  fig{i}.jpg: unreadable ({e})")
        ok = False
        continue
    good = (jpeg and fmt == "JPEG" and w == W and h == H
            and abs(dpi[0] - DPI) < 1 and abs(dpi[1] - DPI) < 1)
    ok &= good
    print(f"  fig{i}.jpg  {w}x{h} px  dpi={dpi[0]:.0f}  "
          f"{p.stat().st_size/1024:.0f} KB  {'OK' if good else 'FAIL'}")
print("ALL FIGURES OK" if ok else "FAIL")
sys.exit(0 if ok else 2)
