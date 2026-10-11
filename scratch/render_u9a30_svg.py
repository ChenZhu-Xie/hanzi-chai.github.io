import json
from pathlib import Path

data = json.load(open("scratch/u9a30_raw_data.json", encoding="utf-8"))
svg_outline = data["svg_outline"]
skel_pts = data["skel_pixels"]

skel_circles = "".join([f'<circle cx="{p[0]}" cy="{p[1]}" r="0.4" fill="red" />' for p in skel_pts])

svg_str = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="800" height="800" style="background:#111;">
  <path d="{svg_outline}" fill="rgba(255,255,255,0.2)" stroke="white" stroke-width="0.3" />
  {skel_circles}
</svg>"""

Path("scratch/u9a30_debug.svg").write_text(svg_str, encoding="utf-8")
print("Saved scratch/u9a30_debug.svg")
