import json
from pathlib import Path

data = json.load(open('scratch/u9a30_raw_data.json', encoding='utf-8'))
svg_outline = data['svg_outline']
skel = [p for p in data['skel_pixels'] if p[0] >= 42 and p[1] >= 65]

svg_skel = ''.join([f'<circle cx="{p[0]}" cy="{p[1]}" r="0.15" fill="#38bdf8" />' for p in skel])

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Crop Ma Bottom</title>
</head>
<body style="background: #0f172a; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center;">
  <h2 style="color: #f8fafc; font-family: sans-serif;">馬部底座与四点底高分辨率骨架与轮廓</h2>
  <div style="width: 800px; height: 500px; background: #1e293b; border-radius: 8px; border: 1px solid #334155; overflow: hidden;">
    <svg viewBox="40 65 52 30" style="width: 100%; height: 100%;">
      <path d="{svg_outline}" fill="rgba(255,255,255,0.06)" stroke="#64748b" stroke-width="0.15" />
      {svg_skel}
    </svg>
  </div>
</body>
</html>
"""

out_path = Path("artifacts/deterministic-decomposer/crop_ma_bottom.html")
out_path.write_text(html, encoding="utf-8")
print("Saved crop_ma_bottom.html successfully!")
