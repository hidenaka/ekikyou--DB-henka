import os
import json

base_dir = "/Users/nakanohideaki/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/images/yao_icons"

html_content = """
<!DOCTYPE html>
<html>
<head>
<style>
  body { font-family: sans-serif; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 5px; text-align: center; }
  .present { background-color: #90ee90; }
  .missing { background-color: #ffcccb; }
  .layout-present { background-color: #add8e6; }
</style>
</head>
<body>
<h1>Yao Icons Progress Report</h1>
<table>
  <thead>
    <tr>
      <th>Hexagram</th>
      <th>Line 1</th>
      <th>Line 2</th>
      <th>Line 3</th>
      <th>Line 4</th>
      <th>Line 5</th>
      <th>Line 6</th>
    </tr>
  </thead>
  <tbody>
"""

total_art = 0
total_layout = 0

for i in range(1, 65):
    hex_dir = os.path.join(base_dir, f"hex_{i:02d}")
    row_html = f"<tr><td>{i:02d}</td>"
    
    if os.path.exists(hex_dir):
        files = os.listdir(hex_dir)
    else:
        files = []

    for line_num in range(1, 7):
        # Check for artwork (exclude layout)
        has_art = False
        has_layout = False
        
        for f in files:
            if f"line_{line_num}_" in f and "layout" not in f and f.endswith(".png"):
                has_art = True
            if f"line_{line_num}_layout.png" in f or f"line_{line_num}_layout" in f: # lax check
                 if "layout" in f: has_layout = True

        cell_class = ""
        status = "Missing"
        
        if has_art:
            cell_class = "present"
            status = "Art OK"
            total_art += 1
        elif has_layout:
             # If only layout exists but no art
             cell_class = "layout-present" 
             status = "Layout Only"
             total_layout += 1
        else:
             cell_class = "missing"
        
        if has_art and has_layout:
            status = "Both OK"
        
        row_html += f"<td class='{cell_class}'>{status}</td>"
    
    row_html += "</tr>"
    html_content += row_html

html_content += f"""
  </tbody>
</table>
<h2>Summary</h2>
<p>Total Artwork Images: {total_art} / 384</p>
<p>Total Layout Images: {total_layout} / 384 (approx)</p>
</body>
</html>
"""

print(html_content)
