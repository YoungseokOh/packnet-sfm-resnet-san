#!/usr/bin/env python3
"""
Create an index.html for Fin_Test_Set_ncdb/viz with thumbnails linking to visualization images.
"""
from pathlib import Path
import argparse
from PIL import Image


def make_thumbnail(inpath: Path, thumb: Path, size=(300, 200)):
    thumb.parent.mkdir(parents=True, exist_ok=True)
    if not thumb.exists():
        try:
            img = Image.open(inpath)
            img.thumbnail(size)
            img.save(thumb)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz_dir', type=str, default='Fin_Test_Set_ncdb/viz')
    parser.add_argument('--out_html', type=str, default='Fin_Test_Set_ncdb/viz/index.html')
    args = parser.parse_args()

    viz_dir = Path(args.viz_dir)
    images = sorted([p for p in viz_dir.glob('*_rgb_gt_fp32.png')])
    thumbs = []
    for p in images:
        thumb = viz_dir / ('thumb_' + p.name)
        make_thumbnail(p, thumb)
        thumbs.append((p, thumb))

    html_lines = ['<html><head><meta charset="utf-8"><title>Fin Test Set Visualizations</title></head><body>']
    html_lines.append('<h1>Fin Test Set Visualizations</h1>')
    html_lines.append('<div style="display:flex;flex-wrap:wrap">')
    for p, t in thumbs:
        html_lines.append(
            f"<div style='margin:8px;'><a href='{p.name}' target='_blank'><img src='{t.name}' style='width:300px;'/></a><div style='font-size:12px'>{p.name}</div></div>"
        )
    html_lines.append('</div></body></html>')
    out_html = Path(args.out_html)
    with open(out_html, 'w') as f:
        f.write('\n'.join(html_lines))
    print('Wrote HTML index to', out_html)


if __name__ == '__main__':
    main()
