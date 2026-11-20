#!/usr/bin/env python3
"""Render the KingBlocker9000 boot splash and text screen images.

Run on a development machine (Pillow + the Jost font), not on a unit; the
PNGs it writes are committed under assets/screens/ and used by the Plymouth
boot splash (plymouth/kingblocker/) and by Minus's INITIALIZING and NO
SIGNAL displays (src/boot_screen.py).

    python3 tools/render_screens.py --mockup boot_mockup.png --font Jost-Italic.ttf

  --mockup  the 1920x1080 boot screen design (logo + "A Kalshi INVENTION"
            on #28CC95, white loading bar at y=950). The bar is painted out;
            Plymouth draws it itself.
  --font    Jost italic variable font (SIL OFL, github.com/google/fonts,
            ofl/jost/Jost-Italic[wght].ttf), used for the text screens: a
            Futura-style heavy italic like the logo lettering.

Outputs (assets/screens/):
  boot_background.png  design without the bar
  no_signal.png        green screen, NO SIGNAL centred
  initializing.png     green screen, INITIALIZING centred
  bar_track.png        the bar's empty track (white, 22% opaque, round ends)
  bar_fill_left.png / bar_fill_mid.png / bar_fill_right.png
                       the white fill in three pieces, so Plymouth can stretch
                       the middle without squashing the round ends
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'assets' / 'screens'

W, H = 1920, 1080
GREEN = (0x28, 0xCC, 0x95)
# Loading bar, measured on the design (keep in sync with src/boot_screen.py
# and plymouth/kingblocker/kingblocker.script).
BAR_X, BAR_Y, BAR_W, BAR_H = 680, 950, 560, 10
TRACK_ALPHA = 0.22          # the design's track is #57D7AC = white at 22% on green
SS = 8                      # supersampling for anti-aliased round ends


def pill(w, h, alpha):
    big = Image.new('L', (w * SS, h * SS), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, w * SS - 1, h * SS - 1),
                                          radius=h * SS // 2, fill=255)
    mask = big.resize((w, h), Image.LANCZOS).point(lambda v: round(v * alpha))
    img = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    img.putalpha(mask)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mockup', required=True)
    ap.add_argument('--font', required=True)
    ap.add_argument('--weight', type=int, default=900)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Boot background: the design with its bar painted out.
    bg = Image.open(args.mockup).convert('RGB')
    if bg.size != (W, H):
        raise SystemExit(f'mockup must be {W}x{H}, got {bg.size}')
    ImageDraw.Draw(bg).rectangle((BAR_X - 4, BAR_Y - 4, BAR_X + BAR_W + 4, BAR_Y + BAR_H + 4),
                                 fill=GREEN)
    bg.save(OUT / 'boot_background.png', optimize=True)

    # Bar pieces.
    pill(BAR_W, BAR_H, TRACK_ALPHA).save(OUT / 'bar_track.png')
    fill = pill(BAR_H * 2, BAR_H, 1.0)
    r = BAR_H // 2
    fill.crop((0, 0, r, BAR_H)).save(OUT / 'bar_fill_left.png')
    fill.crop((r, 0, r + 1, BAR_H)).save(OUT / 'bar_fill_mid.png')
    fill.crop((BAR_H * 2 - r, 0, BAR_H * 2, BAR_H)).save(OUT / 'bar_fill_right.png')

    # Text screens: heavy italic, white, centred, slightly condensed like the
    # logo lettering; same size on both so they match.
    font = ImageFont.truetype(args.font, 150)
    try:
        font.set_variation_by_axes([args.weight])
    except Exception as e:  # static font file: use as is
        print(f'note: no weight axis ({e})')
    for text, name in (('NO SIGNAL', 'no_signal.png'), ('INITIALIZING', 'initializing.png')):
        text_screen(text, font).save(OUT / name, optimize=True)
    print('wrote', ', '.join(sorted(p.name for p in OUT.glob('*.png'))))


def text_screen(text, font, condense=0.88):
    l, t, r, b = font.getbbox(text)
    layer = Image.new('L', (r - l + 40, b - t + 40), 0)
    ImageDraw.Draw(layer).text((20 - l, 20 - t), text, font=font, fill=255)
    layer = layer.resize((round(layer.width * condense), layer.height), Image.LANCZOS)
    img = Image.new('RGB', (W, H), GREEN)
    img.paste((255, 255, 255), ((W - layer.width) // 2, (H - layer.height) // 2), layer)
    return img


if __name__ == '__main__':
    main()
