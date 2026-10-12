"""Recolor the two hand-drawn paper figures (raster) to the muted palette of the other figures.
Input: the images embedded in MARGENT_ACL_ARR_revised.pdf (page 1 = Figure 1, page 8 = Figure 4), extracted to PNG.
Hue families are remapped, keeping their meaning: green -> the paper's blue (chosen path / keep / correct), red -> muted
brick red (wrong / discard), bright blue and purple -> muted steel blue, yellow/orange -> muted warm tones; saturation is
capped so nothing stays neon. Black, gray and white pixels are left exactly as they are.
usage: python3 recolor_hand_figs.py IN.png OUT.png [max_width]"""
import sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# (hue range in degrees) -> (target hue, saturation scale, saturation cap, value scale)
BANDS = [
    ((345, 360), (3, 0.80, 0.52, 0.86)),    # red -> muted brick red (#B9524F family)
    ((0, 15), (3, 0.80, 0.52, 0.86)),
    ((15, 48), (32, 0.55, 0.42, 0.96)),     # orange -> muted sand
    ((48, 70), (40, 0.50, 0.38, 0.97)),     # yellow -> muted wheat
    ((70, 170), (207, 0.95, 0.50, 0.80)),   # green -> paper blue (#4A7BA6 / #2C5A7E)
    ((170, 255), (208, 0.70, 0.45, 0.88)),  # cyan/blue -> muted steel blue
    ((255, 345), (214, 0.55, 0.35, 0.85)),  # purple/magenta -> slate blue
]

# value caps for vivid pixels by target hue: brick red #B9524F (0.73), blue #4A7BA6 (0.65), steel blue, slate blue
VCAP = {3: 0.74, 207: 0.64, 208: 0.70, 214: 0.62}


def recolor(img):
    rgba = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
    rgb, a = rgba[..., :3], rgba[..., 3:]
    mx, mn = rgb.max(-1), rgb.min(-1); v = mx; c = mx - mn
    s = np.where(mx > 0, c / np.maximum(mx, 1e-6), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.zeros_like(mx)
    m = c > 1e-6
    hr = ((g - b) / np.maximum(c, 1e-6)) % 6; hg = (b - r) / np.maximum(c, 1e-6) + 2; hb = (r - g) / np.maximum(c, 1e-6) + 4
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb)) * 60.0
    h = np.where(m, h, 0) % 360
    nh, ns, nv = h.copy(), s.copy(), v.copy()
    colored = s > 0.12  # leave black / gray / white untouched
    for (lo, hi), (th, sf, sc, vf) in BANDS:
        sel = colored & (h >= lo) & (h < hi)
        nh[sel] = th
        ns[sel] = np.minimum(s[sel] * sf, sc)
        # light tints stay light; saturated mid tones get a little darker for print contrast
        nv[sel] = np.where(v[sel] > 0.9, v[sel], v[sel] * vf + (1 - vf) * 0.15)
        cap = VCAP.get(th)
        if cap is not None:  # vivid pixels (text, lines) darken to the paper colour; pale tints and edges blend smoothly
            w = np.clip((s[sel] - 0.15) / 0.30, 0, 1)
            nv[sel] = nv[sel] * (1 - w) + np.minimum(nv[sel], cap) * w
    # very light, slightly tinted areas (panel backgrounds) -> light blue-gray
    bg = colored & (v > 0.86) & (s < 0.25)
    nh[bg] = 210; ns[bg] = np.minimum(s[bg], 0.07); nv[bg] = np.maximum(v[bg], 0.94)
    # faint green tints (panel background, its border, gray-green arrows) -> the same faint blue-gray
    tint = (~colored) & (s > 0.02) & (h >= 70) & (h < 170)
    nh[tint] = 212
    colored = colored | tint
    # HSV -> RGB
    hh = (nh / 60.0) % 6; cc = nv * ns; x = cc * (1 - np.abs(hh % 2 - 1)); m0 = nv - cc
    z = np.zeros_like(hh)
    i = np.floor(hh).astype(int)
    rp = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [cc, x, z, z, x, cc])
    gp = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [x, cc, cc, x, z, z])
    bp = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [z, z, x, cc, cc, x])
    out = np.stack([rp + m0, gp + m0, bp + m0], -1)
    out = np.where(colored[..., None], out, rgb)
    return Image.fromarray((np.clip(np.concatenate([out, a], -1), 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]; maxw = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    im = Image.open(src)
    if maxw and im.size[0] > maxw:
        im = im.resize((maxw, round(im.size[1] * maxw / im.size[0])), Image.LANCZOS)
    recolor(im).save(dst, optimize=True); print(dst, im.size)
