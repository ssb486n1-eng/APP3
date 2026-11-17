import cv2, numpy as np, sys
from PIL import Image, ImageOps
BG = "/root/.claude/uploads/fdf51385-72c5-56f5-bba4-a724897da24d/f2da9e55-image.jpg"
bg = np.asarray(ImageOps.exif_transpose(Image.open(BG)).convert("RGB")).astype(np.float32)
H, W = bg.shape[:2]
cut = np.asarray(Image.open("cut_birefnet-general.png")).astype(np.float32)
rgb, a = cut[..., :3], cut[..., 3] / 255
ys, xs = np.where(a > 0.5); rgb, a = rgb[ys.min():ys.max()+1, xs.min():xs.max()+1], a[ys.min():ys.max()+1, xs.min():xs.max()+1]

# 1) kill print halftone before scaling
rgb = cv2.bilateralFilter(cv2.GaussianBlur(rgb, (0, 0), 3.2), 11, 30, 11)

# 2) scale: target height (incl. hoop) in bg pixels
TH, CX, BOT = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
s = TH / rgb.shape[0]
nw, nh = int(rgb.shape[1] * s), int(rgb.shape[0] * s)
rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
a = cv2.erode(np.clip((a - 0.15) / 0.7, 0, 1), np.ones((3, 3), np.uint8)); a = cv2.GaussianBlur(a, (0, 0), 1.2)  # tighten + soft edge

x0, y0 = int(CX - nw / 2), int(BOT - nh)
region = bg[y0:y0+nh, x0:x0+nw]

# 3) colour match in Lab: neutralise orange print cast toward room light
lab = cv2.cvtColor(rgb / 255, cv2.COLOR_RGB2LAB)
blab = cv2.cvtColor(bg[int(H*.3):int(H*.9), int(W*.15):int(W*.85)] / 255, cv2.COLOR_RGB2LAB)
m = a > 0.5
for c, k in [(0, 0.35), (1, 0.55), (2, 0.55)]:
    mu, sd = lab[..., c][m].mean(), lab[..., c][m].std()
    bmu = blab[..., c].mean()
    shift = (bmu - mu) * k if c else (bmu - mu) * k * 0.4
    lab[..., c] = lab[..., c] + shift
lab[..., 1:] *= 0.86                    # print is over-saturated
lab[..., 0] = np.clip(lab[..., 0] * 0.93 + 7, 0, 100)
rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255)

# soft window light from the right (room's light source)
gx = np.linspace(0.94, 1.04, nw)[None, :, None]
rgb = np.clip(rgb * gx, 0, 255)

# 4) shadows on sofa: contact shadow + soft cast shadow to the left/back
sh = np.zeros((H, W), np.float32)
am = np.zeros((H, W), np.float32); am[y0:y0+nh, x0:x0+nw] = a
cast = cv2.warpAffine(am, np.float32([[1, 0, -nw * 0.06], [0, 1, -nh * 0.01]]), (W, H))
sh += cv2.GaussianBlur(cast, (0, 0), 35) * 0.35
ell = np.zeros((H, W), np.float32)
cv2.ellipse(ell, (int(CX), int(BOT - 10)), (int(nw * 0.42), int(nh * 0.035)), 0, 0, 360, 1, -1)
sh += cv2.GaussianBlur(ell, (0, 0), 25) * 0.55
out = bg * (1 - np.clip(sh, 0, 0.6)[..., None])

# 5) light wrap: bleed a bit of background into the edges
blurbg = cv2.GaussianBlur(out[y0:y0+nh, x0:x0+nw], (0, 0), 8)
edge = np.clip(cv2.GaussianBlur(a, (0, 0), 4) - a, 0, 1)[..., None] * 0  # placeholder
inner = (1 - cv2.GaussianBlur(a, (0, 0), 3))[..., None] * a[..., None]
rgb = rgb * (1 - 0.35 * inner) + blurbg * 0.35 * inner

# 6) match grain/softness of phone photo
rgb = rgb + np.random.default_rng(0).normal(0, 1.6, rgb.shape)
A = a[..., None]
out[y0:y0+nh, x0:x0+nw] = rgb * A + out[y0:y0+nh, x0:x0+nw] * (1 - A)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(sys.argv[4], quality=94)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((W // 2, H // 2)).save(sys.argv[4] + ".prev.jpg", quality=85)
print(W, H, nw, nh, x0, y0)
