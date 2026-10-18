# 1440x900 synthetic desktop: 8 labelled controls with known pixel centres -> screen.png + screen.json
import json
from PIL import Image, ImageDraw, ImageFont
W, H = 1440, 900
im = Image.new("RGB", (W, H), (236, 238, 242)); d = ImageDraw.Draw(im)
def font(sz):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, sz)
        except OSError: pass
d.rectangle([0, 0, W, 34], fill=(250, 250, 250)); d.text((14, 8), "Finder   File   Edit   View   Go   Window   Help", fill=(30, 30, 30), font=font(17))
d.rectangle([180, 90, 1260, 820], fill=(255, 255, 255), outline=(190, 190, 190), width=2)
d.text((210, 110), "Settings - jarvis assistant", fill=(20, 20, 20), font=font(26))
# (label, box) - centres are the ground truth for clicks
C = [("jarvis", (220, 180, 400, 230)), ("Einstellungen", (460, 180, 700, 230)), ("Отправить", (760, 180, 960, 230)),
     ("Save draft", (1020, 180, 1220, 230)), ("Order 4711-B", (220, 420, 470, 470)), ("Kalender", (560, 420, 760, 470)),
     ("Загрузки", (860, 420, 1060, 470)), ("Cancel", (1060, 740, 1220, 790))]
cols = [(52, 120, 246), (40, 160, 90), (220, 80, 60), (120, 90, 200), (230, 150, 30), (20, 150, 160), (200, 60, 150), (110, 110, 110)]
out = []
for (lab, b), c in zip(C, cols):
    d.rounded_rectangle(b, 10, fill=c); f = font(22); tw = d.textlength(lab, font=f)
    d.text(((b[0] + b[2] - tw) / 2, b[1] + 13), lab, fill=(255, 255, 255), font=f)
    out.append(dict(label=lab, cx=(b[0] + b[2]) / 2, cy=(b[1] + b[3]) / 2))
d.text((220, 300), "Status: 3 new messages since 07:45", fill=(60, 60, 60), font=font(20))
im.save("screen.png"); json.dump(dict(W=W, H=H, controls=out), open("screen.json", "w"), indent=1)
