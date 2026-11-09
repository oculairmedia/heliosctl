"""Offline deterministic synthetic-beat comparison; never constructs an SDK."""
import importlib.util
from pathlib import Path
from PIL import Image, ImageDraw

spec = importlib.util.spec_from_file_location('helios', Path(__file__).parents[1] / 'heliosctl.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
image = Image.new('RGB', (1200, 680), '#10151e')
draw = ImageDraw.Draw(image)
for column, (depth, frequency, label) in enumerate([
        (.035, 3, 'BEFORE: depth .035 / spatial cycles 3'),
        (.07, 5, 'AFTER: depth .07 / spatial cycles 5')]):
    contour = h.RippleContour(11, depth, frequency)
    detector = h.OnsetDetector()
    for i in range(81):
        t = i * .025
        level = .8 if i % 24 in (4, 5, 6) else .01
        coords = contour.sample(.8, t, .025, detector.update(level, t, .025))
        if i in (16, 40, 64):
            row = (i - 16) // 24
            center = (300+column*600, 145+row*205)
            points = [(center[0]+x*.016, center[1]-y*.0068) for x,y in coords]
            draw.line(points, fill=(80,160,255), width=2)
            draw.text((column*600+20, 40+row*205), f't = {t:.2f}s, same synthetic beats / seed 11', fill='white')
    draw.text((column*600+20, 12), label, fill='white')
image.save('/tmp/helios-ripple-tuning.png')
