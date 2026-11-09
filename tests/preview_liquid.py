"""Deterministic synthetic-only native-frame preview and CPU benchmark."""
import sys
import math
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from test_liquid import Source, tone
import heliosctl as h

images = []
comparisons = []
timings = []
now = [0.]
source = Source(150)
frames = h.AudioFrames(source, clock=lambda: now[0], pattern='liquid', gain=.6,
                       base_size=.35, expansion=.25, ripple_depth=.07,ripple_frequency=8)
for i in range(240):
    hz = (150,800,4000)[(i//80)%3]
    amplitude = .04 + .18 * max(0,math.sin(i*.22))**4
    source.samples = tone(hz,amplitude)
    start = time.perf_counter()
    now[0] += .025
    native = frames[0]
    timings.append((time.perf_counter()-start)*1000)
    im = Image.new('RGB',(600,450),(8,18,30))
    draw = ImageDraw.Draw(im)
    xy = [(300+(p.x*16-32768)*.018,240-(p.y*16-32768)*.018) for p in native]
    draw.line(xy,fill=(70,165,245),width=2)
    draw.text((15,15),f'LIQUID / synthetic {hz} Hz / t={now[0]:.2f}s',fill='white')
    draw.text((15,35),'native contour; gain .6 depth .07 frequency 8 base .35 expansion .25',fill=(150,170,190))
    if i%2 == 0: images.append(im)
    if i in (60,140,220): comparisons.append(im)
images[0].save('/tmp/helios-liquid.gif',save_all=True,append_images=images[1:],duration=50,loop=0)
comparison = Image.new('RGB',(1800,450))
for i,im in enumerate(comparisons): comparison.paste(im,(i*600,0))
comparison.save('/tmp/helios-liquid-comparison.png')
print('processing+native geometry ms mean=%.3f p95=%.3f max=%.3f / 25ms frame' % (sum(timings)/len(timings),sorted(timings)[int(.95*len(timings))],max(timings)))
