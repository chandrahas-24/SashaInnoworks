from PIL import Image
import math
import os

img = Image.open("bayblade.png").convert("RGBA")

scale = 0.9
img = img.resize(
    (int(img.width * scale), int(img.height * scale)),
    Image.Resampling.LANCZOS
)

canvas_size = math.ceil(math.sqrt(img.width**2 + img.height**2))

fps = 30
duration = 5
revolutions = 12

N = fps * duration
frames = []

for i in range(N):
    t = i / (N - 1)

    if t < 0.5:
        progress = 2 * t * t
    else:
        progress = 1 - 2 * (1 - t) * (1 - t)

    angle = 360 * revolutions * progress

    rotated = img.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        expand=True
    )

    frame = Image.new(
        "RGBA",
        (canvas_size, canvas_size),
        (0, 0, 0, 0)
    )

    x = (canvas_size - rotated.width) // 2
    y = (canvas_size - rotated.height) // 2

    frame.paste(rotated, (x, y), rotated)

    # Quantize while preserving transparency better
    frame = frame.quantize(
        colors=64,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE
    )

    frames.append(frame)

frames[0].save(
    "bayblade.gif",
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / fps),
    loop=0,
    disposal=2,
    optimize=True
)

size_mb = os.path.getsize("bayblade.gif") / (1024 * 1024)
print(f"Saved bayblade.gif ({size_mb:.2f} MB)")