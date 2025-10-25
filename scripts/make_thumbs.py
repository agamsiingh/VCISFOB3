from pathlib import Path
from PIL import Image

SRC = Path(__file__).resolve().parent.parent / "image"
OUT = SRC / "thumbs"
OUT.mkdir(parents=True, exist_ok=True)
MAX_WIDTH = 1280  # change this to lower value like 800 for smaller thumbs

exts = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}

print(f"Scanning {SRC} for images and writing thumbnails to {OUT}...")
for p in sorted(SRC.iterdir()):
    if p.is_dir() and p.name == "thumbs":
        continue
    if p.suffix.lower() not in exts:
        continue
    out = OUT / p.name
    try:
        with Image.open(p) as im:
            w, h = im.size
            if w <= MAX_WIDTH:
                # If already small, just copy (preserve format)
                im.save(out)
            else:
                ratio = MAX_WIDTH / float(w)
                new_h = int(h * ratio)
                im = im.resize((MAX_WIDTH, new_h), Image.LANCZOS)
                # Save with reasonable quality for lossy formats
                if p.suffix.lower() in {'.jpg', '.jpeg'}:
                    im.save(out, optimize=True, quality=85)
                else:
                    im.save(out)
        print(f"Saved {out} ({out.stat().st_size} bytes)")
    except Exception as e:
        print(f"Failed to process {p}: {e}")

print("Thumbnail generation complete.")
