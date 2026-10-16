"""Problem 7: what the model looks at.

This is not part of the assignment. It makes the three changes that Problem 7
of the long report describes, on four of the student's own images from
data/new_images/, measures the exported model on the original and on the
changed copy of each one, and prints every answer.

The changes are the ones the guesses in Problem 7 ask for:

  * the printed name of the type is covered with a grey box,
  * the bell and the mount in the middle of the frame are covered the same way,
  * the photograph is put on a grey margin, so that the square the model gets
    holds the whole cymbal, rim and all, instead of the middle of it.

The boxes are written as fractions of the 224 x 224 square the model gets, so
they cover the same part of the frame in every image, and they are put back
onto the photograph at its own size. The changed copies are written under
data/changed/, and the pictures for the report under long_report/figures/.
Nothing in data/new_images/ is changed.

Run it from the repository folder:

    python src/problem7.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import check  # noqa: E402  the exported-model pipeline check.py uses

NEW = ROOT / "data" / "new_images"
CHANGED = ROOT / "data" / "changed"
FIGURES = ROOT / "long_report" / "figures"
OUT = ROOT / "results" / "problem7.txt"
GREY = (128, 128, 128)

# The photographs used, and where the printing and the mount are in the 224 x 224
# square the model gets from each one, as fractions of that square.
PHOTOS = {
    "china_cymbal/003.png": {
        "what": "a Meinl B8 china from above, the printing 'B8' and 'meinl'",
        "printing": [(0.20, 0.00, 0.55, 0.18), (0.34, 0.60, 0.82, 0.84)],
        "mount": (0.44, 0.26, 0.63, 0.62),
    },
    "crash_cymbal/003.png": {
        "what": "a Sabian SBR crash from above, the printing 'SABIAN'",
        "printing": [(0.30, 0.58, 1.00, 0.92)],
        "mount": (0.43, 0.00, 0.67, 0.36),
    },
    "splash_cymbal/005.jpg": {
        "what": "an Istanbul XIST splash on its stand, the printing 'XIST' and 'istanbul'",
        "printing": [(0.23, 0.00, 0.45, 0.21), (0.43, 0.68, 0.97, 1.00)],
        "mount": (0.36, 0.26, 0.62, 0.56),
    },
    "hihat_cymbal/005.jpg": {
        "what": "a hihat from above on its stand, the printing 'HHX' on the metal, the clutch at the top",
        "printing": [(0.36, 0.53, 0.64, 0.82)],
        "mount": (0.26, 0.00, 0.64, 0.26),
    },
    "ride_cymbal/003.png": {
        "what": "a Paiste Monad ride from above, the bell and the nut in the middle",
        "printing": [(0.30, 0.18, 0.85, 0.52)],
        "mount": (0.36, 0.30, 0.66, 0.66),
    },
}

# The three changes, and the photographs each one is measured on.
GUESSES = [
    {
        "number": 1,
        "what": "cover the printed name of the type",
        "kind": "printing",
        "photos": ["china_cymbal/003.png", "crash_cymbal/003.png",
                   "splash_cymbal/005.jpg", "hihat_cymbal/005.jpg"],
    },
    {
        "number": 2,
        "what": "cover the bell and the mount in the middle of the frame",
        "kind": "mount",
        "photos": ["china_cymbal/003.png", "crash_cymbal/003.png",
                   "splash_cymbal/005.jpg", "hihat_cymbal/005.jpg",
                   "ride_cymbal/003.png"],
    },
    {
        "number": 3,
        "what": "put the photograph on a grey margin, so the whole cymbal fits",
        "kind": "margin",
        "photos": ["china_cymbal/003.png", "crash_cymbal/003.png",
                   "splash_cymbal/005.jpg", "hihat_cymbal/005.jpg"],
    },
]

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def settings():
    return json.loads((ROOT / "docs" / "model.json").read_text())


def square_box_to_photo(size, box):
    """The 224 square is the middle of the photograph, shorter side 256."""
    model = settings()
    resize, crop = int(model["input"]["resize"]), int(model["input"]["crop"])
    width, height = size
    short, long = min(width, height), max(width, height)
    if width <= height:
        new = (resize, int(resize * long / short))
    else:
        new = (int(resize * long / short), resize)
    left, top = (new[0] - crop) / 2.0, (new[1] - crop) / 2.0
    factor = (new[0] / width, new[1] / height)
    return (round((left + box[0] * crop) / factor[0]),
            round((top + box[1] * crop) / factor[1]),
            round((left + box[2] * crop) / factor[0]),
            round((top + box[3] * crop) / factor[1]))


def cover(picture, boxes):
    """The same photograph with grey boxes over the parts named."""
    out = picture.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        draw.rectangle(square_box_to_photo(picture.size, box), fill=GREY)
    return out


def margin(picture, scale=0.7):
    """The same photograph on a grey margin, so the whole cymbal fits in the
    square the model gets."""
    width, height = picture.size
    small = picture.resize((max(1, int(width * scale)), max(1, int(height * scale))),
                           Image.LANCZOS)
    out = Image.new("RGB", (width, height), GREY)
    out.paste(small, ((width - small.width) // 2, (height - small.height) // 2))
    return out


def save(source, target, longest=1100):
    if isinstance(source, Image.Image):
        picture = source.convert("RGB")
    else:
        picture = Image.open(source).convert("RGB")
    if max(picture.size) > longest:
        factor = longest / max(picture.size)
        picture = picture.resize((max(1, round(picture.width * factor)),
                                  max(1, round(picture.height * factor))),
                                 Image.BILINEAR)
    if target.suffix.lower() == ".png":
        picture.save(target, optimize=True)
    else:
        picture.save(target, quality=88)


def answers(session, labels, picture):
    logits = check.forward(session, check.prepare(picture, settings()))
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    order = np.argsort(logits)[::-1]
    return probabilities, [(labels[i], float(probabilities[i])) for i in order[:3]]


def check_square(path, resize, crop):
    picture = Image.open(path).convert("RGB")
    width, height = picture.size
    short, long = min(width, height), max(width, height)
    if width <= height:
        new = (resize, int(resize * long / short))
    else:
        new = (int(resize * long / short), resize)
    small = picture.resize(new, Image.BILINEAR)
    left, top = int(round((new[0] - crop) / 2.0)), int(round((new[1] - crop) / 2.0))
    return small.crop((left, top, left + crop, top + crop))


def short(name):
    return name.replace("_cymbal", "")


def main():
    model = settings()
    labels = model["labels"]
    import onnxruntime
    session = onnxruntime.InferenceSession(
        str(ROOT / "docs" / "model.onnx"), providers=["CPUExecutionProvider"])

    CHANGED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    def change(kind, picture, item):
        if kind == "margin":
            return margin(picture)
        boxes = item["printing"] if kind == "printing" else [item["mount"]]
        return cover(picture, boxes)

    say("Problem 7. What the model looks at")
    say("=" * 72)
    say("The photographs, and what the exported model says about each one before")
    say("any change. Every answer below is docs/model.onnx, through the pipeline")
    say("check.py writes again from docs/model.json, which is the model the page")
    say("serves.")
    say()
    numbers = {}
    for number, (name, item) in enumerate(PHOTOS.items(), 1):
        path = NEW / name
        probabilities, top = answers(session, labels, Image.open(path))
        really = name.split("/")[0]
        numbers[name] = number
        say(f"  photo {number}: data/new_images/{name}, really {really}")
        say(f"    {item['what']}")
        say("    answer: " + ", ".join(f"{n} {p:.1%}" for n, p in top)
            + ("" if top[0][0] == really else "   <- a mistake"))
        save(path, FIGURES / f"photo_{number}.jpg")

    for guess in GUESSES:
        number, kind = guess["number"], guess["kind"]
        say()
        say("-" * 72)
        say(f"Guess {number}. {guess['what']}")
        say(f"  measured on {len(guess['photos'])} photographs, "
            + " ".join(f"photo {numbers[n]}" for n in guess["photos"]))
        say()
        for step, name in enumerate(guess["photos"], 1):
            item = PHOTOS[name]
            path = NEW / name
            folder = CHANGED / f"guess{number}" / name.split("/")[0]
            folder.mkdir(parents=True, exist_ok=True)
            picture = change(kind, Image.open(path).convert("RGB"), item)
            picture.save(folder / name.split("/")[1], quality=95)
            before_probabilities, before = answers(
                session, labels, Image.open(path))
            after_probabilities, after = answers(session, labels, picture)
            really = name.split("/")[0]
            say(f"  photo {numbers[name]} ({name})")
            say("    before: " + ", ".join(f"{n} {p:.1%}" for n, p in before))
            say("    after:  " + ", ".join(f"{n} {p:.1%}" for n, p in after))
            say("    the answer "
                + ("changed" if before[0][0] != after[0][0] else "stayed the same")
                + f": {short(before[0][0])} {before[0][1]:.1%} -> "
                + f"{short(after[0][0])} {after[0][1]:.1%}")
            say(f"    really {really}: "
                f"{before_probabilities[labels.index(really)]:.1%} before, "
                f"{after_probabilities[labels.index(really)]:.1%} after")
            save(picture, FIGURES / f"guess{number}_changed_{step}.jpg")

    other = CHANGED / "none_of_the_classes"
    if other.exists():
        say()
        say("=" * 72)
        say("A photograph of something that is none of the five classes, put into")
        say("the page. The page has no 'none of these' answer, so it names one of")
        say("its five classes anyway.")
        say()
        where = other / "source.txt"
        if where.exists():
            for row in where.read_text(encoding="utf-8").splitlines():
                say(f"  {row}")
            say()
        for path in sorted(p for p in other.iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            _, top = answers(session, labels, Image.open(path))
            say(f"  {path.name}: " + ", ".join(f"{n} {p:.1%}" for n, p in top))
            save(path, FIGURES / f"none_{path.stem}.jpg")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()