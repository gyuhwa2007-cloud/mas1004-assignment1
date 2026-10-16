"""Make the pictures and the counts that the long report needs.

This is not part of the assignment. It only

    * copies the pictures that the given code already wrote into results/,
    * makes the few small pictures the template asks for (an image next to what
      prepare_image made of it, one picture of each kind of junk, the mistakes
      the exported model was most sure about) out of files that are on disk,
    * and re-measures the saved models, so that every number in the report is a
      number that a program here printed.

Run it from the repository folder:

    python src/report_assets.py

It writes long_report/figures/*.png|jpg and results/report_numbers.txt, and
prints the same text on the screen. Nothing in data/ is changed.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import check  # noqa: E402  the exported-model pipeline check.py uses
from data import prepare_image  # noqa: E402  the student's own function
from evaluate import confusion_matrix  # noqa: E402

FIGURES = ROOT / "long_report" / "figures"
RESULTS = ROOT / "results"
CLEAN = ROOT / "data" / "clean"
REMOVED = ROOT / "data" / "removed"
NEW = ROOT / "data" / "new_images"
LOG = REMOVED / "log.csv"
OUT = RESULTS / "report_numbers.txt"
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def images_in(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in SUFFIXES)


def save_small(source, target, longest=1200):
    """Copy a picture, shrinking it if it is big, so that the zip stays small."""
    picture = Image.open(source).convert("RGB")
    if max(picture.size) > longest:
        scale = longest / max(picture.size)
        picture = picture.resize(
            (max(1, round(picture.width * scale)),
             max(1, round(picture.height * scale))), Image.BILINEAR)
    if target.suffix.lower() == ".png":
        picture.save(target, optimize=True)
    else:
        picture.save(target, quality=88)


def web_model():
    model = json.loads((ROOT / "docs" / "model.json").read_text())
    import onnxruntime
    session = onnxruntime.InferenceSession(
        str(ROOT / "docs" / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    return model, session


def copy_run_pictures(tags):
    """The curves, the confusion matrix and the worst mistakes of every run."""
    say("\nPictures copied from results/ into long_report/figures/")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        for kind in ("curves", "confusion", "worst"):
            source = RESULTS / f"{tag}_{kind}.png"
            if source.exists():
                shutil.copyfile(source, FIGURES / f"{tag}_{kind}.png")
                say(f"  {source.name:26s} -> figures/{tag}_{kind}.png")


def prepare_picture():
    """One image next to the 224 x 224 the model actually gets from it."""
    path = CLEAN / "splash_cymbal" / "0001.jpg"
    original = Image.open(path).convert("RGB")
    settings = json.loads((ROOT / "docs" / "model.json").read_text())["input"]
    resize, crop = int(settings["resize"]), int(settings["crop"])
    mean = np.array(settings["mean"], dtype=np.float32)
    std = np.array(settings["std"], dtype=np.float32)

    x = prepare_image(Image.open(path))                 # (3, 224, 224)
    same = np.abs(x - check.prepare(Image.open(path), {"input": settings})).max()
    say(f"\nWhat prepare_image made of {path.relative_to(ROOT).as_posix()}")
    say(f"  largest gap from the pipeline the web page uses: {same:.8f}")

    # back into colours you can look at, in the same order
    pixels = x * std[:, None, None] + mean[:, None, None]
    pixels = np.clip(pixels.transpose(1, 2, 0), 0.0, 1.0)
    Image.fromarray((pixels * 255).astype(np.uint8)).save(FIGURES / "prepare_result.png")

    width, height = original.size
    short, long = min(width, height), max(width, height)
    new_long = int(resize * long / short)
    new_size = (resize, new_long) if width <= height else (new_long, resize)
    left = int(round((new_size[0] - crop) / 2.0))
    top = int(round((new_size[1] - crop) / 2.0))

    # the original, small enough for the zip, with the crop drawn on it
    shown = original.copy()
    if shown.width > 1200:
        shown = shown.resize((1200, round(shown.height * 1200 / shown.width)), Image.BILINEAR)
    shown.save(FIGURES / "prepare_original.jpg", quality=88)
    scale = shown.width / width
    box = (round(left * scale), round(top * scale),
           round((left + crop) * scale), round((top + crop) * scale))
    marked = shown.copy()
    ImageDraw.Draw(marked).rectangle(box, outline=(255, 0, 0), width=5)
    marked.save(FIGURES / "prepare_crop_box.jpg", quality=88)

    say(f"  as downloaded            {width} x {height}")
    say(f"  shorter side to {resize}, shape kept  {new_size[0]} x {new_size[1]}")
    say(f"  centre {crop} x {crop} cut at ({left}, {top})")
    say(f"  thrown away              {left} px on the left, "
        f"{new_size[0] - left - crop} px on the right, "
        f"{top} px on top and {new_size[1] - top - crop} px at the bottom, "
        f"of the {new_size[0]} x {new_size[1]} photo")
    say(f"  kept                     {crop * crop / (new_size[0] * new_size[1]):.1%} "
        f"of the pixels the resize left")
    say("  wrote figures/prepare_original.jpg, figures/prepare_crop_box.jpg, "
        "figures/prepare_result.png")


def removal_counts():
    """Sort the removals in data/removed/log.csv into three kinds of junk."""
    rows = list(csv.DictReader(LOG.open(encoding="utf-8")))
    cross, twice, rule = [], [], []
    for row in rows:
        reason = row["reason"]
        other = None
        for word in ("same product photo as ", "same photo as ", "near copy of "):
            if word in reason:
                other = reason.split(word, 1)[1].split(",")[0].strip()
                break
        if other is None:
            rule.append(row)                       # no twin named in the reason
        elif other.split("/")[0] != row["class"]:
            cross.append(row)                      # the twin is in another class
        else:
            twice.append(row)                      # the twin is in the same class

    say("\nThe removals in data/removed/log.csv, sorted into three kinds")
    say(f"  the same photo under a different class   {len(cross):3d}")
    say(f"  a copy of a photo already in the class   {len(twice):3d}")
    say(f"  an image that breaks the rule            {len(rule):3d}")
    say(f"  all removed                              {len(rows):3d}")

    for name, group in (("different class", cross), ("same class", twice),
                        ("rule", rule)):
        with_twin = [r for r in group if "similarity" in r["reason"]]
        if with_twin:
            values = [float(r["reason"].split("similarity")[1].split(",")[0])
                      for r in with_twin]
            say(f"  of the {len(group):3d} in '{name}', {len(with_twin)} name a "
                f"similarity, {min(values):.4f} to {max(values):.4f}")
    return rows, cross, twice, rule


def junk_pictures(groups):
    """One picture of each kind of junk, taken out of data/removed/."""
    say("\nOne picture of each kind of junk")
    for name, group, wanted in (
        ("junk_1_different_class", groups[0], "china_cymbal/0009.jpg"),
        ("junk_2_copy_same_class", groups[1], "china_cymbal/0062.jpg"),
        ("junk_3_rule", groups[2], "china_cymbal/0048.jpg"),
    ):
        row = next((r for r in group if r["file"] == wanted), group[0])
        source = REMOVED / row["file"]
        target = FIGURES / f"{name}{source.suffix.lower()}"
        save_small(source, target)
        say(f"  {source.relative_to(ROOT).as_posix()} -> figures/{target.name}")
        say(f"    logged reason: {row['reason']}")


def mistake_pictures():
    """The five mistakes the exported model was most sure about."""
    model, session = web_model()
    labels = model["labels"]
    found = []
    for class_dir in sorted(d for d in NEW.iterdir() if d.is_dir()):
        for path in images_in(class_dir):
            logits = check.forward(session, check.prepare(Image.open(path), model))
            shifted = logits - logits.max()
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            guess = int(np.argmax(logits))
            if labels[guess] != class_dir.name:
                found.append((float(probabilities[guess]), class_dir.name, path,
                              labels[guess]))
    found.sort(key=lambda item: -item[0])

    say("\nThe mistakes the exported model was most sure about, on the new images")
    say("  (check.py prints the same five, but only with the file name)")
    for number, (confidence, really, path, said) in enumerate(found[:5], 1):
        target = FIGURES / f"mistake_{number}{path.suffix.lower()}"
        save_small(path, target)
        say(f"  mistake_{number}: {path.relative_to(NEW).as_posix():26s} "
            f"said {said} ({confidence:.0%}), really {really}"
            f"   -> figures/{target.name}")


def suspect_counts(removed_files):
    """How many of the first 20 suspects in each class were removed."""
    path = RESULTS / "cleaning" / "suspects_before" / "suspects.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    # The last rows of the file are the pairs of near copies, written as
    # <file>, "copy of", <twin>, similarity, "".
    pairs = [r for r in rows if r["label"] == "copy of"]
    rows = [r for r in rows if r["label"] != "copy of"]

    say(f"\nThe first 20 suspects in each class "
        f"({path.relative_to(ROOT).as_posix()})")
    say(f"  {'class':22s} {'in first 20':>11s} {'removed':>9s} {'kept':>6s}")
    total_in = total_out = 0
    for name in sorted({row["label"] for row in rows}):
        first = [r for r in rows
                 if r["label"] == name and int(r["rank_in_class"]) <= 20]
        gone = [r for r in first if r["file"] in removed_files]
        total_in += len(first)
        total_out += len(gone)
        say(f"  {name:22s} {len(first):11d} {len(gone):9d} "
            f"{len(first) - len(gone):6d}")
    say(f"  {'all five classes':22s} {total_in:11d} {total_out:9d} "
        f"{total_in - total_out:6d}")
    say(f"  suspects listed altogether         {len(rows):4d}")

    say(f"\n  pairs of near copies in the same file: {len(pairs)}")
    both = neither = one = 0
    for row in pairs:
        first, twin = row["file"], row["rank_in_class"]
        gone = [name for name in (first, twin) if name in removed_files]
        one += len(gone) == 1
        both += len(gone) == 2
        neither += not gone
        say(f"    {first} == {twin}  similarity "
            f"{float(row['probability_of_label']):.4f}  removed: "
            f"{', '.join(gone) if gone else 'neither'}")
    say(f"  of the {len(pairs)} pairs, one of the two went in {one}, both went "
        f"in {both}, neither went in {neither}")


def measure_runs(tags):
    """Measure every saved model again on the test set it was trained on."""
    say("\nEvery saved run measured again on the test set it was trained on")
    for tag in tags:
        settings = json.loads((RESULTS / f"{tag}_settings.json").read_text())
        labels = settings["class_names"]
        model = torch.load(RESULTS / f"{tag}_model.pt", map_location="cpu",
                           weights_only=False).eval()

        X, y, from_removed = [], [], 0
        for saved in settings["test_paths"]:
            path = Path(saved)
            if not path.exists():
                path = REMOVED / path.relative_to(CLEAN)   # cleaning moved it
                from_removed += 1
            X.append(prepare_image(Image.open(path)))
            y.append(labels.index(path.parent.name))

        matrix = confusion_matrix(model, np.stack(X), np.array(y), len(labels))
        right, total = int(np.trace(matrix)), int(matrix.sum())
        say(f"\n  {tag}  (settings say test accuracy "
            f"{settings['test_accuracy']:.1%}; {from_removed} of the {total} test "
            f"images now live in data/removed)")
        width = max(len(name) for name in labels) + 2
        say("   " + " " * (width + 4) + "".join(f"{n[:8]:>9s}" for n in labels))
        for i, name in enumerate(labels):
            say(f"    {name:<{width}s}" + "".join(f"{int(v):9d}" for v in matrix[i]))
        say(f"    right {right}/{total} = {right / total:.1%}")

        off = [(int(matrix[i][j]), labels[i], labels[j])
               for i in range(len(labels)) for j in range(len(labels))
               if i != j and matrix[i][j] > 0]
        off.sort(reverse=True)
        say("    most mixed up: " + ", ".join(
            f"{true} called {said} {count} times" for count, true, said in off[:3]))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows, cross, twice, rule = removal_counts()
    removed_files = {row["file"] for row in rows}

    copy_run_pictures(["run", "scratch", "frozen", "custom", "clean"])
    prepare_picture()
    junk_pictures((cross, twice, rule))
    mistake_pictures()
    suspect_counts(removed_files)
    measure_runs(["run", "scratch", "frozen", "custom", "clean"])

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()