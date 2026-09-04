from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog


VALID_STATES = {"visible", "not_visible"}


def choose_export_folder() -> Path | None:
    app = QApplication.instance()
    owns_app = app is None

    if owns_app:
        app = QApplication(sys.argv)

    selected = QFileDialog.getExistingDirectory(
        None,
        "Choose LabelForge Facemap export folder",
        str(Path.home()),
    )

    if not selected:
        return None

    return Path(selected)


def detect_keypoints(fieldnames: list[str]) -> list[str]:
    suffix = "_state"
    return [
        name[:-len(suffix)]
        for name in fieldnames
        if name.endswith(suffix)
    ]


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    export_root = choose_export_folder()

    if export_root is None:
        print("No folder selected.")
        return

    frames_dir = export_root / "Frames"
    labels_csv = export_root / "Labels" / "labels.csv"
    manifest_path = export_root / "facemap_export.json"

    print("\n=== LabelForge Facemap Export Check ===")
    print(f"Export: {export_root}\n")

    if not frames_dir.is_dir():
        fail("Frames/ folder is missing.")

    if not labels_csv.is_file():
        fail("Labels/labels.csv is missing.")

    with labels_csv.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    keypoints = detect_keypoints(fieldnames)

    if not keypoints:
        fail("No keypoints could be detected from *_state columns.")

    frame_files = sorted(
        path for path in frames_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    print(f"✓ {len(frame_files)} frame file(s) found")
    print(f"✓ {len(rows)} CSV row(s) found")
    print(f"✓ {len(keypoints)} keypoint(s) detected")
    print("  " + ", ".join(keypoints))

    if len(frame_files) != len(rows):
        fail(
            f"Frame/CSV mismatch: {len(frame_files)} image(s), "
            f"{len(rows)} CSV row(s)."
        )

    required_base = {"png_path", "image", "image_folder"}

    missing_base = required_base.difference(fieldnames)

    if missing_base:
        fail(
            "Missing required CSV column(s): "
            + ", ".join(sorted(missing_base))
        )

    for kp in keypoints:
        required = {
            f"{kp}_x",
            f"{kp}_y",
            f"{kp}_visible",
            f"{kp}_state",
        }
        missing = required.difference(fieldnames)

        if missing:
            fail(
                f"{kp}: missing column(s): "
                + ", ".join(sorted(missing))
            )

    visible_count = 0
    not_visible_count = 0
    usable_frame_count = 0
    image_shapes: set[tuple[int, int]] = set()

    # Build the same kind of per-frame target array the custom trainer needs:
    # shape = (num_keypoints, 2), with NaN for unavailable keypoints.
    trainer_targets: list[np.ndarray] = []

    for row_index, row in enumerate(rows, start=1):
        csv_path = Path(row["png_path"])

        # Export CSV should point to the copied Frames/ images.
        if not csv_path.is_file():
            fallback = frames_dir / row.get("image", "")
            if fallback.is_file():
                csv_path = fallback
            else:
                fail(
                    f"Row {row_index}: image path does not exist:\n"
                    f"{row.get('png_path', '')}"
                )

        try:
            with Image.open(csv_path) as image:
                image.verify()

            with Image.open(csv_path) as image:
                width, height = image.size
                image_shapes.add((width, height))
        except Exception as exc:
            fail(
                f"Row {row_index}: image cannot be opened: "
                f"{csv_path}\n{exc}"
            )

        target = np.full(
            (len(keypoints), 2),
            np.nan,
            dtype=np.float32,
        )

        frame_has_visible_point = False

        for kp_index, kp in enumerate(keypoints):
            state = row.get(f"{kp}_state", "").strip()
            visible = row.get(f"{kp}_visible", "").strip()
            x_text = row.get(f"{kp}_x", "").strip()
            y_text = row.get(f"{kp}_y", "").strip()

            if state not in VALID_STATES:
                fail(
                    f"Row {row_index}, {kp}: invalid or unfinished "
                    f"state '{state}'."
                )

            if state == "visible":
                if visible not in {"1", "1.0", "true", "True"}:
                    fail(
                        f"Row {row_index}, {kp}: state=visible but "
                        f"visible flag is '{visible}'."
                    )

                if not x_text or not y_text:
                    fail(
                        f"Row {row_index}, {kp}: visible point has "
                        "missing x/y coordinates."
                    )

                try:
                    x = float(x_text)
                    y = float(y_text)
                except ValueError:
                    fail(
                        f"Row {row_index}, {kp}: x/y are not numeric."
                    )

                if not (0 <= x < width and 0 <= y < height):
                    fail(
                        f"Row {row_index}, {kp}: coordinate "
                        f"({x:.2f}, {y:.2f}) lies outside "
                        f"{width}×{height} image."
                    )

                target[kp_index] = (x, y)
                visible_count += 1
                frame_has_visible_point = True

            else:
                if visible not in {"0", "0.0", "false", "False", ""}:
                    fail(
                        f"Row {row_index}, {kp}: state=not_visible but "
                        f"visible flag is '{visible}'."
                    )

                # This is the critical training behavior:
                # unavailable keypoint -> NaN only for this keypoint.
                if x_text or y_text:
                    fail(
                        f"Row {row_index}, {kp}: not_visible point "
                        "should have blank x/y coordinates."
                    )

                not_visible_count += 1

        trainer_targets.append(target)

        if frame_has_visible_point:
            usable_frame_count += 1

    if usable_frame_count == 0:
        fail(
            "No frame contains a visible keypoint. The custom Facemap "
            "trainer would have nothing usable to train on."
        )

    print("✓ all exported image paths resolve")
    print("✓ all image files can be opened")
    print("✓ all label states are valid")
    print("✓ visible coordinates are numeric and inside the image")
    print("✓ not-visible keypoints convert cleanly to NaN targets")
    print(
        f"✓ trainer-style target tensor built: "
        f"{len(trainer_targets)} frame(s) × {len(keypoints)} keypoint(s)"
    )
    print(f"✓ {usable_frame_count}/{len(rows)} frame(s) usable for training")
    print(f"  visible labels: {visible_count}")
    print(f"  not visible labels: {not_visible_count}")

    if manifest_path.is_file():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest_kps = manifest.get("keypoints", [])

            if manifest_kps and manifest_kps != keypoints:
                fail(
                    "facemap_export.json keypoint order does not match "
                    "labels.csv."
                )

            print("✓ facemap_export.json matches the CSV keypoint order")
        except json.JSONDecodeError:
            fail("facemap_export.json is not valid JSON.")
    else:
        print("! facemap_export.json not found (CSV check still passed)")

    if len(image_shapes) > 1:
        print(
            "! Note: exported frames have multiple image sizes: "
            + ", ".join(
                f"{w}×{h}" for w, h in sorted(image_shapes)
            )
        )
    else:
        width, height = next(iter(image_shapes))
        print(f"✓ image size: {width}×{height}")

    print("\nFACEMAP EXPORT LOOKS GOOD 🔥")
    print(
        "\nThis check validates the dataset at the same input boundary "
        "used by the custom Facemap training workflow. It does NOT run "
        "a training epoch or create/modify a model."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nFACEMAP EXPORT CHECK FAILED")
        print(str(exc))
        raise SystemExit(1)
