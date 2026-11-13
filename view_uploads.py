from __future__ import annotations

from pathlib import Path

VIEW_UPLOAD_FIELDS = (
    ("isometric", "isometric_images", 0),
    ("top", "top_images", 1),
    ("face", "face_images", 2),
)


def unique_view_filename(view: str, original_name: str, used_names: set[str]) -> str:
    """Return a collision-free filename while retaining the original extension."""
    output_name = original_name
    if output_name.casefold() in used_names:
        stem = Path(original_name).stem or "cutter_image"
        suffix = Path(original_name).suffix
        output_name = f"{view}_{original_name}"
        counter = 2
        while output_name.casefold() in used_names:
            output_name = f"{view}_{stem}_{counter}{suffix}"
            counter += 1

    used_names.add(output_name.casefold())
    return output_name
