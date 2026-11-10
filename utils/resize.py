import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def get_session_dirs(base_dir):
    """Return set of relative session paths (date/time) that exist."""
    base_dir = Path(base_dir)
    sessions = set()
    if not base_dir.exists():
        return sessions
    for date_folder in base_dir.iterdir():
        if not date_folder.is_dir():
            continue
        for time_folder in date_folder.iterdir():
            if time_folder.is_dir():
                sessions.add(time_folder.relative_to(base_dir))
    return sessions


def resize_missing_sessions(src_dir, dst_dir, size=(224, 224)):
    """
    Only resize sessions that exist in src_dir but are missing (or incomplete) in dst_dir.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_sessions = get_session_dirs(src_dir)
    dst_sessions = get_session_dirs(dst_dir)

    missing = src_sessions - dst_sessions
    print(f"Source sessions: {len(src_sessions)}")
    print(f"Destination sessions: {len(dst_sessions)}")
    print(f"Missing sessions to resize: {len(missing)}")

    if not missing:
        print("All sessions already exist. Nothing to do.")
        return

    image_extensions = {".jpg", ".jpeg", ".png"}

    for rel_session in tqdm(sorted(missing), desc="Resizing missing sessions"):
        src_session = src_dir / rel_session
        dst_session = dst_dir / rel_session

        # Copy/resize all images under this session
        for src_path in src_session.rglob("*"):
            if src_path.suffix.lower() not in image_extensions:
                continue
            rel = src_path.relative_to(src_session)
            dst_path = dst_session / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if dst_path.exists():
                continue

            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.save(dst_path, quality=95)
            except Exception as e:
                print(f"Failed {src_path}: {e}")

    print(f"Done. Missing sessions resized into {dst_dir}")


def resize_images_in_directory(src_dir, dst_dir, size=(224, 224), overwrite=False):
    """Original full resize (kept for compatibility)."""
    resize_missing_sessions(src_dir, dst_dir, size)
