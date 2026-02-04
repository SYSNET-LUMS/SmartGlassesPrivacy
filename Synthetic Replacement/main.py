#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import json
import cv2
import shutil
import subprocess
from typing import Optional
from shutil import which

# ---------------- paths next to main.py ----------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTH_IMG_DIR = os.path.join(THIS_DIR, "synthetic_faces")
SYNTH_LM_DIR  = os.path.join(THIS_DIR, "synthetic_landmarks")
OUTPUT_ROOT   = THIS_DIR  
MFS_SCRIPT    = os.path.join(THIS_DIR, "MobileFaceSwap", "video_test.py")
VCOLORS_JSON  = os.path.join(THIS_DIR, "video_colors.json")


if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
try:
    import warping
except Exception as e:
    print("❌ Could not import warping.py from:", THIS_DIR)
    print("   Error:", e)
    sys.exit(1)

# ---------------- helpers ----------------
def _coerce_rgb_tuple(val):
    def _ints_ok(seq):
        try:
            nums = [int(x) for x in seq]
        except Exception:
            return None
        if len(nums) != 3:
            return None
        for v in nums:
            if v < 0 or v > 255:
                return None
        return tuple(nums)
    if isinstance(val, (list, tuple)):
        t = _ints_ok(val)
        if t is None: raise ValueError(f"Invalid RGB triplet: {val}")
        return t
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            s = s[1:-1]
        parts = [x.strip() for x in s.split(",")]
        t = _coerce_rgb_tuple(parts) if len(parts) == 3 else None
        if t is not None: return t
        import re as _re
        nums = _re.findall(r"-?\d+", val)
        t = _ints_ok(nums)
        if t is None: raise ValueError(f"Invalid RGB string: {val}")
        return t
    raise ValueError(f"Unsupported color value type: {type(val)}")

def load_video_color_map_if_any(path: str) -> dict:
    if not os.path.isfile(path):
        print("ℹ️ No video_colors.json found next to main.py; will estimate color from video.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {k: _coerce_rgb_tuple(v) for k, v in raw.items()}
    print("Resolved VIDEO_COLORS (name: (R,G,B)):", out)
    return out

def estimate_color_from_video_bgr(video_path: str, samples: int = 30) -> tuple:
    """Estimate a BGR color by sampling a few frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"❌ OpenCV could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, (total // samples) if total > 0 else 1)
    accum = [0.0, 0.0, 0.0]
    count = 0
    while True and count < samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        small = cv2.resize(frame, (64, 64))
        mean_bgr = cv2.mean(small)[:3]
        accum[0] += mean_bgr[0]; accum[1] += mean_bgr[1]; accum[2] += mean_bgr[2]
        count += 1
        # skip ahead to sample across the clip
        for _ in range(step - 1):
            _ret, _ = cap.read()
            if not _ret:
                break
    cap.release()
    if count == 0:
        raise SystemExit("❌ Could not sample frames to estimate color.")
    avg = (int(accum[0]/count), int(accum[1]/count), int(accum[2]/count))
    print(f"🎯 Estimated target BGR from video: {avg}")
    return avg

def find_output_video(output_root: str, video_name: str) -> Optional[str]:
    exts = (".mp4", ".avi", ".mov", ".mkv")
    candidates = []
    # prefer exact <video_name>.ext
    for ext in exts:
        p = os.path.join(output_root, f"{video_name}{ext}")
        if os.path.isfile(p):
            candidates.append(p)
    # fallback: any file starting with video_name
    if not candidates:
        for path in glob.glob(os.path.join(output_root, f"{video_name}*")):
            if os.path.splitext(path)[1].lower() in exts:
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def find_saved_best_image(output_root: str) -> Optional[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    for d in (os.path.join(output_root, "best_match"), os.path.join(os.getcwd(), "best_match")):
        if not os.path.isdir(d):
            continue
        candidates = []
        for ext in exts:
            candidates += glob.glob(os.path.join(d, f"best_*{ext}"))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return None

def run_mobilefaceswap(target_video_path: str, source_img_path: str, output_root: str):
    """Run MobileFaceSwap from its own folder with staged local filenames."""
    if not os.path.isfile(MFS_SCRIPT):
        raise SystemExit(f"❌ MobileFaceSwap script not found: {MFS_SCRIPT}")
    script_dir = os.path.dirname(MFS_SCRIPT)
    ckpt = os.path.join(script_dir, "checkpoints", "arcface.pdparams")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"❌ Missing checkpoints: {ckpt}")

    # stage short names inside MobileFaceSwap directory
    local_target = os.path.join(script_dir, "_mfs_target.mp4")
    local_source = os.path.join(script_dir, "_mfs_source.jpg")
    try:
        shutil.copyfile(target_video_path, local_target)
        shutil.copyfile(source_img_path, local_source)
    except Exception as e:
        raise SystemExit(f"❌ Failed to stage inputs for MobileFaceSwap: {e}")

    cmd = [
        sys.executable, MFS_SCRIPT,
        "--target_video_path", "_mfs_target.mp4",
        "--source_img_path",   "_mfs_source.jpg",
        "--output_path",       output_root,
    ]
    print("▶ Running MobileFaceSwap:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=script_dir)
    if res.returncode != 0:
        raise SystemExit("❌ MobileFaceSwap failed (see log above).")
    print("✅ Post-swap done.")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Warp a video using local synthetic assets; optionally run MobileFaceSwap."
    )
    ap.add_argument("--video", required=True, type=str, help="Path to the input video file.")
    ap.add_argument("--landmarks_root", required=True, type=str, help="Root dir containing per-video landmarks.")
    ap.add_argument("--after_swap", action="store_true", help="Run MobileFaceSwap after warping.")
    ap.add_argument("--source_img_path", type=str,
                    help="(Optional) override: specific face image for MobileFaceSwap instead of best_match.")

    args = ap.parse_args()

    # Validate core inputs
    if not os.path.isfile(args.video):
        ap.error(f"--video path does not exist or is not a file: {args.video}")
    if not os.path.isdir(args.landmarks_root):
        ap.error(f"--landmarks_root does not exist or is not a directory: {args.landmarks_root}")
    if not os.path.isdir(SYNTH_IMG_DIR):
        ap.error(f"Missing ./synthetic_faces next to main.py: {SYNTH_IMG_DIR}")
    if not os.path.isdir(SYNTH_LM_DIR):
        ap.error(f"Missing ./synthetic_landmarks next to main.py: {SYNTH_LM_DIR}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Resolve video name and color
    video_path = os.path.abspath(args.video)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    color_map = load_video_color_map_if_any(VCOLORS_JSON)
    if video_name in color_map:
        rgb = color_map[video_name]
        target_bgr = (rgb[2], rgb[1], rgb[0])  # convert RGB -> BGR for OpenCV pipeline
        print(f"🎨 Using color from video_colors.json for '{video_name}': RGB={rgb} → BGR={target_bgr}")
    else:
        target_bgr = estimate_color_from_video_bgr(video_path)

    # Run warping
    swapper = getattr(warping, "FaceSwapper", None) or getattr(warping, "BatchFaceSwapper", None)
    if swapper is None:
        raise SystemExit("❌ Could not find FaceSwapper/BatchFaceSwapper in warping.py")
    swapper = swapper(SYNTH_IMG_DIR, SYNTH_LM_DIR)

    print(f"▶ Processing single video: {video_name}")
    swapper.process_video(video_path, args.landmarks_root, OUTPUT_ROOT, target_bgr)

    # Locate outputs
    warped_video = find_output_video(OUTPUT_ROOT, video_name)
    if not warped_video:
        raise SystemExit(f"❌ Could not locate warped output video for '{video_name}' in {OUTPUT_ROOT}")

    # Post-swap (optional)
    if args.after_swap:
        if args.source_img_path and os.path.isfile(args.source_img_path):
            source_img = args.source_img_path
        else:
            source_img = find_saved_best_image(OUTPUT_ROOT)
            if not source_img:
                raise SystemExit("❌ Couldn't find a saved best-match image in 'best_match/'. "
                                 "Provide --source_img_path or ensure warping saved best_* there.")
        print(f"Post-swap on: {os.path.basename(warped_video)} with source {os.path.basename(source_img)}")
        run_mobilefaceswap(warped_video, source_img, OUTPUT_ROOT)

if __name__ == "__main__":
    main()
