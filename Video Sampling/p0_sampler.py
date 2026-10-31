'''video_sampler - extract frames from a video.'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def _get_video_info(cap: cv2.VideoCapture, video_name: str) -> tuple[float, int, float]:
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0

    print(f"视频信息: {video_name}")
    print(f"  帧率:     {src_fps:.2f} fps")
    print(f"  总帧数:   {total_frames}")
    print(f"  时长:     {duration:.1f}s ({duration/60:.1f}min)")
    print(f"  分辨率:   {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return src_fps, total_frames, duration


def sample_frames(
    video_path: Path,
    output_dir: Path,
    *,
    interval_sec: float | None = None,
    total_count: int | None = None,
    fmt: str = "jpg",
    quality: int = 95,
    dry_run: bool = False,
) -> int:
    '''从视频抽帧。'''
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    src_fps, total_frames, duration = _get_video_info(cap, video_path.name)

    if total_count is not None:
        frame_interval = max(1, total_frames / total_count)
        mode_desc = f"均匀抽 {total_count} 张 — 每隔 {frame_interval:.1f} 原始帧抽一帧"
    else:
        sec = interval_sec or 1.0
        frame_interval = max(1, src_fps * sec)
        mode_desc = f"每 {sec}s 抽一帧 — 每隔 {frame_interval:.1f} 原始帧"

    est_count = max(1, int(total_frames / frame_interval))
    print(f"  抽帧模式: {mode_desc}")
    print(f"  预计抽出: ~{est_count} 张")
    print()

    if dry_run:
        cap.release()
        return est_count

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    count = 0
    next_frame = 0.0

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if int(next_frame) >= total_frames:
            break

        current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if current_idx >= round(next_frame):
            out_path = output_dir / f"{stem}_{count:06d}.{fmt}"
            params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt in ("jpg", "jpeg") else []
            cv2.imwrite(str(out_path), frame, params)
            count += 1
            next_frame += frame_interval

    cap.release()
    print(f"完成: 共抽出 {count} 帧 → {output_dir}")
    return count


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts"}


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def sample_folder(
    folder: Path,
    output_root: Path,
    *,
    interval_sec: float | None = None,
    total_count: int | None = None,
    fmt: str = "jpg",
    quality: int = 95,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict[str, int]:
    '''批量处理文件夹内所有视频，每个视频输出到一个子目录。'''
    videos = sorted(
        p for p in folder.iterdir() if is_video_file(p)
    )
    if not videos:
        print(f"文件夹内未找到视频文件: {folder}")
        return {}

    print(f"找到 {len(videos)} 个视频:")
    for v in videos:
        print(f"  - {v.name}")

    results: dict[str, int] = {}
    skipped: list[str] = []
    for i, vp in enumerate(videos, 1):
        sub_dir = output_root / f"{vp.stem}_frames"

        if skip_existing and sub_dir.is_dir() and any(sub_dir.iterdir()):
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(videos)}] 跳过: {vp.name} (已有输出目录)")
            print(f"{'=' * 60}")
            skipped.append(vp.name)
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(videos)}] 处理: {vp.name}")
        print(f"{'=' * 60}")
        try:
            n = sample_frames(
                vp, sub_dir,
                interval_sec=interval_sec,
                total_count=total_count,
                fmt=fmt,
                quality=quality,
                dry_run=dry_run,
            )
            results[vp.name] = n
        except Exception as e:
            print(f"  !! 跳过: {e}")
            results[vp.name] = -1

    print(f"\n{'=' * 60}")
    print("批量处理完成:")
    if skipped:
        print(f"  已跳过 {len(skipped)} 个 (已有输出):")
        for name in skipped:
            print(f"    - {name}")
    total = 0
    for name, n in results.items():
        status = f"{n} 张" if n >= 0 else "失败"
        print(f"  {name}: {status}")
        if n > 0:
            total += n
    print(f"  本次抽出: {total} 张")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="video_sampler — 视频抽帧工具"
    )
    parser.add_argument(
        "video", type=Path, nargs="?", default=None,
        help="输入视频路径（单文件模式）",
    )
    parser.add_argument(
        "-d", "--dir", type=Path, default=None,
        help="批量模式：输入文件夹，处理其中所有视频",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="输出目录（默认在视频同目录下创建 {视频名}_frames/）",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-i", "--interval", type=float, default=None,
        help="时间间隔模式：每隔 N 秒抽一帧（默认 1 秒）",
    )
    mode.add_argument(
        "-n", "--count", type=int, default=None,
        help="总数模式：均匀抽取 N 张（e.g. -n 100）",
    )

    parser.add_argument(
        "--fmt", choices=("jpg", "png"), default="jpg",
        help="输出格式，默认 jpg",
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=95,
        help="jpg 质量 1-100，默认 95",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览视频信息和预计抽帧数，不实际执行",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="强制重新处理所有视频，即使已有输出目录",
    )
    args = parser.parse_args()

    interval = args.interval
    count = args.count
    if interval is None and count is None:
        interval = 1.0

    if args.dir:
        folder = args.dir
        if not folder.is_dir():
            print(f"错误: 文件夹不存在 — {folder}", file=sys.stderr)
            sys.exit(1)
        output_root = args.output or (folder / "_sampled_frames")
        sample_folder(
            folder, output_root,
            interval_sec=interval,
            total_count=count,
            fmt=args.fmt,
            quality=args.quality,
            dry_run=args.dry_run,
            skip_existing=not args.force,
        )
        return

    if args.video is None:
        parser.print_help()
        sys.exit(1)

    if not args.video.exists():
        print(f"错误: 视频文件不存在 — {args.video}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or (args.video.parent / f"{args.video.stem}_frames")
    sample_frames(
        args.video, output_dir,
        interval_sec=interval,
        total_count=count,
        fmt=args.fmt,
        quality=args.quality,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
