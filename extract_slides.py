import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

try:
    import img2pdf

    HAS_IMG2PDF = True
except ImportError:
    HAS_IMG2PDF = False


def check_ffmpeg() -> str:
    """Проверяет наличие FFmpeg в системе и возвращает путь к нему."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin

    common_win_paths = [
        Path(sys.prefix) / "Scripts" / "ffmpeg.exe",
        Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    ]
    for p in common_win_paths:
        if p.exists():
            return str(p)

    sys.exit(
        "Ошибка: FFmpeg не найден в системе!\n"
        "Установите его:\n"
        "  - Windows: winget install Gyan.FFmpeg\n"
        "  - macOS: brew install ffmpeg\n"
        "  - Linux: sudo apt install ffmpeg"
    )


def get_vfr_arg(ffmpeg_bin: str) -> list[str]:
    """Выбирает -fps_mode для FFmpeg 5.1+ или legacy -vsync для старых версий (до 5.0)."""
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
        for old_v in ("version 2.", "version 3.", "version 4."):
            if old_v in first_line:
                return ["-vsync", "vfr"]
    except (subprocess.SubprocessError, OSError):
        pass

    return ["-fps_mode", "vfr"]


def extract_slides_from_video(
    video_path: Path,
    output_dir: Path,
    ffmpeg_bin: str,
    scene_threshold: float = 0.025,
    min_interval: float = 1.5,
    use_gpu: bool = True,
) -> bool:
    """Извлекает слайды из одного видеофайла."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.glob("slide_*.jpg"):
        old_file.unlink()

    vf_filter = (
        f"select='isnan(prev_selected_t)+gt(scene,{scene_threshold})*"
        f"gt(t-prev_selected_t,{min_interval})'"
    )

    vfr_option = get_vfr_arg(ffmpeg_bin)

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
    ]

    if use_gpu:
        cmd += ["-hwaccel", "cuda"]

    cmd += [
        "-skip_frame",
        "nokey",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        vf_filter,
        *vfr_option,
        "-q:v",
        "2",
        str(output_dir / "slide_%04d.jpg"),
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0 and use_gpu:
        print("  [INFO] Ошибка декодирования через CUDA. Переключаюсь на CPU...")
        return extract_slides_from_video(
            video_path,
            output_dir,
            ffmpeg_bin,
            scene_threshold,
            min_interval,
            use_gpu=False,
        )

    if result.returncode != 0:
        return False

    slides = sorted(output_dir.glob("slide_*.jpg"))
    if not slides:
        print("  [!] Слайды не найдены (попробуйте уменьшить --threshold).")
        return False

    pdf_path = output_dir / "presentation.pdf"
    save_pdf(slides, pdf_path)
    create_contact_sheet(slides, output_dir / "contact_sheet.jpg")

    print(f"  [OK] Сохранено слайдов: {len(slides)} -> {pdf_path.name}")
    return True


def save_pdf(image_paths: list[Path], output_path: Path) -> None:
    """Сборка изображений в PDF."""
    if not image_paths:
        return

    if HAS_IMG2PDF:
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert([str(p) for p in image_paths]))
    else:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        try:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                resolution=150,
            )
        finally:
            for img in images:
                img.close()


def create_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    """Генерация листа предпросмотра (сетка слайдов)."""
    if not image_paths:
        return

    cols = 4
    thumb_w, thumb_h = 320, 180
    label_h = 24
    rows = (len(image_paths) + cols - 1) // cols

    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, path in enumerate(image_paths):
        with Image.open(path) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail((thumb_w - 8, thumb_h - 8))

        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)

        sheet.paste(thumb, (x + 4, y + 4))
        draw.text((x + 6, y + thumb_h + 2), path.name, fill="black")

    sheet.save(output_path, quality=85)


def process_tree(
    root_dir: Path,
    ffmpeg_bin: str,
    threshold: float,
    min_interval: float,
    force: bool,
    use_gpu: bool,
) -> None:
    """Рекурсивный поиск и обработка всех .mp4 внутри переданной директории."""
    videos = sorted(p for p in root_dir.rglob("*.mp4") if "презентация" not in p.parts)

    if not videos:
        print(f"В папке '{root_dir}' видеофайлы .mp4 не найдены.")
        return

    videos_by_dir: dict[Path, list[Path]] = {}
    for v in videos:
        videos_by_dir.setdefault(v.parent, []).append(v)

    processed = skipped = failed = 0
    print(f"Найдено видео для обработки: {len(videos)}")

    for parent_dir, dir_videos in videos_by_dir.items():
        multiple = len(dir_videos) > 1

        for video in dir_videos:
            out_dir = (
                parent_dir / "презентация" / video.stem
                if multiple
                else parent_dir / "презентация"
            )
            pdf_path = out_dir / "presentation.pdf"

            if not force and pdf_path.exists() and pdf_path.stat().st_size > 0:
                print(f"[SKIP] {video.name} — PDF уже существует")
                skipped += 1
                continue

            print(f"\n[PROCESS] {video.relative_to(root_dir)}")
            ok = extract_slides_from_video(
                video, out_dir, ffmpeg_bin, threshold, min_interval, use_gpu
            )

            if ok and pdf_path.exists() and pdf_path.stat().st_size > 0:
                processed += 1
            else:
                failed += 1

    print(
        f"\nЗавершено. Успешно: {processed} | Пропущено: {skipped} | Ошибок: {failed}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Автоматическое извлечение презентаций из видеозаписей лекций в PDF."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Путь к видеофайлу (.mp4) или к папке с лекциями (по умолчанию: текущая папка)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Директория сохранения (применяется только при обработке одного видеофайла)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.025,
        help="Порог чувствительности смены слайда от 0.01 до 0.1 (по умолчанию: 0.025)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.5,
        help="Минимальное время между соседними слайдами в секундах (по умолчанию: 1.5)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Принудительно использовать только процессор (без CUDA)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Перезаписывать существующие presentation.pdf",
    )

    args = parser.parse_args()
    ffmpeg_bin = check_ffmpeg()
    target_path = Path(args.target).resolve()

    if not target_path.exists():
        sys.exit(f"Ошибка: указанный путь не существует: {target_path}")

    use_gpu = not args.cpu

    if target_path.is_file():
        out_dir = (
            Path(args.output) if args.output else target_path.parent / "презентация"
        )
        print(f"Обработка файла: {target_path.name}")
        extract_slides_from_video(
            target_path, out_dir, ffmpeg_bin, args.threshold, args.interval, use_gpu
        )
    else:
        print(f"Обработка директории: {target_path}")
        process_tree(
            target_path, ffmpeg_bin, args.threshold, args.interval, args.force, use_gpu
        )


if __name__ == "__main__":
    main()
