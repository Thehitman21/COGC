"""Command-line inference for Controllable Optimizable Gamma Correction."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from compressed_final_model_mean import LogExpEnhancement


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Profile:
    gamma_range: tuple[float, float]
    residual: bool | None
    use_bm3d: bool
    mode: str


# Dataset/task configurations reported in Table 1 of the BMVC 2026 paper.
PROFILES: dict[str, Profile] = {
    "auto": Profile((0.3, 1.0), None, False, "auto"),
    "lolv2": Profile((0.3, 1.0), True, True, "brighten"),
    "lsrw-huawei": Profile((0.3, 1.0), True, True, "brighten"),
    "lsrw-nikon": Profile((0.3, 1.0), False, True, "brighten"),
    "lom-under-severe": Profile((0.15, 1.0), True, True, "brighten"),
    "lom-under-buu": Profile((0.3, 1.0), True, True, "brighten"),
    "lom-over": Profile((3.0, 3.5), False, True, "darken"),
    "unpaired": Profile((0.3, 1.0), False, False, "brighten"),
    "darkface": Profile((0.3, 1.0), True, False, "brighten"),
    "mfnet": Profile((0.7, 1.0), False, False, "brighten"),
}


def parse_switch(value: str | None, fallback: bool | None) -> bool | None:
    if value is None or value == "profile":
        return fallback
    if value == "auto":
        return None
    return value == "on"


def enhance_image(
    input_path: Path,
    output_path: Path,
    iterations: int,
    learning_rate: float,
    seed: int,
    gamma_range: tuple[float, float],
    residual: bool | None,
    use_bm3d: bool,
    mode: str,
    bm3d_sigma: float,
) -> None:
    """Enhance one image and save the result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enhancer = LogExpEnhancement(
        str(input_path),
        gt_path=None,
        exp_path=None,
        random_number=seed,
        filter=use_bm3d,
        quality="BAD",
        c=1,
        param_range=gamma_range,
        residual=residual,
        mode=mode,
        bm3d_sigma=bm3d_sigma,
    )
    enhancer.optimize_brightness(
        num_iterations=iterations,
        learning_rate=learning_rate,
    )
    enhancer.get_enhanced_image().save(output_path)


def collect_images(input_path: Path) -> list[Path]:
    """Return one image or supported images directly inside a directory."""
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Enhance under/over-exposed images with COGC"
    )
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("output", type=Path, help="Output image or directory")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="auto",
        help="Paper configuration to use (default: auto)",
    )
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--seed",
        type=int,
        default=6272,
        help="Gamma-MLP initialization seed used by the released experiments",
    )
    parser.add_argument("--gamma-min", type=float, default=None)
    parser.add_argument("--gamma-max", type=float, default=None)
    parser.add_argument(
        "--residual",
        choices=("profile", "auto", "on", "off"),
        default="profile",
        help="Override the profile residual setting",
    )
    parser.add_argument(
        "--bm3d",
        choices=("profile", "on", "off"),
        default="profile",
        help="Override the profile BM3D setting",
    )
    parser.add_argument("--bm3d-sigma", type=float, default=20.0)
    parser.add_argument(
        "--mode",
        choices=("profile", "auto", "brighten", "darken"),
        default="profile",
        help="Override the profile correction direction",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = PROFILES[args.profile]

    gamma_min = profile.gamma_range[0] if args.gamma_min is None else args.gamma_min
    gamma_max = profile.gamma_range[1] if args.gamma_max is None else args.gamma_max
    if not 0 < gamma_min <= gamma_max:
        raise ValueError("Gamma values must satisfy 0 < gamma_min <= gamma_max")

    residual = parse_switch(args.residual, profile.residual)
    use_bm3d = parse_switch(args.bm3d, profile.use_bm3d)
    mode = profile.mode if args.mode == "profile" else args.mode

    images = collect_images(args.input)
    if not images:
        raise RuntimeError(f"No supported images found in: {args.input}")

    if args.input.is_file():
        enhance_image(
            args.input,
            args.output,
            args.iterations,
            args.learning_rate,
            args.seed,
            (gamma_min, gamma_max),
            residual,
            bool(use_bm3d),
            mode,
            args.bm3d_sigma,
        )
        print(f"Saved: {args.output}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        output_path = args.output / image_path.name
        enhance_image(
            image_path,
            output_path,
            args.iterations,
            args.learning_rate,
            args.seed,
            (gamma_min, gamma_max),
            residual,
            bool(use_bm3d),
            mode,
            args.bm3d_sigma,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
