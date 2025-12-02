from pathlib import Path
from PIL import Image

from exegan_service import ExeGanGuidedRecovery


def main():
    # repo_root = EXE-GAN folder
    repo_root = Path(__file__).resolve().parents[1]

    examples_dir = repo_root / "service" / "examples"

    test_path = examples_dir / "test.png"
    mask_path = examples_dir / "mask.png"
    ex_path   = examples_dir / "exemplar.png"

    if not test_path.exists() or not mask_path.exists() or not ex_path.exists():
        raise FileNotFoundError(
            f"Missing example images in {examples_dir}. "
            f"Expected test.png, mask.png, exemplar.png"
        )

    # EXE-GAN expects 256x256, so for this local test we enforce it here
    size = (256, 256)

    test_img = Image.open(test_path).convert("RGB").resize(size)
    # Mask should be single-channel (L); keep NEAREST if you want it crisp
    mask_img = Image.open(mask_path).convert("L").resize(size)
    ex_img   = Image.open(ex_path).convert("RGB").resize(size)

    service = ExeGanGuidedRecovery(repo_root=repo_root)

    print("Running EXE-GAN guided recovery...")
    outputs = service.run(test_img, mask_img, ex_img, sample_times=1)

    out_path = examples_dir / "reconstructed.png"
    outputs[0].save(out_path)
    print("Saved reconstructed image to:", out_path)


if __name__ == "__main__":
    main()
