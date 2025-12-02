from pathlib import Path
from typing import List, Optional
import subprocess

from PIL import Image


class ExeGanGuidedRecovery:
    """
    Wrapper around EXE-GAN's guided_recovery.py.

    Cloud-side assumptions:
      * All inputs are already preprocessed locally.
      * test_img, mask_img, exemplar_img are already (sample_size x sample_size).
      * No face detection / cropping / resizing is done here.

    Inputs (all PIL.Image.Image):
      - test_img:      face image to be edited (goes to images/target)
      - mask_img:      mask image (white = region to inpaint, goes to images/mask)
      - exemplar_img:  guidance face image (goes to images/exemplar)

    Behavior:
      - Saves these 3 images into the folders EXE-GAN expects.
      - Calls guided_recovery.py with those dirs.
      - Loads generated images from recover_out and returns them as PIL images.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        psp_ckpt_rel: str = "pre-train/psp_ffhq_encode.pt",
        exegan_ckpt_rel: str = "checkpoint/EXE_GAN_model.pt",
        sample_size: int = 256,
    ):
        """
        :param repo_root: Path to EXE-GAN repo root. If None, assumes this file is
                          located at <repo_root>/service/exegan_service.py>.
        :param psp_ckpt_rel: Path to pSp checkpoint relative to repo_root.
        :param exegan_ckpt_rel: Path to EXE-GAN checkpoint relative to repo_root.
        :param sample_size: Image size EXE-GAN expects (default 256).
        """
        if repo_root is None:
            # <repo_root>/service/exegan_service.py → parents[1] is repo_root
            self.repo_root = Path(__file__).resolve().parents[1]
        else:
            self.repo_root = Path(repo_root).resolve()

        self.psp_ckpt = self.repo_root / psp_ckpt_rel
        self.exegan_ckpt = self.repo_root / exegan_ckpt_rel

        # New cleaned structure: images/mask, images/target, images/exemplar
        self.mask_dir = self.repo_root / "images" / "mask"          # --masked_dir
        self.gt_dir = self.repo_root / "images" / "target"          # --gt_dir
        self.exemplar_dir = self.repo_root / "images" / "exemplar"  # --exemplar_dir
        self.out_dir = self.repo_root / "recover_out"               # --eval_dir

        self.sample_size = sample_size

        for d in (self.mask_dir, self.gt_dir, self.exemplar_dir, self.out_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------------- internal helpers ---------------- #

    def _clear_io_dirs(self) -> None:
        """Remove old images from input/output dirs to avoid collisions."""
        for d in (self.mask_dir, self.gt_dir, self.exemplar_dir, self.out_dir):
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()

    def _save_triplet(
        self,
        test_img: Image.Image,
        mask_img: Image.Image,
        exemplar_img: Image.Image,
        index: int = 0,
    ) -> None:
        """
        Save inputs to the correct folders with a consistent filename (<index>.png).

        Mapping:
          test_img     -> images/target    (image to edit)
          mask_img     -> images/mask      (mask)
          exemplar_img -> images/exemplar  (guidance)

        Assumes all images are already (sample_size x sample_size).
        """
        expected = (self.sample_size, self.sample_size)

        if test_img.size != expected:
            raise ValueError(
                f"test_img must be {expected}, got {test_img.size}. "
                "Do face cropping/resizing on the client before calling EXE-GAN."
            )
        if mask_img.size != expected:
            raise ValueError(
                f"mask_img must be {expected}, got {mask_img.size}. "
                "Do mask cropping/resizing on the client before calling EXE-GAN."
            )
        if exemplar_img.size != expected:
            raise ValueError(
                f"exemplar_img must be {expected}, got {exemplar_img.size}. "
                "Do exemplar cropping/resizing on the client before calling EXE-GAN."
            )

        # No resizing here – just save as-is
        test_img.save(self.gt_dir / f"{index}.png")
        mask_img.save(self.mask_dir / f"{index}.png")
        exemplar_img.save(self.exemplar_dir / f"{index}.png")

    def _run_script(self, sample_times: int = 1) -> None:
        """Call guided_recovery.py with our image directories."""
        cmd = [
            "python",
            "guided_recovery.py",
            "--psp_checkpoint_path",
            str(self.psp_ckpt),
            "--ckpt",
            str(self.exegan_ckpt),
            "--masked_dir",
            str(self.mask_dir),
            "--gt_dir",
            str(self.gt_dir),
            "--exemplar_dir",
            str(self.exemplar_dir),
            "--sample_times",
            str(sample_times),
            "--eval_dir",
            str(self.out_dir),
        ]
        subprocess.run(cmd, cwd=str(self.repo_root), check=True)

    def _load_outputs(self, index: int = 0, sample_times: int = 1) -> List[Image.Image]:
        """
        EXE-GAN names outputs like: <name>_<jj>_inpaint.png.

        With our convention:
          name = index
        so outputs are: '0_0_inpaint.png', '0_1_inpaint.png', ...
        """
        imgs: List[Image.Image] = []
        for j in range(sample_times):
            fname = f"{index}_{j}_inpaint.png"
            path = self.out_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Missing EXE-GAN output: {path}")
            imgs.append(Image.open(path).convert("RGB"))
        return imgs

    # ---------------- public API ---------------- #

    def run(
        self,
        test_img: Image.Image,
        mask_img: Image.Image,
        exemplar_img: Image.Image,
        sample_times: int = 1,
    ) -> List[Image.Image]:
        """
        Main callable.

        :param test_img:      PIL image, preprocessed face to be edited (→ images/target).
        :param mask_img:      PIL image, preprocessed mask (→ images/mask).
        :param exemplar_img:  PIL image, preprocessed guidance face (→ images/exemplar).
        :param sample_times:  How many stochastic samples EXE-GAN should generate.
        :return: list of reconstructed images (length = sample_times).
        """
        self._clear_io_dirs()
        self._save_triplet(test_img, mask_img, exemplar_img, index=0)
        self._run_script(sample_times=sample_times)
        return self._load_outputs(index=0, sample_times=sample_times)
