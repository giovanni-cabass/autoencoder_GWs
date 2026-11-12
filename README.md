# autoencoder_GWs

[![CI](https://github.com/giovanni-cabass/autoencoder_GWs/actions/workflows/ci.yml/badge.svg)](https://github.com/giovanni-cabass/autoencoder_GWs/actions/workflows/ci.yml)

Metric-preserving (distance-preserving) autoencoders, demonstrated first on a toy 3D cylinder and
then on real gravitational-wave template banks.

**Paper:** [arXiv:2511.10466](https://arxiv.org/abs/2511.10466), "Generating optimal
Gravitational-Wave template banks with metric-preserving autoencoders" (Cabass, Wadekar,
Zaldarriaga, Zhou).

## Development note

This repository's packaging (README, CI) was built with help from Claude Code. The autoencoder
research behind the paper was built with help from Claude Code and Cursor.

(Earlier commits were missing the git trailer so GitHub never recognized Claude Code as co-author.)

## What this does (plain language)

A gravitational-wave search matches an incoming signal against a large bank of template
waveforms. Comparing waveforms directly is expensive, so it helps to compress each one down to a
handful of coordinates first — but a naive compression can distort distances: two waveforms that
are actually far apart can land oddly close together in the compressed coordinates (or vice
versa), which breaks any later reasoning about how many templates are needed to cover the bank
without gaps.

The autoencoders here are trained to avoid that distortion, on top of the usual reconstruction
loss — but the two examples in this repo enforce it two different ways:

- **Cylinder toy example** (`autoencoder_training_example/`): a *local* condition. At each encoded
  point, the decoder's Jacobian $J$ is computed (`torch.func.jacrev` + `vmap`, one point at a
  time, batched), and the loss penalizes $J^{\mathsf T}J - I$ — pushing the decoder toward a local
  isometry.
- **Gravitational-wave autoencoder** (`autoencoder_use_example/`): a *global* condition. The loss
  penalizes the mismatch between pairwise distances among encoded points in a batch and their
  pairwise distances in the original, high-dimensional space, weighted so nearby pairs matter more
  than far-apart ones.

Either way, the result is a low-dimensional latent space where distance between encoded points is
a reliable stand-in for distance between the original, high-dimensional waveforms.

## What's inside

- `autoencoder_training_example/` — toy problem: a 3→2→3 autoencoder trained so the induced
  metric on a cylinder stays flat everywhere.
  - `train_cylinder.py` — laptop-sized (CPU, ~30s, 20 epochs) version of the training run, the
    entry point to actually run.
  - `cylinder_train_data.pt`, `cylinder_valid_data.pt` — the training/validation data, shared by
    `train_cylinder.py` and the script below.
  - `paper_training_run/` — the real, as-run paper training (8 GPUs, 10000 epochs) and its real
    checkpoint, kept for provenance; see its own README.
- `autoencoder_use_example/` — the real use case: a 10→2→10 autoencoder for gravitational-wave
  template banks (sub-bank 0 of bank 1; see [gwIAS-HM](https://github.com/JayWadekar/gwIAS-HM) for
  more on these template banks).
  - `demo_autoencoder_b_1_sb_0.py` — loads the trained sub-bank-0 autoencoder and a small, already
    truncated sample of waveforms, and reports reconstruction error and median phase overlap.
  - `autoencoder_b_1_sb_0.pth`, `norm_factor_b_1_sb_0.txt`, and the matching raw sample files — the
    data the demo needs.
  - `paper_training_run/` — the training script for all three sub-banks, plus the orphaned
    sub-bank 1/2 checkpoints, kept for provenance; see its own README.

## Running it

```bash
pip install -r requirements.txt

cd autoencoder_training_example && python train_cylinder.py
cd ../autoencoder_use_example && python demo_autoencoder_b_1_sb_0.py
```

Both scripts are CPU-friendly; `train_cylinder.py` uses a GPU automatically if one is available.

## License

MIT, see `LICENSE`.
