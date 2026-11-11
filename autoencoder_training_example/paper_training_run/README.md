# `paper_training_run/`

Historical training artifacts for the cylinder autoencoder — the actual run that produced the
checkpoint shipped with this repo, kept here for provenance. The runnable, laptop-sized version of
this same method lives one level up, at [`../train_cylinder.py`](../train_cylinder.py).

## What's in this folder

- `autoencoder_training_example_appendix_B.py` — the full training script, as actually run.
- `submit.autoencoder_training_example_appendix_B.sh` — its SLURM launcher.
- `cylinder_riemann.pth` — the real trained checkpoint.
- `cylinder_riemann.train_valid_losses.png` — its training/validation loss curve.

## Data

Trained on `../cylinder_train_data.pt` and `../cylinder_valid_data.pt` (not duplicated into this
subfolder — both scripts in this repo share the same two data files).

## Validation numbers

Best validation loss 0.00977 at epoch 9518; reconstruction MSE on the validation set
1.46e-3.

## Cluster launch configuration

1 node, 8×H200 GPUs, `torch_cuda_pyro` conda environment, DDP, 10000 epochs. No early stopping —
the full schedule always runs, and the checkpoint is only saved when the validation loss improves
(best-val-only checkpointing). See
[`ddp-training-harness`](https://github.com/giovanni-cabass/ddp-training-harness) for why that
pattern is used; not re-explained here.

## Do not run this here

This script needs the 8-GPU cluster/conda setup above. It is not reproducible in this repo or in
CI.

## Code-quality note

This script predates `../train_cylinder.py` by a lot and computes the decoder Jacobian with
`torch.autograd.functional.jacobian` over the whole batch, then indexes out the diagonal —
wasteful (it builds and discards zero cross-sample blocks), though harmless on 8 GPUs. It's the
same loss and the same network as `../train_cylinder.py`, shown here as it was actually run, for
provenance — not as a model for how to write this today.
