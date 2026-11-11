# `paper_training_run/`

Historical training script and orphaned checkpoints for sub-banks 1 and 2 — kept here for
provenance, not for use. The live GW demo is
[`../demo_autoencoder_b_1_sb_0.py`](../demo_autoencoder_b_1_sb_0.py).

## What's in this folder

- `train_autoencoders_b_1_sb_0_to_2.py` — trains autoencoders for all three sub-banks (0, 1, 2).
- `autoencoder_b_1_sb_1.pth`, `autoencoder_b_1_sb_2.pth` — real trained weights for sub-banks 1
  and 2.
- `norm_factor_b_1_sb_1.txt`, `norm_factor_b_1_sb_2.txt` — their matching normalization factors.

## Why the sub-bank 1/2 weights are orphaned

These are real trained weights, but no test/input data for sub-banks 1 or 2 was ever published in
this repo or its source handoff — there is no `ca_bank_1_subbank_1-...`/`_2-...` or
`phases_bank_1_subbank_1-...`/`_2-...` file anywhere. They are not demoable here regardless of
anything else in this repo; kept purely for provenance, the same reasoning as keeping
`cylinder_riemann.pth` in `autoencoder_training_example/paper_training_run/`.

## Why this script doesn't run here or in CI

`train_autoencoders_b_1_sb_0_to_2.py` needs 3 GPUs and the full, unsuffixed bank files
(`ca_bank_1_subbank_{0,1,2}.txt`, `phases_bank_1_subbank_{0,1,2}.txt`, unsuffixed `Vph_...`) for
all three sub-banks — none of which are published anywhere in this repo or its archives. Not
runnable here regardless of GPU access.

## Where the published raw files (one level up) came from

The three `.tar.gz` archives previously in `autoencoder_use_example/` have been replaced by their
already-unpacked raw files (same bytes, no unpacking needed):

| Archive (removed) | Size | Sub-bank scope |
|---|---|---|
| `svd_data_for_training_and_testing_b_1_sb_0.tar.gz` | 0.2 MB | Sub-bank 0 only |
| `pth_files_for_autoencoder_b_1_sb_0_to_2.tar.gz` | 0.4 MB | All three (0, 1, 2) |
| `phases_data_for_testing_b_1_sb_0.tar.gz` | 9.5 MB | Sub-bank 0 only |

Only `pth_files_for_autoencoder_b_1_sb_0_to_2.tar.gz` touched sub-banks 1/2 at all — via the
orphaned weights above. The other two archives were already scoped to sub-bank 0 only.
