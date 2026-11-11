# MoG-DCN (IEEE TIP 2021)

Dong et al., *Model-Guided Deep Hyperspectral Image Super-Resolution*,
[DOI: 10.1109/TIP.2021.3078058](https://doi.org/10.1109/TIP.2021.3078058).
[Author code](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution/tree/35eef37dbef797739212cc3b65f747a432a28892/sf_8_CAVE),
specifically `network_31.py` as imported by `train.py`.

This independent PyTorch implementation follows the active CAVE path: four
unfolding stages, a shared spatial U-Net used twice per stage, dense reuse of
the encoder features, and learned spatial and spectral measurement operators.
The original code fixes 31 HSI bands, three RGB bands and ×8. This adapter
parameterizes HSI/MSI bands and uses stride 4 in the two learned 13×13 spatial
operators. It retains their dense channel mixing and their residual signs.
The shared PSF and SRF still generate the input observations; they do not
replace the model's learned internal operators. No author code or checkpoints
are shipped.

Training retains L1 and Adam (LR 1e-4, weight decay 1e-8, constant LR).
Use the public benchmark split and retrain per dataset. Inference uses a
fixed checkpoint. See the [shared registered protocol](../registered_fusion/README.md)
for commands, resume, outputs and smoke checks.

For a large validation/test discrepancy, audit both splits with one fixed
checkpoint. This read-only probe uses the official metrics; the optional tile
probe is diagnostic and does not replace full-patch benchmark inference.

```bash
python comparison/MoG-DCN/diagnose_splits.py --dataset Houston13 --device cuda:0 \
  --check_tiled --output_json /tmp/mog_houston_diagnosis.json
```

Use `--checkpoint` and `--validation_log` if training used custom paths. Check
`recomputed_minus_logged` first, then compare the absolute versus relative
error, spectral magnitude contribution, and learned operator traces.
