# ESFS (IEEE TGRS 2025)

Xu et al., *Enhanced Spatial-Frequency Synergistic Network for Multispectral
and Hyperspectral Image Fusion*,
[DOI: 10.1109/TGRS.2025.3589097](https://doi.org/10.1109/TGRS.2025.3589097).
[Author code](https://github.com/ziqianmo/ESFS/tree/38bd5b373438d2d89cadeb3ae58f4a67f6fffbda),
specifically `model/ESFS.py` and `main.py`.

This independent PyTorch implementation follows the released ×4 path: bicubic
HSI stem and separate MSI stem, 180 feature channels, six spatial–frequency
blocks with alternating shifted 8×8 attention windows and six heads, and
three passes through the shared body with dense feature aggregation. The
frequency path retains amplitude and phase processing and the spatial path
retains condensed keys and values. Modules unused by the released forward
pass are omitted. HSI/MSI channel counts are set from the shared loader.
Only standard PyTorch operators are needed; no author code or checkpoints
are shipped.

Training retains L1 and Adam (LR 1e-4, weight decay 0), with the released
epoch milestones 100/150/175/190/195 and gamma 0.5. The total training budget
and validation interval are those of the shared benchmark. Inference uses a
fixed checkpoint. See the [shared registered protocol](../registered_fusion/README.md)
for commands, resume, outputs and smoke checks.
