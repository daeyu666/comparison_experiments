# IR&ArF adapter for comparison_experiments

Upstream implementation: https://github.com/Jiahuiqu/IR-ArF  
Paper: J. Qu et al., "IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images," IEEE TIP, 2025. DOI: 10.1109/TIP.2025.3551531.

This adapter retains the upstream model blocks and three-stage fusion/denoising
topology. Engineering adaptations:
- Dynamic HSI band count (upstream Pavia setting was fixed at 102).
- Dynamic MSI band count (upstream implementation used 4 MSI channels).
- Read the actual LR-HSI spatial size instead of hard-coding 20 pixels.
- Create coordinate tensors on the active input device rather than a fixed CUDA device.
- Retain the upstream L1 + weighted NCC training objective in the benchmark trainer.

The shared `data_loader.build_datasets` remains authoritative for scene splits,
normalization, spatial degradation, SRF integration and HSI-side deformation.
The upstream fixed-shape dataloader is not reused.

Initial target: PaviaU (103 HSI bands, 4 MSI bands), x4 scale, shared 64x64 HR
patch and 16x16 LR-HSI. Run `python comparison/IR-ArF/smoke_test.py` before
training. The adapter is not yet validated on the remaining datasets.
