# IKONOS SRF provenance

`ikonos_relative_spectral_response.csv` is a text conversion of the numerical
`ikonos_sp` array distributed by the open-source HySure project:

- source repository: https://github.com/alfaiate/HySure
- source file: `data/ikonos_spec_resp.mat`
- source blob SHA: `6d60df334687adaaafd826cd215ecfb17f789bc4`
- array layout documented by HySure: wavelength, pan, blue, green, red, NIR
- wavelength sampling: 350-1035 nm in 5 nm increments

For the default PaviaU comparison protocol, the four multispectral channels are
IKONOS Blue, Green, Red and NIR. The legacy WV2 six-band option is kept as
`wv2_visible6` for backward-compatible experiments.

The default IKONOS path uses the same nominal 103-band 430-860 nm PaviaU grid
as the current UFGNet experiment setup. The old `PaviaU.txt` file is retained
unchanged for legacy WV2 experiments.
