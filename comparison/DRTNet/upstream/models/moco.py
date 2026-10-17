import torch
import torch.nn as nn
import torch.nn.functional as F


class Moco(nn.Module):
    """Contrast-enhanced spectra recovery (CESR).

    The paper defines fused HSI spectra as anchors, real HSI spectra as
    positives, and MSI spectra as negatives. Positives/negatives are split by
    the average spectral similarity between anchor spectra and initial HSI
    spectra, then optimized with a contrastive loss.
    """

    def __init__(
        self,
        base_encoder=None,
        image_lr=None,
        image_hr=None,
        dim=128,
        K=32 * 128,
        m=0.999,
        T=0.07,
        max_samples=None,
        eps=1e-8,
    ):
        super(Moco, self).__init__()
        self.T = T
        self.max_samples = K if max_samples is None else max_samples
        self.eps = eps

    def _flatten_spectra(self, x):
        # [B, C, H, W] -> [B*H*W, C]
        return x.permute(0, 2, 3, 1).reshape(-1, x.size(1))

    def _msi_to_hsi_bands(self, image_hr, n_bands):
        """Lift uniformly selected MSI bands back to an HSI-length sequence."""
        b, m, h, w = image_hr.shape
        spectra = image_hr.permute(0, 2, 3, 1).reshape(-1, 1, m)
        spectra = F.interpolate(spectra, size=n_bands, mode='linear', align_corners=True)
        spectra = spectra.reshape(b, h, w, n_bands).permute(0, 3, 1, 2)
        return spectra

    def _sample_rows(self, *tensors):
        n = tensors[0].size(0)
        if n <= self.max_samples:
            return tensors
        index = torch.randperm(n, device=tensors[0].device)[:self.max_samples]
        return tuple(t[index] for t in tensors)

    def _split_positive_negative(self, anchor, hsi, msi):
        # Similarity is implemented as reciprocal Euclidean distance so that
        # larger scores are more similar, matching the paper's threshold rule.
        distance = torch.norm(anchor.detach() - hsi.detach(), p=2, dim=1)
        similarity = 1.0 / (distance + self.eps)
        threshold = similarity.mean()
        pos_mask = similarity >= threshold
        neg_mask = similarity < threshold

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            order = torch.argsort(similarity, descending=True)
            split = max(1, order.numel() // 2)
            pos_index = order[:split]
            neg_index = order[split:]
            if neg_index.numel() == 0:
                neg_index = order[-1:]
            positives = hsi[pos_index]
            negatives = msi[neg_index]
        else:
            positives = hsi[pos_mask]
            negatives = msi[neg_mask]
        return positives, negatives

    def forward(self, fused_hsi, reference_hsi, image_hr):
        n_bands = fused_hsi.size(1)
        negative_hsi = self._msi_to_hsi_bands(image_hr, n_bands)

        anchor = self._flatten_spectra(fused_hsi)
        hsi = self._flatten_spectra(reference_hsi)
        msi = self._flatten_spectra(negative_hsi)
        anchor, hsi, msi = self._sample_rows(anchor, hsi, msi)

        positives, negatives = self._split_positive_negative(anchor, hsi, msi)
        anchor = F.normalize(anchor, dim=1)
        positives = F.normalize(positives, dim=1)
        negatives = F.normalize(negatives, dim=1)

        logits_pos = torch.matmul(anchor, positives.t()) / self.T
        logits_neg = torch.matmul(anchor, negatives.t()) / self.T
        numerator = torch.logsumexp(logits_pos, dim=1)
        denominator = torch.logsumexp(torch.cat([logits_pos, logits_neg], dim=1), dim=1)
        return -(numerator - denominator).mean()
