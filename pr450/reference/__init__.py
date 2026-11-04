"""
Reference-signal strategies for the passive radar (plan Phase 2d).

  raw           -- the received reference channel as-is (noise + multipath included)
  demod_remod   -- equalise, hard-decide every RE, re-modulate a clean copy
  pilot_only    -- only the a-priori known REs (sync + reference signals)
  perfect       -- the transmitted signal (cooperative tap / upper bound)

The pilot-only reference is a *matched-filter* reference only: it cannot be
used for ECA cancellation, because the direct path contains the full signal
and a pilot-only dictionary cannot represent the data REs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..waveform.base import Waveform, ofdm_demodulate, qam_hard_decision


def perfect(wf: Waveform) -> np.ndarray:
    return wf.iq


def pilot_only(wf: Waveform) -> np.ndarray:
    return wf.synthesize(mask=wf.known_mask())


@dataclass
class RemodStats:
    symbol_error_rate: float
    missed_re_fraction: float
    false_re_fraction: float
    channel_estimate: np.ndarray


def estimate_static_channel(y: np.ndarray, wf: Waveform, channel_taps: int | None = None) -> np.ndarray:
    """Per-subcarrier channel from every known RE in the record, via a delay-domain LS fit.

    The channel impulse response is constrained to `channel_taps` samples
    (default: the short CP length), which is the physically meaningful OFDM
    channel model and interpolates subcarriers that carry few or no pilots
    without the bias a frequency-domain smoother introduces.
    """
    num = wf.num
    known = wf.known_mask()
    rows, cols = np.nonzero(known)
    h_ls = y[rows, cols] / wf.grid[rows, cols]
    counts = np.bincount(cols, minlength=num.n_sc).astype(float)
    h_sum = np.bincount(cols, weights=h_ls.real, minlength=num.n_sc) + 1j * np.bincount(
        cols, weights=h_ls.imag, minlength=num.n_sc
    )
    have = counts > 0
    h_avg = np.zeros(num.n_sc, dtype=complex)
    h_avg[have] = h_sum[have] / counts[have]

    taps = channel_taps or num.cp_short
    lags = np.arange(-16, taps)  # pre-cursor taps absorb band-limited tails of sub-sample delays
    basis = np.exp(-2j * np.pi * np.outer(num.subcarrier_offsets(), lags) / num.nfft)
    w = np.sqrt(counts)[:, None]
    coef, *_ = np.linalg.lstsq(basis[have] * w[have], h_avg[have] * w[have, 0], rcond=None)
    return basis @ coef


def demod_remod(
    rx: np.ndarray,
    wf: Waveform,
    mod_source: str = "oracle",
    block_noise_factor: float = 10.0,
    re_noise_factor: float = 30.0,
    min_re_energy: float = 0.004,
    channel_taps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, RemodStats]:
    """Reconstruct a clean reference from the received reference channel.

    Assumes `rx` is time-aligned with `wf` (direct path at zero delay) and the
    reference channel is static over the record (see estimate_static_channel).

    Reference fidelity requirement: any reconstruction error leaks into the
    ECA output at (error energy / signal energy) x surveillance DNR. With a
    direct path ~50 dB above noise, errors must stay well below -50 dB, so a
    handful of falsely "active" REs already matters.

    mod_source="oracle" takes each RE's modulation order from the generator.
    A real receiver would have to decode the DCI or classify modulation
    blindly per PRB; "blind" is reserved for that and not implemented yet.
    """
    if mod_source != "oracle":
        raise NotImplementedError("blind modulation classification is not implemented yet")

    num = wf.num
    y = ofdm_demodulate(rx, num, wf.sym_starts, wf.cp_lens)
    known = wf.known_mask()

    h = estimate_static_channel(y, wf, channel_taps)

    z = y / h[None, :]
    noise_var = float(np.mean(np.abs(z[known] - wf.grid[known]) ** 2))

    # Activity detection. A PRB x symbol block is "on" if its candidate REs
    # carry energy well above the noise (a block test, because single REs are
    # unreliable: inner high-order QAM points are weak). Inside an "on" block
    # an RE must also clear a per-RE noise test, since blocks can straddle a
    # channel edge (e.g. a PRB half covered by PBCH / SSB, half unscheduled).
    # Structurally empty REs (known from the cell configuration) are never on.
    # The per-RE test limits remod to inner-point SNR > re_noise_factor, i.e.
    # ~31 dB reference SNR for 256QAM -- ample for a reference Yagi.
    # Both tests also have an absolute floor tied to the constellation: the
    # weakest legitimate point (256QAM inner, energy 2/170 = 0.0118) less ~5 dB.
    # A noise-relative threshold alone is not enough: with a 12-bit ADC at idle
    # load, quantisation spreads non-Gaussian energy into empty REs (~35-47 x the
    # pilot-measured noise variance), and 2-4 false REs per CPI raised the ECA
    # residual from 0.9 to 3.3 dB above thermal.
    energy = np.abs(z) ** 2
    candidate = ~known
    if wf.structural_empty is not None:
        candidate &= ~wf.structural_empty
    n_rb = num.n_rb
    n_cand = candidate.reshape(-1, n_rb, 12).sum(axis=2)
    block_e = np.where(candidate, energy, 0).reshape(-1, n_rb, 12).sum(axis=2) / np.maximum(n_cand, 1)
    block_on = np.repeat((n_cand > 0) & (block_e > max(block_noise_factor * noise_var, min_re_energy / 12)), 12, axis=1)
    active = block_on & candidate & (energy > max(re_noise_factor * noise_var, min_re_energy))
    grid_hat = np.zeros_like(wf.grid)
    grid_hat[known] = wf.grid[known]
    for bits in np.unique(wf.mod_bits[active]):
        sel = active & (wf.mod_bits == bits)
        grid_hat[sel] = qam_hard_decision(z[sel], int(bits) if bits > 0 else 2)

    truly_active = (wf.re_type != 0) & ~known
    n_unknown = max(1, int(np.count_nonzero(~known)))
    both = active & truly_active
    errors = np.count_nonzero(np.abs(grid_hat[both] - wf.grid[both]) > 1e-6)
    stats = RemodStats(
        symbol_error_rate=errors / max(1, int(np.count_nonzero(both))),
        missed_re_fraction=np.count_nonzero(truly_active & ~active) / n_unknown,
        false_re_fraction=np.count_nonzero(active & ~truly_active) / n_unknown,
        channel_estimate=h,
    )
    return wf.synthesize(grid=grid_hat), grid_hat, stats


__all__ = ["perfect", "pilot_only", "demod_remod", "estimate_static_channel", "RemodStats"]
