### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 0a450000-0000-4000-8000-000000000002
begin
	ENV["GKSwstype"] = "nul"  # headless GR backend (safe inside Pluto's worker on Windows)
	using FFTW
	using LinearAlgebra
	using Plots
	using PlutoUI
	using Random
	using Statistics
end

# ╔═╡ 0a450000-0000-4000-8000-000000000001
md"""
# LTE450 downlink waveform, step by step

This notebook builds an **LTE FDD downlink** signal for a **5 MHz carrier in the 450 MHz band** (Band 31 or 72) from first principles, following 3GPP **TS 36.211**. It is the Julia twin of `pr450/waveform/lte.py` in this repository.

What we build, in the order the steps follow:

1. **Numerology** — how many subcarriers, what sample rate, how long each symbol and cyclic prefix is.
2. **Gold sequence** — the pseudo-random bit generator that scrambles almost every LTE reference signal.
3. **PSS** — the primary synchronisation signal (Zadoff-Chu sequence).
4. **SSS** — the secondary synchronisation signal (m-sequences); together with the PSS it carries the cell ID.
5. **CRS** — the cell-specific reference signals, LTE's always-on pilots.
6. **Resource grid** — place everything for one 10 ms frame, plus random control and data payload.
7. **OFDM modulation** — turn the grid into time-domain IQ samples.
8. **Look at it** — resource-element map, power over time, spectrum.
9. **Verify it** — against spec arithmetic and against srsRAN 4G's own generator.
10. **Save it** — an IQ file for Python, GNU Radio or MATLAB.

Exactly specified signals (PSS, SSS, CRS) are generated **bit-exact**. Payloads (PBCH, control, data) are **random QAM on the correct resource elements**. That is all a passive radar needs, and it is what a real cell looks like statistically.

Change the controls in Step 0 and every later cell updates.
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000003
md"""
## Step 0 — Controls

* **Cell ID** (PCI, 0–503): selects the PSS root, the SSS pattern, the CRS frequency shift and the CRS scrambling.
* **CRS antenna ports**: 1 or 2 (450 MHz macro sites are commonly 2T).
* **Load**: how busy the cell is. *Idle* leaves only the always-on signals; *full* fills every free resource element.

Cell ID: $(@bind cell_id Slider(0:503, default = 1, show_value = true))

CRS ports: $(@bind n_ports Select([1, 2], default = 2))

Load: $(@bind load_name Select(["idle", "low", "medium", "full"], default = "medium"))
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000004
md"""
## Step 1 — Numerology (TS 36.211 §6.2, §6.12)

LTE uses **15 kHz subcarrier spacing**. A 5 MHz carrier has **25 resource blocks (PRB)** of 12 subcarriers each, so **300 active subcarriers** (4.5 MHz). The rest of the 5 MHz is guard band.

The IFFT size must exceed 300, and the standard choice for 5 MHz is **512**. The sample rate is then 512 × 15 kHz = **7.68 Msps**, exactly a quarter of LTE's 30.72 Msps reference rate.

Time structure:

* A **slot** is 0.5 ms and holds **7 OFDM symbols** (normal cyclic prefix).
* Each symbol is 512 samples (66.7 µs) plus a **cyclic prefix (CP)**: 40 samples for the first symbol of a slot, 36 for the other six. (These are 160 and 144 samples at 30.72 Msps, divided by 4.)
* A **subframe** (1 ms, the scheduling unit or TTI) is 2 slots = 14 symbols; a **frame** (10 ms) is 10 subframes = 140 symbols.

LTE leaves the **DC subcarrier empty**: the 300 subcarriers sit at offsets −150…−1 and +1…+150 (× 15 kHz) around the carrier centre. (NR, by contrast, uses DC.)
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000005
begin
	const N_RB      = 25            # resource blocks in 5 MHz
	const N_SC      = 12 * N_RB     # 300 active subcarriers
	const NFFT      = 512           # IFFT size
	const SCS       = 15e3          # subcarrier spacing [Hz]
	const FS        = NFFT * SCS    # 7.68e6 samples/s
	const CP_LONG   = 40            # CP of symbol 0 in each slot [samples]
	const CP_SHORT  = 36            # CP of symbols 1..6 [samples]
	const N_RB_MAX  = 110           # CRS sequences are defined for the largest bandwidth
	const FC        = 463.5e6       # assumed carrier (Band 72: DL 461-466 MHz)
end

# ╔═╡ 0a450000-0000-4000-8000-000000000006
begin
	"CP length of every symbol (0-based symbol l: long CP when l is the first of its slot)."
	cp_lengths(nsym) = [mod(l, 7) == 0 ? CP_LONG : CP_SHORT for l in 0:nsym-1]

	"Frequency offset (in units of 15 kHz) of grid subcarrier k = 0..299, skipping DC."
	subcarrier_offsets() = [k < N_SC ÷ 2 ? k - N_SC ÷ 2 : k - N_SC ÷ 2 + 1 for k in 0:N_SC-1]

	"1-based IFFT bin of a frequency offset (negative offsets wrap to the top half)."
	fft_bin(offset) = mod(offset, NFFT) + 1
end

# ╔═╡ 0a450000-0000-4000-8000-000000000007
let
	slot = CP_LONG + 6CP_SHORT + 7NFFT
	offs = subcarrier_offsets()
	md"""
	**Check:** one slot = $(slot) samples = $(round(slot / FS * 1e3, digits = 3)) ms; one frame = $(20slot) samples.
	Subcarrier offsets run from $(first(offs)) to $(last(offs)); offset 0 (DC) is present: $(0 in offs).
	"""
end

# ╔═╡ 0a450000-0000-4000-8000-000000000008
md"""
## Step 2 — Gold sequence (TS 36.211 §7.2)

Almost every LTE reference signal is scrambled with the same length-31 **Gold sequence**. It is two linear-feedback shift registers XOR-ed together:

* ``x_1(n+31) = x_1(n+3) \oplus x_1(n)``, always started as ``x_1(0)=1``, the rest 0
* ``x_2(n+31) = x_2(n+3) \oplus x_2(n+2) \oplus x_2(n+1) \oplus x_2(n)``, started from the 31 bits of ``c_{init}``
* output ``c(n) = x_1(n + N_c) \oplus x_2(n + N_c)`` with ``N_c = 1600`` (the first 1600 outputs are discarded)

The initial value ``c_{init}`` is what makes each signal unique: it encodes the cell ID, the slot number and the symbol. Two bits ``c(2m), c(2m+1)`` then form one QPSK symbol ``\big((1-2c(2m)) + j(1-2c(2m+1))\big)/\sqrt{2}``.

Julia note: arrays are 1-based, so ``x(n)`` in the spec is `x[n+1]` here.
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000009
begin
	function gold_sequence(c_init::Integer, len::Integer)
		Nc = 1600
		total = Nc + len
		x1 = zeros(UInt8, total + 31)
		x2 = zeros(UInt8, total + 31)
		x1[1] = 1
		for i in 0:30
			x2[i+1] = (c_init >> i) & 1
		end
		for j in 1:total                     # j = n + 1
			x1[j+31] = x1[j+3] ⊻ x1[j]
			x2[j+31] = x2[j+3] ⊻ x2[j+2] ⊻ x2[j+1] ⊻ x2[j]
		end
		return x1[Nc+1:Nc+len] .⊻ x2[Nc+1:Nc+len]
	end

	"QPSK symbols from consecutive bit pairs."
	qpsk(c) = [complex(1 - 2.0c[2m-1], 1 - 2.0c[2m]) / sqrt(2) for m in 1:length(c)÷2]
end

# ╔═╡ 0a450000-0000-4000-8000-000000000010
md"""
First bits for ``c_{init} = 12345``: `$(join(gold_sequence(12345, 24)))`. These match py3gpp's `nrPRBS` (the same generator is used by NR); the Python package checks this in its self-test.
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000011
md"""
## Step 3 — Primary synchronisation signal, PSS (TS 36.211 §6.11.1)

The PSS is the first thing a UE (or our passive radar) looks for. It is a **Zadoff-Chu sequence** of length 63 with the middle element (the DC position) punctured, leaving 62 values. The root ``u`` depends on ``N_{ID}^{(2)} = \text{PCI} \bmod 3``: 25, 29 or 34.

```math
d_u(n) = \begin{cases} e^{-j\pi u n(n+1)/63} & n = 0,\dots,30 \\ e^{-j\pi u (n+1)(n+2)/63} & n = 31,\dots,61 \end{cases}
```

Zadoff-Chu sequences have **constant amplitude** and **ideal periodic autocorrelation**, which makes them easy to detect even with a frequency offset.

**Precision trap.** ``\pi u n(n+1)/63`` reaches about 6600 rad. In single precision (float32) that loses about 5·10⁻⁴ rad, and srsRAN 4G does exactly that (our cross-check found it). Here we reduce ``u\,n(n+1) \bmod 126`` **in integers first**, so the phase is exact.

**Placement:** the last symbol (l = 6) of slot 0 in subframes **0 and 5**, on the 62 subcarriers around DC (5 empty guard subcarriers on each side).
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000012
function lte_pss(nid2::Integer)
	u = (25, 29, 34)[nid2+1]
	d = Vector{ComplexF64}(undef, 62)
	for n in 0:61
		m = n < 31 ? n : n + 1
		d[n+1] = cispi(-((u * m * (m + 1)) % 126) / 63)   # exp(-jπ·(integer phase)/63)
	end
	return d
end

# ╔═╡ 0a450000-0000-4000-8000-000000000013
let
	p = [lte_pss(i) for i in 0:2]
	auto = abs(dot(p[1], p[1])) / 62
	cross = maximum(abs(dot(p[i], p[j])) / 62 for i in 1:3, j in 1:3 if i != j)
	plot(real.(p[cell_id % 3 + 1]), imag.(p[cell_id % 3 + 1]); seriestype = :scatter, aspect_ratio = 1,
		 title = "PSS of N_ID2 = $(cell_id % 3): all 62 values on the unit circle", xlabel = "I", ylabel = "Q",
		 label = "|cross-corr| between roots ≤ $(round(cross, digits = 2)) (auto = $(round(auto, digits = 2)))",
		 size = (520, 420), legend = :bottom)
end

# ╔═╡ 0a450000-0000-4000-8000-000000000014
md"""
## Step 4 — Secondary synchronisation signal, SSS (TS 36.211 §6.11.2)

The SSS carries the other part of the cell ID, ``N_{ID}^{(1)} = \lfloor \text{PCI}/3 \rfloor`` (0–167). It is built from three length-31 **m-sequences** (``\tilde s``, ``\tilde c``, ``\tilde z``), cyclically shifted and interleaved:

* ``\tilde s`` is shifted by ``m_0`` and ``m_1``, derived from ``N_{ID}^{(1)}``
* ``\tilde c`` is shifted by ``N_{ID}^{(2)}``: this ties the SSS to the PSS
* ``\tilde z`` is shifted by ``m_0 \bmod 8`` or ``m_1 \bmod 8``

The even and odd elements swap roles between **subframe 0 and subframe 5**. That is how a receiver tells which half of the frame it is in, and so finds the 10 ms frame boundary.

The SSS is BPSK (±1) and sits one symbol before the PSS (l = 5), on the same 62 subcarriers.
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000015
begin
	"Length-31 m-sequence (±1) with recursion x(i+5) = Σ x(i+t) mod 2 over `taps`, x(0..4) = 0,0,0,0,1."
	function mseq31(taps)
		x = zeros(Int, 31)
		x[5] = 1
		for i in 0:25
			x[i+6] = mod(sum(x[i+1+t] for t in taps), 2)
		end
		return 1 .- 2 .* x
	end

	function lte_sss(nid1::Integer, nid2::Integer)
		s̃ = mseq31((2, 0)); c̃ = mseq31((3, 0)); z̃ = mseq31((4, 2, 1, 0))
		qp = nid1 ÷ 30
		q  = (nid1 + qp * (qp + 1) ÷ 2) ÷ 30
		mp = nid1 + q * (q + 1) ÷ 2
		m0 = mp % 31
		m1 = (m0 + mp ÷ 31 + 1) % 31
		at(seq, n, shift) = seq[mod(n + shift, 31) + 1]
		d0 = zeros(62); d5 = zeros(62)
		for n in 0:30
			s0, s1 = at(s̃, n, m0), at(s̃, n, m1)
			c0, c1 = at(c̃, n, nid2), at(c̃, n, nid2 + 3)
			z0, z1 = at(z̃, n, m0 % 8), at(z̃, n, m1 % 8)
			d0[2n+1], d0[2n+2] = s0 * c0, s1 * c1 * z0      # subframe 0
			d5[2n+1], d5[2n+2] = s1 * c0, s0 * c1 * z1      # subframe 5
		end
		return ComplexF64.(d0), ComplexF64.(d5)
	end
end

# ╔═╡ 0a450000-0000-4000-8000-000000000016
md"""
## Step 5 — Cell-specific reference signals, CRS (TS 36.211 §6.10.1)

The CRS are LTE's **always-on pilots**. They are transmitted in every subframe whether or not anyone is being served. For a passive radar that is the key difference from NR, where an idle cell is almost silent.

* **Values:** QPSK from the Gold sequence with
  ``c_{init} = 2^{10}\,(7(n_s+1) + l + 1)(2N_{ID}^{cell} + 1) + 2N_{ID}^{cell} + 1``,
  where ``n_s`` is the slot in the frame (0–19) and ``l`` the symbol in the slot. The sequence is defined for 110 PRB; a 5 MHz cell uses the middle part (offset ``N_{RB}^{max} - N_{RB}``).
* **Time:** symbols l = 0 and l = 4 of every slot (ports 0 and 1).
* **Frequency:** every 6th subcarrier, ``k = 6m + (v + v_{shift}) \bmod 6``, with ``v_{shift} = \text{PCI} \bmod 6``.
  Port 0 uses ``v = 0`` at l = 0 and ``v = 3`` at l = 4; port 1 the opposite.
  The staggering makes the pilots form a diagonal lattice, so the effective pilot spacing is every 3rd subcarrier. This is what creates the passive radar's range ghosts at ±6.6 km (= c·T_u/3).
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000017
begin
	function crs_values(cell::Integer, ns::Integer, l::Integer)
		c_init = 2^10 * (7 * (ns + 1) + l + 1) * (2cell + 1) + 2cell + 1
		r = qpsk(gold_sequence(c_init, 4N_RB_MAX))
		return r[(0:2N_RB-1) .+ (N_RB_MAX - N_RB) .+ 1]
	end

	"0-based subcarriers of the CRS of antenna `port` in symbol l (0 or 4) of a slot."
	function crs_subcarriers(cell::Integer, port::Integer, l::Integer)
		v = port == 0 ? (l == 0 ? 0 : 3) : (l == 0 ? 3 : 0)
		return [6m + mod(v + cell % 6, 6) for m in 0:2N_RB-1]
	end
end

# ╔═╡ 0a450000-0000-4000-8000-000000000018
md"""
## Step 6 — Fill the resource grid for one frame

The grid is a matrix `grid[k, l]`: subcarrier k (300 rows) × OFDM symbol l (140 columns for one frame). This is the same orientation MATLAB's LTE Toolbox uses. A second matrix `retype` records what each resource element (RE) carries:

| code | type | content here |
|---|---|---|
| 0 | empty | nothing scheduled |
| 1 | pilot | CRS (bit-exact) |
| 2 | sync | PSS / SSS (bit-exact) |
| 3 | control | PBCH, PCFICH/PHICH/PDCCH (random QPSK) |
| 4 | data | PDSCH (random QAM) |

Rules, per subframe:

1. **CRS** on their lattice (Step 5), for 1 or 2 ports.
2. **PSS/SSS** in subframes 0 and 5, symbols 6 and 5 of slot 0 (Steps 3–4). The central 72 subcarriers of those symbols are reserved.
3. **PBCH** in subframe 0, slot 1, symbols 0–3, central 72 subcarriers. It skips REs reserved for CRS of up to 4 ports.
4. **Control region**: the first `cfi` symbols (here 2). It is partly filled, in PRB-sized blocks.
5. **PDSCH**: in an *active* TTI, a random set of PRBs is filled in symbols `cfi`…13 with QAM, avoiding everything above.

The load presets set how often a TTI is active and how many PRBs it uses (the same presets as the Python package).
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000019
begin
	const EMPTY, PILOT, SYNC, CONTROL, DATA = Int8(0), Int8(1), Int8(2), Int8(3), Int8(4)

	# load name => (probability a TTI carries data, fraction of PRBs used, bits per QAM symbol)
	const LOADS = Dict("idle"   => (0.0, 0.0, 2), "low"  => (0.3, 0.33, 4),
	                   "medium" => (0.6, 0.5, 6), "full" => (1.0, 1.0, 2))

	"Unit-average-energy square QAM symbols with random content (bits = 2, 4, 6, 8)."
	function random_qam(rng, bits, n)
		m = 2^(bits ÷ 2)
		levels = -(m - 1):2:(m - 1)
		norm = sqrt(2 * (m^2 - 1) / 3)
		return (rand(rng, levels, n) .+ im .* rand(rng, levels, n)) ./ norm
	end

	"Fill the REs selected by `mask` (a BitMatrix over the subframe) with random QAM."
	function fill_random!(G, T, mask, kind, bits, rng)
		n = count(mask)
		n == 0 && return
		G[mask] .= random_qam(rng, bits, n)
		T[mask] .= kind
	end
end

# ╔═╡ 0a450000-0000-4000-8000-000000000020
function build_frame(cell::Integer; n_ports = 2, cfi = 2, load = "medium", seed = 1, n_subframes = 10)
	p_tti, prb_frac, data_bits = LOADS[load]
	rng = MersenneTwister(seed)
	grid = zeros(ComplexF64, N_SC, 14n_subframes)
	retype = zeros(Int8, N_SC, 14n_subframes)

	nid1, nid2, vshift = cell ÷ 3, cell % 3, cell % 6
	pss = lte_pss(nid2)
	sss0, sss5 = lte_sss(nid1, nid2)
	centre = N_SC ÷ 2                                  # 150 (0-based subcarrier just above DC)
	sync_k = (0:61) .- 31 .+ centre .+ 1               # 1-based rows of PSS/SSS
	central = falses(N_SC); central[(0:71) .- 36 .+ centre .+ 1] .= true
	pbch_skip = [(k - vshift) % 3 == 0 for k in 0:N_SC-1]   # CRS positions of up to 4 ports

	for sf in 0:n_subframes-1
		cols = 14sf+1:14sf+14
		G, T = view(grid, :, cols), view(retype, :, cols)
		reserved = falses(N_SC, 14)
		sfn_sf = sf % 10

		# 1. CRS
		for slot in 0:1, l in (0, 4)
			vals = crs_values(cell, 2sfn_sf + slot, l)
			for port in 0:n_ports-1
				ks = crs_subcarriers(cell, port, l) .+ 1
				G[ks, 7slot+l+1] .= vals
				T[ks, 7slot+l+1] .= PILOT
			end
		end

		# 2. PSS / SSS in subframes 0 and 5
		if sfn_sf in (0, 5)
			G[sync_k, 7] .= pss;                                T[sync_k, 7] .= SYNC
			G[sync_k, 6] .= (sfn_sf == 0 ? sss0 : sss5);       T[sync_k, 6] .= SYNC
			reserved[central, 6:7] .= true
		end

		# 3. PBCH: subframe 0, slot 1, symbols 0..3
		if sfn_sf == 0
			for l in 0:3
				col = 8 + l
				mask = falses(N_SC, 14)
				mask[:, col] .= central .& (T[:, col] .== EMPTY) .& (l < 2 ? .!pbch_skip : trues(N_SC))
				fill_random!(G, T, mask, CONTROL, 2, rng)
				reserved[central, col] .= true
			end
		end

		active = rand(rng) < p_tti
		# 4. Control region, PRB-sized blocks
		ctrl_frac = min(1.0, 0.1 + 0.5active)
		ctrl = falses(N_SC, 14)
		for l in 1:cfi, prb in 0:N_RB-1
			if rand(rng) < ctrl_frac
				rows = 12prb+1:12prb+12
				ctrl[rows, l] .= T[rows, l] .== EMPTY
			end
		end
		fill_random!(G, T, ctrl, CONTROL, 2, rng)
		reserved[:, 1:cfi] .= true

		# 5. PDSCH on a random set of PRBs
		if active
			prbs = randperm(rng, N_RB)[1:max(1, round(Int, prb_frac * N_RB))] .- 1
			rowmask = falses(N_SC)
			for p in prbs
				rowmask[12p+1:12p+12] .= true
			end
			fill_random!(G, T, (T .== EMPTY) .& .!reserved .& rowmask, DATA, data_bits, rng)
		end
	end
	return grid, retype
end

# ╔═╡ 0a450000-0000-4000-8000-000000000021
grid, retype = build_frame(cell_id; n_ports = n_ports, load = load_name);

# ╔═╡ 0a450000-0000-4000-8000-000000000022
let
	share(t) = round(100 * count(==(t), retype) / length(retype), digits = 1)
	md"""
	**This frame** (PCI $(cell_id), $(n_ports) port(s), load *$(load_name)*): pilots $(share(PILOT)) %, sync $(share(SYNC)) %, control $(share(CONTROL)) %, data $(share(DATA)) %, empty $(share(EMPTY)) % of all REs.
	One CRS port is 4.8 % of REs (8 per PRB per subframe of 168); two ports double that.
	"""
end

# ╔═╡ 0a450000-0000-4000-8000-000000000023
md"""
## Step 7 — OFDM modulation (TS 36.211 §6.12)

For every symbol:

1. Place the 300 subcarrier values into a 512-point vector at their IFFT bins. Negative frequencies wrap to the top half, and DC stays empty.
2. Take the **IFFT** to get 512 time samples.
3. Prepend the **cyclic prefix**, a copy of the last 40 or 36 samples. It absorbs multipath (up to 4.7 µs here) so each subcarrier sees a simple complex gain.

Scaling: multiplying by ``N_{FFT}/\sqrt{300}`` gives a fully loaded carrier unit mean power per sample, so a lightly loaded cell shows proportionally less power.

Demodulation is the exact reverse (drop the CP, FFT, pick the 300 bins). A round trip must return the grid unchanged.
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000024
begin
	const BINS = fft_bin.(subcarrier_offsets())

	function ofdm_modulate(grid::AbstractMatrix)
		nsym = size(grid, 2)
		cps = cp_lengths(nsym)
		out = Vector{ComplexF64}(undef, sum(cps) + nsym * NFFT)
		X = zeros(ComplexF64, NFFT)
		P = plan_ifft(X)
		pos = 1
		for l in 1:nsym
			fill!(X, 0)
			X[BINS] .= view(grid, :, l)
			body = (P * X) .* (NFFT / sqrt(N_SC))
			cp = cps[l]
			out[pos:pos+cp-1] .= @view body[end-cp+1:end]
			out[pos+cp:pos+cp+NFFT-1] .= body
			pos += cp + NFFT
		end
		return out
	end

	function ofdm_demodulate(iq::AbstractVector, nsym::Integer)
		cps = cp_lengths(nsym)
		grid = zeros(ComplexF64, N_SC, nsym)
		pos = 1
		for l in 1:nsym
			body = iq[pos+cps[l]:pos+cps[l]+NFFT-1]
			grid[:, l] .= fft(body)[BINS] .* (sqrt(N_SC) / NFFT)
			pos += cps[l] + NFFT
		end
		return grid
	end
end

# ╔═╡ 0a450000-0000-4000-8000-000000000025
iq = ofdm_modulate(grid);

# ╔═╡ 0a450000-0000-4000-8000-000000000026
md"""
## Step 8 — Look at the result

**Resource-element map** of the first subframe: CRS lattice, the control region in symbols 0–1, PSS and SSS in symbols 6 and 5, PBCH in symbols 7–10 around DC, and the PDSCH allocation.
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000027
heatmap(0:13, 0:N_SC-1, Float64.(retype[:, 1:14]);
	c = cgrad([:white, :crimson, :royalblue, :orange, :mediumseagreen], 5, categorical = true),
	clims = (-0.5, 4.5), colorbar_title = "RE type: 0 empty, 1 CRS, 2 PSS/SSS, 3 control, 4 data",
	xlabel = "OFDM symbol (subframe 0)", ylabel = "subcarrier k", xticks = 0:13,
	title = "Resource elements, PCI $(cell_id), $(n_ports) port(s), $(load_name) load", size = (760, 520))

# ╔═╡ 0a450000-0000-4000-8000-000000000028
md"""
**Power over time.** Mean power of each OFDM symbol across the frame. Bursty traffic shows directly; for a passive radar, this is the amplitude modulation every echo inherits.
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000029
let
	sym_power = vec(sum(abs2, grid; dims = 1)) ./ N_SC
	bar(0:length(sym_power)-1, 10 .* log10.(sym_power .+ 1e-6); legend = false, ylims = (-25, 2),
		xlabel = "OFDM symbol in frame", ylabel = "power [dB rel. full load]",
		title = "Per-symbol power ($(load_name) load)", size = (760, 300), linecolor = :match)
end

# ╔═╡ 0a450000-0000-4000-8000-000000000030
md"""
**Spectrum** (averaged periodogram, 512-point Hann segments ≈ 15 kHz resolution). The 4.5 MHz occupied band sits inside the 7.68 MHz sample rate. The guard bands are only as clean as plain OFDM allows: a real eNB adds a transmit filter. The empty DC subcarrier appears as a small notch at 0 Hz.
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000031
let
	seg = NFFT
	w = 0.5 .- 0.5 .* cos.(2π .* (0:seg-1) ./ seg)
	nseg = length(iq) ÷ seg
	psd = zeros(seg)
	for s in 0:nseg-1
		psd .+= abs2.(fft(w .* iq[s*seg+1:(s+1)*seg]))
	end
	psd = fftshift(psd) ./ maximum(psd)
	f = fftshift(fftfreq(seg, FS)) ./ 1e6
	plot(f, 10 .* log10.(psd); legend = false, xlabel = "frequency offset [MHz]", ylabel = "dB",
		 title = "Spectrum: 300 subcarriers × 15 kHz = 4.5 MHz occupied", ylims = (-40, 3), size = (760, 320))
	vline!([-2.25, 2.25]; ls = :dash, color = :gray)
end

# ╔═╡ 0a450000-0000-4000-8000-000000000032
md"""
## Step 9 — Verify it

A waveform generator is only useful if it is *right*. Three independent checks:

1. **Frame timing from the spec.** A frame must be 76 800 samples. The PSS body (after its CP) in subframe 0 must start at sample 552 + 5 × 548 + 36 = **3328** (0-based), worked out by hand from Step 1. We find it by correlating the IQ with a PSS-only replica, which is exactly what a receiver does during cell search.
2. **Round trip.** Demodulating the IQ must return the grid.
3. **Independent implementation.** The repository holds CRS, PSS and SSS vectors dumped from **srsRAN 4G's own PHY library** (`tests/data/srsran/`, made by `tools/srsran/build_and_dump.sh`) for PCI 0, 1, 2, 5, 101, 250 and 503 with 1 and 2 ports. If the PCI selected in Step 0 is one of those, this compares every CRS and sync RE. PSS uses a 5·10⁻⁴ rad phase tolerance because srsRAN computes it in float32 (Step 3).
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000033
begin
	"Read srsRAN dump records: (kind, symbol, k, re, im) little-endian; kind 1 = CRS, 2 = PSS/SSS."
	function read_srsran(path)
		recs = Tuple{Int32,Int32,Int32,ComplexF64}[]
		open(path) do io
			while !eof(io)
				kind, sym, k = read(io, Int32), read(io, Int32), read(io, Int32)
				re, im_ = read(io, Float32), read(io, Float32)
				push!(recs, (kind, sym, k, complex(Float64(re), Float64(im_))))
			end
		end
		return recs
	end

	function srsran_grids(path)
		crs = zeros(ComplexF64, N_SC, 140); sync = zeros(ComplexF64, N_SC, 140)
		for (kind, sym, k, v) in read_srsran(path)
			(kind == 1 ? crs : sync)[k+1, sym+1] = v
		end
		return crs, sync
	end
end

# ╔═╡ 0a450000-0000-4000-8000-000000000034
verification = let
	# 1. timing
	frame_len = length(iq)
	pss_col = zeros(ComplexF64, N_SC, 1)
	pss_rows = findall(==(SYNC), retype[:, 7])
	pss_col[pss_rows, 1] .= grid[pss_rows, 7]
	replica = ofdm_modulate(pss_col)[CP_LONG+1:end]          # PSS body without CP
	corr = [abs(dot(replica, view(iq, i:i+NFFT-1))) for i in 1:8000]
	pss_at = argmax(corr) - 1                                # 0-based sample index

	# 2. round trip
	rt_err = maximum(abs.(ofdm_demodulate(iq, size(grid, 2)) .- grid))

	# 3. srsRAN
	path = joinpath(@__DIR__, "..", "tests", "data", "srsran", "lte_c$(cell_id)_p$(n_ports).bin")
	srs = if isfile(path)
		crs_ref, sync_ref = srsran_grids(path)
		mine_crs = ifelse.(retype[:, 1:140] .== PILOT, grid[:, 1:140], 0.0im)
		mine_sync = ifelse.(retype[:, 1:140] .== SYNC, grid[:, 1:140], 0.0im)
		sss_cols, pss_cols = [6, 76], [7, 77]
		(crs_positions = (mine_crs .!= 0) == (crs_ref .!= 0),
		 crs_max_err = maximum(abs.(mine_crs .- crs_ref)),
		 sss_max_err = maximum(abs.(mine_sync[:, sss_cols] .- sync_ref[:, sss_cols])),
		 pss_max_phase_err = maximum(abs.(angle.(mine_sync[pss_rows, pss_cols] ./ sync_ref[pss_rows, pss_cols]))))
	else
		"no srsRAN vectors for PCI $(cell_id) (available: 0, 1, 2, 5, 101, 250, 503)"
	end
	(frame_samples = frame_len, frame_ok = frame_len == 76800,
	 pss_sample = pss_at, pss_ok = pss_at == 3328,
	 roundtrip_max_err = rt_err, roundtrip_ok = rt_err < 1e-9, srsran = srs)
end

# ╔═╡ 0a450000-0000-4000-8000-000000000035
let
	v = verification
	s = v.srsran
	srs_line = s isa String ? s :
		"CRS positions identical: $(s.crs_positions); max |CRS error| = $(round(s.crs_max_err, sigdigits = 2)); " *
		"max |SSS error| = $(round(s.sss_max_err, sigdigits = 2)); max PSS phase difference = $(round(s.pss_max_phase_err, sigdigits = 2)) rad " *
		"(srsRAN float32 rounding)"
	md"""
	| check | result | expected |
	|---|---|---|
	| samples per frame | $(v.frame_samples) | 76800 |
	| PSS found at sample | $(v.pss_sample) | 3328 |
	| modulate → demodulate max error | $(round(v.roundtrip_max_err, sigdigits = 2)) | < 1e-9 |
	| vs srsRAN 4G | $(srs_line) | identical / ≤ 5e-4 rad |
	"""
end

# ╔═╡ 0a450000-0000-4000-8000-000000000036
md"""
## Step 10 — Save it

`write` stores the samples as interleaved 32-bit floats (I, Q, I, Q, …), the "cf32" format most SDR tools read:

* **Python:** `np.fromfile("lte450_pci1.cf32", dtype=np.complex64)`
* **GNU Radio:** File Source, type complex
* **MATLAB:** `fread` with `'float32'`, then pair the values

Remember the sample rate (7.68 Msps). The file has no header.
"""

# ╔═╡ 0a450000-0000-4000-8000-000000000037
iq_file = let
	path = joinpath(@__DIR__, "lte450_pci$(cell_id)_$(load_name).cf32")
	write(path, ComplexF32.(iq))
	"wrote $(length(iq)) samples ($(round(length(iq) / FS * 1e3, digits = 1)) ms at $(FS / 1e6) Msps) to $(basename(path))"
end

# ╔═╡ 0a450000-0000-4000-8000-000000000038
md"""
## Things to try

* Move the **PCI** slider and watch the CRS lattice shift (``v_{shift}`` = PCI mod 6) and the PSS root change (PCI mod 3).
* Switch to **1 port** and compare the RE map: the port-1 CRS positions turn into data REs.
* Set the load to **idle**: only CRS, PSS/SSS, PBCH and a little control remain. That is what a passive radar sees at night.
* Extend `build_frame` to more subframes (`n_subframes = 20`) and check that the PSS appears every 5 ms.
* Next step on the radar side: delay a copy of `iq`, add a Doppler shift and noise, and cross-correlate. That is a one-target passive radar in about ten lines.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
FFTW = "~1.10.0"
Plots = "~1.41.7"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "086e998594751ec336ae26fa99540f7909595dc8"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f4d39eee89f1e58c26bf447f1d4156c0125d6838"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "c3f99f8e7c98031b8845ece9db86dca713ab9bf8"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14003.0+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "88352712893ec50bee3680605891eaf0e9ed6368"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.8.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b7970cef8ae1c990ba0c09cd8bdc1145e006632f"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "22.1.7+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7787087cbc5ec110986e94d9aa1f17639a79c5de"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05f45c2e0de6259db764adbfd2f1dc6d3f8de13c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "2e59214e017a55cb87474a00fa76035c82ac0e17"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+2"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ef17c47d22224aaecc76e597ab21a072e025cf7b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.14.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "da8c1f6eee04831f14edcfa5dae611d309807e57"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.3.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─0a450000-0000-4000-8000-000000000001
# ╠═0a450000-0000-4000-8000-000000000002
# ╟─0a450000-0000-4000-8000-000000000004
# ╠═0a450000-0000-4000-8000-000000000005
# ╠═0a450000-0000-4000-8000-000000000006
# ╟─0a450000-0000-4000-8000-000000000007
# ╟─0a450000-0000-4000-8000-000000000008
# ╠═0a450000-0000-4000-8000-000000000009
# ╟─0a450000-0000-4000-8000-000000000010
# ╟─0a450000-0000-4000-8000-000000000011
# ╠═0a450000-0000-4000-8000-000000000012
# ╟─0a450000-0000-4000-8000-000000000013
# ╟─0a450000-0000-4000-8000-000000000014
# ╠═0a450000-0000-4000-8000-000000000015
# ╟─0a450000-0000-4000-8000-000000000016
# ╠═0a450000-0000-4000-8000-000000000017
# ╟─0a450000-0000-4000-8000-000000000018
# ╠═0a450000-0000-4000-8000-000000000019
# ╠═0a450000-0000-4000-8000-000000000020
# ╠═0a450000-0000-4000-8000-000000000021
# ╟─0a450000-0000-4000-8000-000000000022
# ╟─0a450000-0000-4000-8000-000000000023
# ╠═0a450000-0000-4000-8000-000000000024
# ╠═0a450000-0000-4000-8000-000000000025
# ╟─0a450000-0000-4000-8000-000000000003
# ╟─0a450000-0000-4000-8000-000000000026
# ╟─0a450000-0000-4000-8000-000000000027
# ╟─0a450000-0000-4000-8000-000000000028
# ╟─0a450000-0000-4000-8000-000000000029
# ╟─0a450000-0000-4000-8000-000000000030
# ╟─0a450000-0000-4000-8000-000000000031
# ╟─0a450000-0000-4000-8000-000000000032
# ╠═0a450000-0000-4000-8000-000000000033
# ╠═0a450000-0000-4000-8000-000000000034
# ╟─0a450000-0000-4000-8000-000000000035
# ╟─0a450000-0000-4000-8000-000000000036
# ╠═0a450000-0000-4000-8000-000000000037
# ╟─0a450000-0000-4000-8000-000000000038
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
