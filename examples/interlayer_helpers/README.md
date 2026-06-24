# interlayer_helpers

Helper modules that provide the stacked intra-layer + inter-layer ASE
calculator used by `mlip_phonon_scattering`'s interlayer mode.

## Contents

- `n_layer.py` — defines `NLayerCalculator`, an ASE calculator that re-splits a
  passed structure by `atoms.arrays['atom_types']` value ranges and sums the
  per-layer (intra-layer) and adjacent-pair (inter-layer) contributions.
- `macewrapper.py` — defines `MACEWCalculator`, a MACE wrapper that binds to a
  fixed atom subset (storing its `atom_types` / `layer_ids`) and evaluates
  either an intra-layer or inter-layer model.

## Note on the `ittnotify` preload

`macewrapper.py` contains an optional `ittnotify` preload that guards against
PyTorch builds whose oneDNN JIT profiling references undefined `iJIT_*` symbols
at `import torch`. It is a **no-op** when no such library is found, and it is
**not needed** on the NERSC `pytorch/2.11.0` module (torch 2.11.0+cu129 imports
cleanly and a full MACE evaluation runs without any stub), so no stub binary is
shipped here. If you ever hit an `undefined symbol: __itt_*` error on a different
PyTorch build, build the tiny stub from `mlip_phonon_scattering/ittnotify_stub/`
(`make`) and place the resulting `libittnotify.so` in an `ittnotify_stub/`
directory next to `macewrapper.py`.

## Provenance

`n_layer.py` and `macewrapper.py` were copied verbatim from the (non-git)
`mace-interlayer-example` working directory. Put this folder on `PYTHONPATH`
(via `load_mace_phonon_env.sh`) so that `from macewrapper import MACEWCalculator`
and `from n_layer import NLayerCalculator` resolve at runtime.
