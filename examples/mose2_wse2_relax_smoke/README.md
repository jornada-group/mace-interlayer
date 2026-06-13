# MoSe2-WSe2 Relaxation Smoke Test

This smoke test validates the interlayer calculator path against the existing
`mace-interlayer-example` assets without rewriting the source extxyz files.

## Version Policy

- `paper-v1` is the immutable paper snapshot.
- `paper-maintenance` is for paper-only reproducibility fixes.
- `port/mace-v0.3.16-interlayer` is the active MACE v0.3.16 port branch.

The extxyz compatibility contract is unchanged:

- `atom_types`
- `layer_ids`
- existing symbols, positions, cell, PBC, and reference fields

## Run On Perlmutter

From this repository:

```bash
RELAX_DEFAULT_DTYPE=float64 \
RELAX_FMAX=1e-6 \
RELAX_STEPS=200 \
RELAX_RATTLE_STD=0.1 \
RELAX_RATTLE_SEED=123 \
bash examples/mose2_wse2_relax_smoke/run_relax_smoke_gpu.sh
```

By default the launcher expects the existing example assets at:

```text
../mace-interlayer-example
```

Override that with:

```bash
EXAMPLE_ROOT=/path/to/mace-interlayer-example bash examples/mose2_wse2_relax_smoke/run_relax_smoke_gpu.sh
```

Outputs are written under:

```text
$EXAMPLE_ROOT/relax_smoke/<timestamp>/
```

Use `float64` for tight force tolerances. In tests, `float32` reached the right
geometry but did not reliably converge to `1e-6 eV/A` because of force noise.
