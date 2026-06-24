# TMD trained MACE models

Compiled MACE models for transition-metal dichalcogenide (TMD) monolayers and
bilayer interlayer interactions. These checkpoints were trained for the layered
materials work described in
[Georgaras et al., arXiv:2503.15432](https://arxiv.org/abs/2503.15432).

## Intralayer models

Use one model per monolayer species in a bilayer calculation:

| File | Material |
| --- | --- |
| `MoS2-models_rmax2_stagetwo_compiled.model` | MoS₂ monolayer |
| `MoSe2-models_rmax2_stagetwo_compiled.model` | MoSe₂ monolayer |
| `WS2-models_rmax2_stagetwo_compiled.model` | WS₂ monolayer |
| `WSe2-models_rmax2_stagetwo_compiled.model` | WSe₂ monolayer |

## Interlayer models

Use with the corresponding intralayer pair in an `NLayerCalculator` stack:

| File | Bilayer pair |
| --- | --- |
| `MoS2_WS2-models_rmax2_stagetwo_compiled.model` | MoS₂ / WS₂ |
| `MoS2_WSe2-models_rmax2_stagetwo_compiled.model` | MoS₂ / WSe₂ |
| `MoSe2_WSe2-models_rmax2_stagetwo_compiled.model` | MoSe₂ / WSe₂ |
| `MoS2-WSe2-inter-for-jdg.model` | MoS₂ / WSe₂ (alternate interlayer checkpoint) |

## Usage

A three-model stack needs two intralayer models (bottom and top) plus one
interlayer model. See `examples/mose2_wse2_relax_smoke/relax_mose2_wse2.py` for
a MoSe₂/WSe₂ relaxation example.

```bash
python examples/mose2_wse2_relax_smoke/relax_mose2_wse2.py \
  --mo-model examples/TMD_trained_models/MoSe2-models_rmax2_stagetwo_compiled.model \
  --w-model examples/TMD_trained_models/WSe2-models_rmax2_stagetwo_compiled.model \
  --interlayer-model examples/TMD_trained_models/MoSe2_WSe2-models_rmax2_stagetwo_compiled.model
```
