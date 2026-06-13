#!/usr/bin/env python3

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_EXAMPLE_ROOT = THIS_DIR.parents[2] / "mace-interlayer-example"


def import_example_helpers(example_root: Path):
    sys.path.insert(0, str(example_root))
    from macewrapper import MACEWCalculator  # pylint: disable=import-outside-toplevel
    from n_layer import NLayerCalculator  # pylint: disable=import-outside-toplevel

    return MACEWCalculator, NLayerCalculator


def build_calculator(
    atoms,
    mo_model: Path,
    w_model: Path,
    interlayer_model: Path,
    layer_symbols: List[List[str]],
    device: str,
    default_dtype: str,
    macew_calculator_cls,
    n_layer_calculator_cls,
):
    atom_types = atoms.arrays["atom_types"]

    layer0_atoms = atoms[atom_types < len(layer_symbols[0])]
    layer1_atoms = atoms[
        np.logical_and(
            atom_types >= len(layer_symbols[0]),
            atom_types < len(layer_symbols[0]) + len(layer_symbols[1]),
        )
    ]

    intralayer_calcs = [
        macew_calculator_cls(
            layer0_atoms,
            layer_symbols[0],
            model_file=str(mo_model),
            device=device,
            default_dtype=default_dtype,
            is_interlayer_calc=False,
        ),
        macew_calculator_cls(
            layer1_atoms,
            layer_symbols[1],
            model_file=str(w_model),
            device=device,
            default_dtype=default_dtype,
            is_interlayer_calc=False,
        ),
    ]

    interlayer_calc = macew_calculator_cls(
        atoms,
        layer_symbols,
        model_file=str(interlayer_model),
        device=device,
        default_dtype=default_dtype,
        is_interlayer_calc=True,
    )
    return n_layer_calculator_cls(
        [atoms], intralayer_calcs, [interlayer_calc], layer_symbols
    )


def layer_spacing(atoms) -> float:
    layer_ids = atoms.arrays["layer_ids"]
    unique_layers = np.unique(layer_ids)
    if len(unique_layers) != 2:
        raise ValueError(f"Expected exactly two layer_ids values, got {unique_layers}")
    z0 = atoms.positions[layer_ids == unique_layers[0], 2].mean()
    z1 = atoms.positions[layer_ids == unique_layers[1], 2].mean()
    return float(abs(z1 - z0))


def write_extxyz(path: Path, atoms) -> None:
    atoms_to_write = atoms.copy()
    atoms_to_write.calc = None
    write(path, atoms_to_write, format="extxyz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relax the MoSe2-WSe2 bilayer example with the three existing MACE models."
    )
    parser.add_argument("--example-root", type=Path, default=DEFAULT_EXAMPLE_ROOT)
    parser.add_argument("--structure", type=Path, default=None)
    parser.add_argument("--mo-model", type=Path, default=None)
    parser.add_argument("--w-model", type=Path, default=None)
    parser.add_argument("--interlayer-model", type=Path, default=None)
    parser.add_argument(
        "--layer-symbols",
        default="[['Mo','Se','Se'],['W','Se','Se']]",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--default-dtype",
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--rattle-std",
        type=float,
        default=0.0,
        help="Gaussian Cartesian position rattle standard deviation in Angstrom.",
    )
    parser.add_argument("--rattle-seed", type=int, default=123)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    example_root = args.example_root.resolve()
    structure = args.structure or example_root / "MoSe2-WSe2_bilayer_relaxed.xyz"
    mo_model = (
        args.mo_model
        or example_root / "mace_models/MoSe2-models_rmax2_stagetwo_compiled.model"
    )
    w_model = (
        args.w_model
        or example_root / "mace_models/WSe2-models_rmax2_stagetwo_compiled.model"
    )
    interlayer_model = (
        args.interlayer_model
        or example_root
        / "mace_models/MoSe2_WSe2-models_rmax2_stagetwo_compiled.model"
    )
    output_root = args.output_root or example_root / "relax_smoke"

    macew_calculator_cls, n_layer_calculator_cls = import_example_helpers(example_root)
    layer_symbols = ast.literal_eval(args.layer_symbols)

    atoms = read(structure, format="extxyz", index=-1)
    for required_array in ("atom_types", "layer_ids"):
        if required_array not in atoms.arrays:
            raise ValueError(
                f"{structure} is missing atoms.arrays['{required_array}']"
            )

    if args.rattle_std > 0.0:
        rng = np.random.default_rng(args.rattle_seed)
        atoms.positions[:] = atoms.positions + rng.normal(
            scale=args.rattle_std, size=atoms.positions.shape
        )

    initial_positions = atoms.positions.copy()
    initial_spacing = layer_spacing(atoms)

    atoms.calc = build_calculator(
        atoms=atoms,
        mo_model=mo_model,
        w_model=w_model,
        interlayer_model=interlayer_model,
        layer_symbols=layer_symbols,
        device=args.device,
        default_dtype=args.default_dtype,
        macew_calculator_cls=macew_calculator_cls,
        n_layer_calculator_cls=n_layer_calculator_cls,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    initial_energy = float(atoms.get_potential_energy())
    atoms.info["energy"] = initial_energy
    atoms.arrays["forces"] = atoms.get_forces()
    write_extxyz(output_dir / "initial.xyz", atoms)

    optimizer = BFGS(
        atoms,
        trajectory=str(output_dir / "relax.traj"),
        logfile=str(output_dir / "relax.log"),
    )
    optimizer.run(fmax=args.fmax, steps=args.steps)

    final_energy = float(atoms.get_potential_energy())
    final_forces = atoms.get_forces()
    atoms.arrays["forces"] = final_forces
    atoms.info["energy"] = final_energy
    write_extxyz(output_dir / "relaxed.xyz", atoms)

    displacement = atoms.positions - initial_positions
    metrics = {
        "structure": str(structure),
        "mo_model": str(mo_model),
        "w_model": str(w_model),
        "interlayer_model": str(interlayer_model),
        "device": args.device,
        "default_dtype": args.default_dtype,
        "fmax_target": args.fmax,
        "steps_requested": args.steps,
        "steps_run": int(optimizer.nsteps),
        "rattle_std_A": args.rattle_std,
        "rattle_seed": args.rattle_seed,
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "final_fmax_eV_per_A": float(np.linalg.norm(final_forces, axis=1).max()),
        "max_displacement_A": float(np.linalg.norm(displacement, axis=1).max()),
        "rms_displacement_A": float(
            np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
        ),
        "initial_layer_spacing_A": initial_spacing,
        "final_layer_spacing_A": layer_spacing(atoms),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
