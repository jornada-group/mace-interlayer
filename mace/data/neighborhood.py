import itertools
from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list

NMAX_Z = 400


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    # temp_cell = np.copy(cell)
    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, cell


def get_neighborhood_layered(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
    atomic_numbers: Optional[np.ndarray] = None,
    layer_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if layer_ids is None:
        raise ValueError("layer_ids are required for layered neighborhoods")

    layer_ids = np.asarray(layer_ids)
    unique_layer_ids = np.unique(layer_ids)
    if len(unique_layer_ids) != 2:
        raise ValueError(
            "Layered neighborhoods currently require exactly two layer_ids values; "
            f"got {unique_layer_ids}"
        )

    if atomic_numbers is None:
        atomic_numbers = np.zeros(positions.shape[0], dtype=np.int32)
    atomic_numbers = np.asarray(atomic_numbers)
    typed_numbers = layer_ids * NMAX_Z + atomic_numbers

    cutoff_dict = {}
    for pair in itertools.combinations(np.unique(typed_numbers), 2):
        if abs(pair[1] - pair[0]) > NMAX_Z // 2:
            cutoff_dict[pair] = cutoff

    if pbc is None:
        pbc = (True, True, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff_dict,
        numbers=typed_numbers.astype(np.int32),
    )

    if not true_self_interaction:
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    edge_index = np.stack((sender, receiver))
    shifts = np.dot(unit_shifts, cell)

    return edge_index, shifts, unit_shifts, cell
