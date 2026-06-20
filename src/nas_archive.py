import copy

from nas_encoding import decode_individual, hamming_distance, individual_key, normalize_individual
from nsga2 import assign_rank_and_crowding


def update_archive(
    archive: list[dict],
    population: list[dict],
    experiment: int,
    capacity: int = 30,
    top_k: int = 3,
) -> list[dict]:
    ranked_population = assign_rank_and_crowding(population)
    pareto = [row for row in ranked_population if row["rank"] == 0]
    top_f1 = sorted(ranked_population, key=lambda row: row["dev_f1"], reverse=True)[:top_k]
    candidates = assign_rank_and_crowding([*archive, *pareto, *top_f1])
    candidates.sort(key=lambda row: (row["rank"], -row["crowding_distance"], -row["dev_f1"]))

    unique = []
    seen = set()
    for row in candidates:
        normalized = normalize_individual(row["individual"], experiment)
        key = individual_key(normalized, experiment)
        if key in seen:
            continue
        seen.add(key)
        item = dict(row)
        item["individual"] = list(normalized)
        unique.append(item)

    selected = []
    for row in unique:
        if len(selected) == capacity:
            break
        if not selected or min(
            hamming_distance(row["individual"], item["individual"], experiment) for item in selected
        ) > 1:
            selected.append(row)
    if len(selected) < capacity:
        for row in unique:
            if len(selected) == capacity:
                break
            if row in selected:
                continue
            if not selected or min(
                hamming_distance(row["individual"], item["individual"], experiment) for item in selected
            ) > 0:
                selected.append(row)
    return selected


def select_weight_source(individual, archive: list[dict], experiment: int) -> dict | None:
    normalized = normalize_individual(individual, experiment)
    compatible = [
        row
        for row in archive
        if tuple(normalize_individual(row["individual"], experiment)[:2]) == normalized[:2]
    ]
    if not compatible:
        return None
    return min(
        compatible,
        key=lambda row: (
            hamming_distance(normalized, row["individual"], experiment),
            -row["dev_f1"],
        ),
    )


def _copy_module(target, source) -> None:
    target.load_state_dict(copy.deepcopy(source.state_dict()))


def transfer_partial_weights(
    target_model,
    source_model,
    target_individual,
    source_individual,
    experiment: int,
) -> str | None:
    _copy_module(target_model.encoder.embedding, source_model.encoder.embedding)
    target_arch = decode_individual(target_individual, experiment)
    source_arch = decode_individual(source_individual, experiment)

    for index, (target_spec, source_spec) in enumerate(zip(target_arch["ops"], source_arch["ops"])):
        if target_spec["type"] == "identity":
            continue
        if (
            target_spec == source_spec
            and target_model.encoder.cell.operation_channels[index]
            == source_model.encoder.cell.operation_channels[index]
        ):
            _copy_module(
                target_model.encoder.cell.operations[index],
                source_model.encoder.cell.operations[index],
            )
            return f"op{index + 1}"

    target_final = target_model.encoder.cell.final_conv
    source_final = source_model.encoder.cell.final_conv
    if (
        target_final.in_channels == source_final.in_channels
        and target_final.out_channels == source_final.out_channels
        and target_final.kernel_size == source_final.kernel_size
        and target_final.dilation == source_final.dilation
    ):
        _copy_module(target_final, source_final)
        return "final_conv"
    return None
