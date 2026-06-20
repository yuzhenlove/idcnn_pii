import random
from collections.abc import Sequence


C_VALUES = (64, 128, 256, 512)
RATIO_VALUES = (0.5, 1.0, 2.0)
CELL_NUM_VALUES = (1, 2, 3, 4)
OP_VALUES = ("conv", "dwconv", "sepconv", "identity")
KERNEL_VALUES = (3, 5, 7)
DILATION_VALUES = (1, 2, 4, 8)


def normalize_individual(individual: Sequence[int], experiment: int) -> tuple[int, ...]:
    if len(individual) != 12:
        raise ValueError("NAS individual must contain 12 genes")
    genes = list(individual)
    if experiment == 1:
        genes[3:] = [0, 0, 0, 0, 0, 1, 0, 0, 2]
    elif experiment == 2:
        genes[3], genes[6], genes[9] = 0, 0, 0
    elif experiment == 3:
        for offset in (3, 6, 9):
            if genes[offset] == 3:
                genes[offset + 1] = 0
                genes[offset + 2] = 0
    else:
        raise ValueError("experiment must be 1, 2, or 3")
    return tuple(genes)


def decode_individual(individual: Sequence[int], experiment: int) -> dict:
    genes = normalize_individual(individual, experiment)
    operations = []
    for offset in (3, 6, 9):
        operations.append(
            {
                "type": OP_VALUES[genes[offset]],
                "kernel_size": KERNEL_VALUES[genes[offset + 1]],
                "dilation": DILATION_VALUES[genes[offset + 2]],
            }
        )
    return {
        "C": C_VALUES[genes[0]],
        "ratio": RATIO_VALUES[genes[1]],
        "cell_num": CELL_NUM_VALUES[genes[2]],
        "ops": operations,
    }


def gene_domains(experiment: int) -> tuple[tuple[int, ...], ...]:
    variable = (
        tuple(range(len(C_VALUES))),
        tuple(range(len(RATIO_VALUES))),
        tuple(range(len(CELL_NUM_VALUES))),
    )
    if experiment == 1:
        return variable + ((0,), (0,), (0,), (0,), (0,), (1,), (0,), (0,), (2,))
    if experiment == 2:
        return variable + ((0,), tuple(range(3)), tuple(range(4))) * 3
    if experiment == 3:
        return variable + (tuple(range(4)), tuple(range(3)), tuple(range(4))) * 3
    raise ValueError("experiment must be 1, 2, or 3")


def random_individual(experiment: int, rng: random.Random) -> tuple[int, ...]:
    genes = tuple(rng.choice(domain) for domain in gene_domains(experiment))
    return normalize_individual(genes, experiment)


def crossover(
    parent_a: Sequence[int],
    parent_b: Sequence[int],
    experiment: int,
    rng: random.Random,
    probability: float = 0.9,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if rng.random() >= probability:
        return normalize_individual(parent_a, experiment), normalize_individual(parent_b, experiment)
    child_a, child_b = list(parent_a), list(parent_b)
    for index in range(12):
        if rng.random() < 0.5:
            child_a[index], child_b[index] = child_b[index], child_a[index]
    return normalize_individual(child_a, experiment), normalize_individual(child_b, experiment)


def mutate(
    individual: Sequence[int],
    experiment: int,
    rng: random.Random,
    gene_probability: float = 1 / 12,
) -> tuple[int, ...]:
    genes = list(individual)
    for index, domain in enumerate(gene_domains(experiment)):
        if len(domain) > 1 and rng.random() < gene_probability:
            alternatives = [value for value in domain if value != genes[index]]
            genes[index] = rng.choice(alternatives)
    return normalize_individual(genes, experiment)


def individual_key(individual: Sequence[int], experiment: int) -> str:
    return "-".join(str(value) for value in normalize_individual(individual, experiment))


def hamming_distance(left: Sequence[int], right: Sequence[int], experiment: int) -> int:
    left_normalized = normalize_individual(left, experiment)
    right_normalized = normalize_individual(right, experiment)
    return sum(a != b for a, b in zip(left_normalized, right_normalized))
