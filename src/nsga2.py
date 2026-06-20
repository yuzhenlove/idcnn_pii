import math
import random


def dominates(left: dict, right: dict) -> bool:
    no_worse = left["dev_f1"] >= right["dev_f1"] and left["flops"] <= right["flops"]
    strictly_better = left["dev_f1"] > right["dev_f1"] or left["flops"] < right["flops"]
    return no_worse and strictly_better


def non_dominated_fronts(population: list[dict]) -> list[list[int]]:
    dominated = [set() for _ in population]
    domination_counts = [0] * len(population)
    fronts = [[]]
    for left_index, left in enumerate(population):
        for right_index, right in enumerate(population):
            if left_index == right_index:
                continue
            if dominates(left, right):
                dominated[left_index].add(right_index)
            elif dominates(right, left):
                domination_counts[left_index] += 1
        if domination_counts[left_index] == 0:
            fronts[0].append(left_index)
    current = 0
    while current < len(fronts) and fronts[current]:
        next_front = []
        for left_index in fronts[current]:
            for right_index in dominated[left_index]:
                domination_counts[right_index] -= 1
                if domination_counts[right_index] == 0:
                    next_front.append(right_index)
        if next_front:
            fronts.append(next_front)
        current += 1
    return fronts


def _crowding_distances(population: list[dict], front: list[int]) -> dict[int, float]:
    distances = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    for objective in ("dev_f1", "flops"):
        ordered = sorted(front, key=lambda index: population[index][objective])
        distances[ordered[0]] = distances[ordered[-1]] = float("inf")
        low = population[ordered[0]][objective]
        high = population[ordered[-1]][objective]
        if high == low:
            continue
        for position in range(1, len(ordered) - 1):
            if math.isinf(distances[ordered[position]]):
                continue
            previous = population[ordered[position - 1]][objective]
            following = population[ordered[position + 1]][objective]
            distances[ordered[position]] += (following - previous) / (high - low)
    return distances


def assign_rank_and_crowding(population: list[dict]) -> list[dict]:
    ranked = [dict(candidate) for candidate in population]
    for rank, front in enumerate(non_dominated_fronts(ranked)):
        distances = _crowding_distances(ranked, front)
        for index in front:
            ranked[index]["rank"] = rank
            ranked[index]["crowding_distance"] = distances[index]
    return ranked


def environmental_selection(population: list[dict], population_size: int) -> list[dict]:
    ranked = assign_rank_and_crowding(population)
    ranked.sort(key=lambda row: (row["rank"], -row["crowding_distance"], -row["dev_f1"]))
    return ranked[:population_size]


def binary_tournament(
    population: list[dict],
    rng: random.Random,
    indices: tuple[int, int] | None = None,
) -> dict:
    left_index, right_index = indices or tuple(rng.sample(range(len(population)), 2))
    left, right = population[left_index], population[right_index]
    left_key = (left["rank"], -left["crowding_distance"], -left["dev_f1"])
    right_key = (right["rank"], -right["crowding_distance"], -right["dev_f1"])
    if left_key == right_key:
        return rng.choice([left, right])
    return left if left_key < right_key else right
