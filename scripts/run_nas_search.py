import argparse
import ast
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nas_archive import select_weight_source, update_archive
from nas_encoding import (
    crossover,
    decode_individual,
    hamming_distance,
    individual_key,
    mutate,
    normalize_individual,
    random_individual,
)
from nsga2 import assign_rank_and_crowding, binary_tournament, environmental_selection
from utils import read_json, write_json


def search_budget(population_size: int, generations: int) -> int:
    return population_size * (generations + 1)


def individual_argument(individual) -> str:
    return ",".join(str(value) for value in individual)


def parse_gpu_list(value: str) -> tuple[str, ...]:
    gpus = tuple(item.strip() for item in value.split(",") if item.strip())
    if not gpus:
        raise ValueError("GPU list must not be empty")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must not contain duplicates")
    return gpus


def estimate_candidate_cost(individual, experiment: int) -> float:
    architecture = decode_individual(individual, experiment)
    channels = architecture["C"]
    bottleneck = int(channels * architecture["ratio"])
    current_channels = channels
    cell_cost = 0
    for operation in architecture["ops"]:
        operation_type = operation["type"]
        kernel_size = operation["kernel_size"]
        if operation_type == "conv":
            cell_cost += current_channels * bottleneck * kernel_size
            current_channels = bottleneck
        elif operation_type == "dwconv":
            cell_cost += current_channels * kernel_size
        elif operation_type == "sepconv":
            cell_cost += current_channels * kernel_size + current_channels * bottleneck
            current_channels = bottleneck
    cell_cost += current_channels * channels * 3
    initial_conv_cost = channels * channels * 3
    head_cost = channels * 128
    return initial_conv_cost + architecture["cell_num"] * cell_cost + head_cost


def estimate_training_seconds(individual, history: list[dict], experiment: int) -> float:
    normalized = normalize_individual(individual, experiment)
    compatible = [
        row
        for row in history
        if tuple(normalize_individual(row["individual"], experiment)[:2]) == normalized[:2]
    ]
    candidates = compatible or history
    if not candidates:
        return estimate_candidate_cost(normalized, experiment)
    nearest = sorted(
        candidates,
        key=lambda row: hamming_distance(normalized, row["individual"], experiment),
    )[:5]
    weighted_seconds = 0.0
    total_weight = 0.0
    for row in nearest:
        distance = hamming_distance(normalized, row["individual"], experiment)
        weight = 1 / (distance + 1)
        weighted_seconds += row["train_seconds"] * weight
        total_weight += weight
    return weighted_seconds / total_weight


def assign_candidates_to_gpus(
    candidates: list[tuple[object, float]],
    gpus: tuple[str, ...],
    workers_per_gpu: int,
) -> list[tuple[object, float, str]]:
    max_candidates_per_gpu = max(
        workers_per_gpu,
        (len(candidates) + len(gpus) - 1) // len(gpus),
    )
    loads = {gpu_id: 0.0 for gpu_id in gpus}
    counts = {gpu_id: 0 for gpu_id in gpus}
    assignments = []
    for candidate, cost in sorted(candidates, key=lambda item: item[1], reverse=True):
        available = [
            gpu_id for gpu_id in gpus if counts[gpu_id] < max_candidates_per_gpu
        ]
        gpu_id = min(available, key=lambda item: (loads[item], counts[item], gpus.index(item)))
        assignments.append((candidate, cost, gpu_id))
        loads[gpu_id] += cost
        counts[gpu_id] += 1
    return assignments


def build_candidate_command(
    individual,
    experiment: int,
    output_dir: Path,
    source: dict | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    cpu: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "src/nas_train.py",
        "--experiment",
        str(experiment),
        "--individual",
        individual_argument(individual),
        "--output-dir",
        str(output_dir),
        "--early-stop-patience",
        "20",
    ]
    if source is not None:
        command.extend(
            [
                "--source-checkpoint",
                source["checkpoint"],
                "--source-individual",
                individual_argument(source["individual"]),
            ]
        )
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if cpu:
        command.append("--cpu")
    return command


def generate_unique_initial_population(
    population_size: int,
    experiment: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    population = []
    seen = set()
    while len(population) < population_size:
        individual = random_individual(experiment, rng)
        key = individual_key(individual, experiment)
        if key not in seen:
            seen.add(key)
            population.append(individual)
    return population


def generate_offspring(
    ranked_population: list[dict],
    population_size: int,
    experiment: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    offspring = []
    while len(offspring) < population_size:
        parent_a = binary_tournament(ranked_population, rng)["individual"]
        parent_b = binary_tournament(ranked_population, rng)["individual"]
        child_a, child_b = crossover(parent_a, parent_b, experiment, rng, probability=0.9)
        offspring.append(mutate(child_a, experiment, rng, gene_probability=1 / 12))
        if len(offspring) < population_size:
            offspring.append(mutate(child_b, experiment, rng, gene_probability=1 / 12))
    return offspring


def result_row(result: dict) -> dict:
    return {
        "candidate_id": result.get("candidate_id"),
        "individual": result["individual"],
        "dev_f1": result["dev_f1"],
        "flops": result["flops"],
        "checkpoint": result["checkpoint"],
        "train_seconds": result.get("train_seconds", 0.0),
        "source_checkpoint": result.get("source_checkpoint"),
        "transferred_module": result.get("transferred_module"),
    }


def load_cached_results(
    individuals,
    candidates_dir: Path,
    experiment: int,
) -> tuple[dict[str, dict], list[tuple[int, ...]]]:
    cached = {}
    missing = []
    seen_missing = set()
    for individual in individuals:
        normalized = normalize_individual(individual, experiment)
        key = individual_key(normalized, experiment)
        result_path = candidates_dir / key / "candidate.json"
        if result_path.exists():
            cached[key] = result_row(read_json(result_path))
        elif key not in seen_missing:
            seen_missing.add(key)
            missing.append(normalized)
    return cached, missing


def run_parallel_jobs(
    jobs: list[tuple[list[str], Path] | tuple[list[str], Path, str]],
    workers: int,
    status_interval: int,
    gpus: tuple[str, ...] | None = None,
    workers_per_gpu: int = 1,
) -> None:
    pending = list(jobs)
    active = []
    available_gpu_slots = list(gpus) * workers_per_gpu if gpus is not None else []
    active_gpu_counts = {gpu_id: 0 for gpu_id in gpus or ()}
    completed = 0
    started = time.time()
    last_status = started
    while pending or active:
        while pending and len(active) < workers:
            job_index = None
            for index, job in enumerate(pending):
                assigned_gpu = job[2] if len(job) == 3 else None
                if assigned_gpu is not None:
                    if active_gpu_counts[assigned_gpu] < workers_per_gpu:
                        job_index = index
                        break
                elif gpus is None or available_gpu_slots:
                    job_index = index
                    break
            if job_index is None:
                break
            job = pending.pop(job_index)
            command, output_dir = job[:2]
            preassigned = len(job) == 3
            gpu_id = job[2] if preassigned else None
            if gpu_id is None and gpus is not None:
                gpu_id = available_gpu_slots.pop(0)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(output_dir / "process.log", "w", encoding="utf-8")
            process_env = None
            if gpu_id is not None:
                active_gpu_counts[gpu_id] += 1
                process_env = os.environ.copy()
                process_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                process_env["CUDA_VISIBLE_DEVICES"] = gpu_id
                print(
                    f"launch candidate={output_dir.name} gpu={gpu_id}",
                    flush=True,
                )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                **({"env": process_env} if process_env is not None else {}),
            )
            active.append((process, log_file, command, gpu_id, preassigned))
        for item in list(active):
            process, log_file, command, gpu_id, preassigned = item
            return_code = process.poll()
            if return_code is None:
                continue
            log_file.close()
            active.remove(item)
            if gpu_id is not None:
                active_gpu_counts[gpu_id] -= 1
                if not preassigned:
                    available_gpu_slots.append(gpu_id)
            completed += 1
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        now = time.time()
        if now - last_status >= status_interval:
            remaining = len(jobs) - completed
            eta = (now - started) / completed * remaining if completed else None
            eta_text = f"{eta / 60:.1f} min" if eta is not None else "unknown"
            print(
                f"candidate jobs completed={completed}/{len(jobs)} "
                f"active={len(active)} pending={len(pending)} "
                f"elapsed={(now - started) / 60:.1f} min eta={eta_text}",
                flush=True,
            )
            last_status = now
        if active:
            time.sleep(1)


def evaluate_population(
    individuals,
    experiment: int,
    candidates_dir: Path,
    archive: list[dict],
    workers: int,
    status_interval: int,
    epochs: int | None,
    batch_size: int | None,
    cpu: bool,
    gpus: tuple[str, ...] | None = None,
    workers_per_gpu: int = 1,
) -> list[dict]:
    cached, missing = load_cached_results(individuals, candidates_dir, experiment)
    history = []
    for result_path in candidates_dir.glob("*/candidate.json"):
        result = read_json(result_path)
        if result.get("train_seconds") is not None:
            history.append(result)
    jobs = []
    job_costs = []
    for individual in missing:
        key = individual_key(individual, experiment)
        output_dir = candidates_dir / key
        source = select_weight_source(individual, archive, experiment)
        command = build_candidate_command(
            individual,
            experiment,
            output_dir,
            source=source,
            epochs=epochs,
            batch_size=batch_size,
            cpu=cpu,
        )
        jobs.append((command, output_dir))
        job_costs.append(
            ((command, output_dir), estimate_training_seconds(individual, history, experiment))
        )
        print(" ".join(command), flush=True)
    if jobs:
        if gpus:
            assignments = assign_candidates_to_gpus(job_costs, gpus, workers_per_gpu)
            jobs = [
                (command, output_dir, gpu_id)
                for (command, output_dir), _cost, gpu_id in assignments
            ]
            gpu_loads = {
                gpu_id: sum(
                    cost
                    for _job, cost, assigned_gpu in assignments
                    if assigned_gpu == gpu_id
                )
                for gpu_id in gpus
            }
            print(
                "estimated GPU loads "
                + " ".join(f"gpu={gpu_id}:{gpu_loads[gpu_id]:.1f}" for gpu_id in gpus),
                flush=True,
            )
        run_parallel_jobs(
            jobs,
            workers,
            status_interval,
            gpus=gpus,
            workers_per_gpu=workers_per_gpu,
        )
    completed, still_missing = load_cached_results(individuals, candidates_dir, experiment)
    if still_missing:
        raise RuntimeError(f"candidate results missing after evaluation: {still_missing}")
    return [completed[individual_key(individual, experiment)] for individual in individuals]


def write_pareto_outputs(rows: list[dict], output_dir: Path) -> list[dict]:
    ranked = assign_rank_and_crowding(rows)
    pareto = [row for row in ranked if row["rank"] == 0]
    pareto.sort(key=lambda row: (row["flops"], -row["dev_f1"]))
    write_json(pareto, output_dir / "pareto.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pareto.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["candidate_id", "dev_f1", "flops", "checkpoint", "individual"],
        )
        writer.writeheader()
        for row in pareto:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "dev_f1": row["dev_f1"],
                    "flops": row["flops"],
                    "checkpoint": row["checkpoint"],
                    "individual": json.dumps(row["individual"]),
                }
            )
    return pareto


def save_search_state(
    path: Path,
    population: list[dict],
    archive: list[dict],
    all_results: dict[str, dict],
    completed_generations: int,
    rng: random.Random,
    elapsed_seconds: float,
) -> None:
    write_json(
        {
            "population": population,
            "archive": archive,
            "all_results": all_results,
            "completed_generations": completed_generations,
            "rng_state": repr(rng.getstate()),
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def run_search(args, experiment: int) -> list[dict]:
    experiment_dir = ROOT / "outputs" / "nas" / f"experiment_{experiment}"
    candidates_dir = experiment_dir / "candidates"
    generations_dir = experiment_dir / "generations"
    state_path = experiment_dir / "search_state.json"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.search_seed)
    run_started = time.time()

    if state_path.exists() and not args.force:
        state = read_json(state_path)
        population = state["population"]
        archive = state["archive"]
        all_results = state["all_results"]
        completed_generations = state["completed_generations"]
        elapsed_before = state["elapsed_seconds"]
        rng.setstate(ast.literal_eval(state["rng_state"]))
        print(f"resume experiment {experiment} after generation {completed_generations}", flush=True)
    else:
        population_individuals = generate_unique_initial_population(
            args.population_size,
            experiment,
            rng,
        )
        if args.dry_run:
            dry_run_rows = []
            for individual in population_individuals:
                key = individual_key(individual, experiment)
                command = " ".join(
                    build_candidate_command(
                        individual,
                        experiment,
                        candidates_dir / key,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        cpu=args.cpu,
                    )
                )
                dry_run_rows.append((command, estimate_candidate_cost(individual, experiment)))
            if args.gpus:
                assignments = assign_candidates_to_gpus(
                    dry_run_rows,
                    args.gpus,
                    args.workers_per_gpu,
                )
                dry_run_rows = [
                    (f"CUDA_VISIBLE_DEVICES={gpu_id} {command}", cost)
                    for command, cost, gpu_id in assignments
                ]
            for command, _cost in dry_run_rows:
                print(command)
            return []
        population = evaluate_population(
            population_individuals,
            experiment,
            candidates_dir,
            archive=[],
            workers=args.workers,
            status_interval=args.status_interval,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cpu=args.cpu,
            gpus=args.gpus,
            workers_per_gpu=args.workers_per_gpu,
        )
        archive = update_archive([], population, experiment, capacity=30, top_k=3)
        all_results = {row["candidate_id"]: row for row in population}
        completed_generations = 0
        elapsed_before = 0.0
        write_json(
            {"generation": 0, "kind": "initial_population", "population": population, "archive": archive},
            generations_dir / "generation_000.json",
        )
        save_search_state(
            state_path,
            population,
            archive,
            all_results,
            completed_generations,
            rng,
            time.time() - run_started,
        )

    for generation in range(completed_generations + 1, args.generations + 1):
        ranked_population = assign_rank_and_crowding(population)
        offspring_individuals = generate_offspring(
            ranked_population,
            args.population_size,
            experiment,
            rng,
        )
        offspring = evaluate_population(
            offspring_individuals,
            experiment,
            candidates_dir,
            archive,
            args.workers,
            args.status_interval,
            args.epochs,
            args.batch_size,
            args.cpu,
            args.gpus,
            args.workers_per_gpu,
        )
        for row in offspring:
            all_results[row["candidate_id"]] = row
        combined = [*population, *offspring]
        population = environmental_selection(combined, args.population_size)
        archive = update_archive(archive, combined, experiment, capacity=30, top_k=3)
        elapsed_total = elapsed_before + time.time() - run_started
        average_phase = elapsed_total / (generation + 1)
        remaining_seconds = average_phase * (args.generations - generation)
        print(
            f"experiment={experiment} generation={generation}/{args.generations} "
            f"evaluated={len(all_results)}/{search_budget(args.population_size, args.generations)} "
            f"elapsed={elapsed_total / 3600:.2f}h eta={remaining_seconds / 3600:.2f}h",
            flush=True,
        )
        write_json(
            {
                "generation": generation,
                "population": population,
                "offspring": offspring,
                "archive": archive,
            },
            generations_dir / f"generation_{generation:03d}.json",
        )
        save_search_state(
            state_path,
            population,
            archive,
            all_results,
            generation,
            rng,
            elapsed_total,
        )
        elapsed_before = elapsed_total
        run_started = time.time()

    write_json(archive, experiment_dir / "archive.json")
    pareto = write_pareto_outputs(list(all_results.values()), experiment_dir)
    total_seconds = elapsed_before + time.time() - run_started
    write_json(
        {
            "experiment": experiment,
            "population_size": args.population_size,
            "completed_generations": args.generations,
            "planned_evaluation_slots": search_budget(args.population_size, args.generations),
            "unique_evaluated_candidates": len(all_results),
            "pareto_candidates": len(pareto),
            "total_seconds": total_seconds,
        },
        experiment_dir / "search_summary.json",
    )
    print(
        f"experiment {experiment} complete: evaluated={len(all_results)} "
        f"pareto={len(pareto)} total={total_seconds / 3600:.2f}h",
        flush=True,
    )
    return pareto


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--gpus", type=parse_gpu_list)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--status-interval", type=int, default=60)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be at least 1")
    if args.cpu and args.gpus:
        parser.error("--cpu and --gpus cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    experiments = [3, 1, 2] if args.experiment == "all" else [int(args.experiment)]
    for experiment in experiments:
        run_search(args, experiment)


if __name__ == "__main__":
    main()
