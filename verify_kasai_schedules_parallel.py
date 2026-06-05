#!/usr/bin/env python3
"""One-off parallel verifier for the Kasai schedules in tests/test_kasai.py.

This intentionally does not import macroscheduler.qecc.kasai.  The edge formulas below are
the sparse equivalent of get_qubit_pairs_kasai() for the fixed test instance.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import combinations, permutations
from math import comb, factorial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


L = 12
J = 3
P = 768
L_OVER_2 = L // 2

F_PARAMS = (
    (763, 435),
    (679, 69),
    (397, 330),
    (61, 18),
    (697, 612),
    (373, 246),
)
G_PARAMS = (
    (289, 496),
    (257, 640),
    (625, 200),
    (41, 524),
    (193, 672),
    (449, 672),
)

NUM_X_CHECK = J * P
NUM_Z_CHECK = J * P
NUM_DATA_Q = L * P
DEPTH = L
MID_SIZE = L_OVER_2 // 2
SCHEDULES_PER_TASK = factorial(MID_SIZE) ** 3 * factorial(L_OVER_2)
TOTAL_SCHEDULES = comb(L_OVER_2, MID_SIZE) * factorial(MID_SIZE) ** 4 * factorial(L_OVER_2) * 2

ALL_MID_PERMS = tuple(permutations(range(L_OVER_2)))

# Label index layout:
#   0..5   XF0..XF5
#   6..11  XG0..XG5
#   12..17 ZF0..ZF5
#   18..23 ZG0..ZG5
LABEL_PREFIXES = ("XF", "XG", "ZF", "ZG")
NUM_LABELS = 4 * L_OVER_2

_CONFLICT_PAIRS: Tuple[Tuple[int, int], ...] = ()
_ORDERING_CONSTRAINTS: Tuple[Tuple[Tuple[int, int], ...], ...] = ()


@dataclass(frozen=True)
class Task:
    task_id: int
    section_a: Tuple[int, ...]
    section_b: Tuple[int, ...]
    middle: str
    early_x: Tuple[int, ...]
    stop_on_invalid: bool


@dataclass(frozen=True)
class TaskResult:
    task_id: int
    checked: int
    invalid_count: int
    first_failure: Optional[Dict[str, object]]


def label_index(check_type: str, mat_type: str, mat_idx: int) -> int:
    if check_type == "X" and mat_type == "F":
        return mat_idx
    if check_type == "X" and mat_type == "G":
        return L_OVER_2 + mat_idx
    if check_type == "Z" and mat_type == "F":
        return 2 * L_OVER_2 + mat_idx
    if check_type == "Z" and mat_type == "G":
        return 3 * L_OVER_2 + mat_idx
    raise ValueError(f"bad label components: {check_type=} {mat_type=} {mat_idx=}")


def label_name(idx: int) -> str:
    prefix = LABEL_PREFIXES[idx // L_OVER_2]
    return f"{prefix}{idx % L_OVER_2}"


def iter_qubit_pairs(check_type: str, mat_type: str, mat_idx: int) -> Iterable[Tuple[int, int]]:
    """Yield (check_no, qubit_no) pairs for one Kasai affine-permutation label.

    For Perm.to_matrix(), mat[(a * col + b) % P, col] = 1.  The transposed
    case swaps those raw row/column coordinates.
    """
    if mat_type == "F":
        a, b = F_PARAMS[mat_idx]
    elif mat_type == "G":
        a, b = G_PARAMS[mat_idx]
    else:
        raise ValueError("mat_type must be F or G")

    if check_type == "X":
        transposed = False
        base_col_offset = 0 if mat_type == "F" else L_OVER_2 * P
    elif check_type == "Z":
        transposed = True
        base_col_offset = 0 if mat_type == "G" else L_OVER_2 * P
    else:
        raise ValueError("check_type must be X or Z")

    for col in range(P):
        image = (a * col + b) % P
        if transposed:
            raw_row = col
            raw_col = image
        else:
            raw_row = image
            raw_col = col

        for block_row in range(J):
            if check_type == "X":
                block_col = (mat_idx + block_row) % L_OVER_2
            else:
                block_col = (block_row - mat_idx) % L_OVER_2

            check_no = block_row * P + raw_row
            qubit_no = base_col_offset + block_col * P + raw_col
            yield check_no, qubit_no


def build_constraints() -> Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[Tuple[int, int], ...], ...]]:
    labels_by_qubit: List[List[int]] = [[] for _ in range(NUM_DATA_Q)]
    x_by_qubit: List[List[Tuple[int, int]]] = [[] for _ in range(NUM_DATA_Q)]
    z_by_qubit: List[List[Tuple[int, int]]] = [[] for _ in range(NUM_DATA_Q)]
    edge_count_by_label = [0] * NUM_LABELS

    for check_type in ("X", "Z"):
        for mat_type in ("F", "G"):
            for mat_idx in range(L_OVER_2):
                idx = label_index(check_type, mat_type, mat_idx)
                for check_no, qubit_no in iter_qubit_pairs(check_type, mat_type, mat_idx):
                    if not 0 <= qubit_no < NUM_DATA_Q:
                        raise RuntimeError(f"{label_name(idx)} produced bad qubit {qubit_no}")
                    edge_count_by_label[idx] += 1
                    labels_by_qubit[qubit_no].append(idx)
                    if check_type == "X":
                        if not 0 <= check_no < NUM_X_CHECK:
                            raise RuntimeError(f"{label_name(idx)} produced bad X check {check_no}")
                        x_by_qubit[qubit_no].append((check_no, idx))
                    else:
                        if not 0 <= check_no < NUM_Z_CHECK:
                            raise RuntimeError(f"{label_name(idx)} produced bad Z check {check_no}")
                        z_by_qubit[qubit_no].append((check_no, idx))

    expected_edges_per_label = J * P
    bad_counts = [
        (label_name(idx), count)
        for idx, count in enumerate(edge_count_by_label)
        if count != expected_edges_per_label
    ]
    if bad_counts:
        raise RuntimeError(f"bad edge counts by label: {bad_counts}")

    conflict_pairs = set()
    for labels in labels_by_qubit:
        for left, right in combinations(sorted(set(labels)), 2):
            conflict_pairs.add((left, right))

    pair_terms: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for qubit_no in range(NUM_DATA_Q):
        for x_check, x_label in x_by_qubit[qubit_no]:
            for z_check, z_label in z_by_qubit[qubit_no]:
                pair_terms[(x_check, z_check)].append((x_label, z_label))

    ordering_constraints = {
        tuple(sorted(terms))
        for terms in pair_terms.values()
        if len(terms) >= 2
    }

    return tuple(sorted(conflict_pairs)), tuple(sorted(ordering_constraints))


def init_worker(
    conflict_pairs: Tuple[Tuple[int, int], ...],
    ordering_constraints: Tuple[Tuple[Tuple[int, int], ...], ...],
) -> None:
    global _CONFLICT_PAIRS, _ORDERING_CONSTRAINTS
    _CONFLICT_PAIRS = conflict_pairs
    _ORDERING_CONSTRAINTS = ordering_constraints


def assign_outside(stages: List[int], base: int, early: Sequence[int], late: Sequence[int]) -> None:
    for stage, idx in enumerate(early):
        stages[base + idx] = stage
    for offset, idx in enumerate(late):
        stages[base + idx] = MID_SIZE + L_OVER_2 + offset


def assign_middle(stages: List[int], x_base: int, z_base: int, mid_perm: Sequence[int]) -> None:
    for offset, idx in enumerate(mid_perm):
        stage = MID_SIZE + offset
        stages[x_base + idx] = stage
        stages[z_base + idx] = stage


def build_stage_map(
    middle: str,
    early_x: Sequence[int],
    early_z: Sequence[int],
    late_x: Sequence[int],
    late_z: Sequence[int],
    mid_perm: Sequence[int],
) -> List[int]:
    stages = [-1] * NUM_LABELS
    if middle == "G":
        assign_outside(stages, 0, early_x, late_x)  # XF
        assign_outside(stages, 2 * L_OVER_2, early_z, late_z)  # ZF
        assign_middle(stages, L_OVER_2, 3 * L_OVER_2, mid_perm)  # XG/ZG
    elif middle == "F":
        assign_outside(stages, L_OVER_2, early_x, late_x)  # XG
        assign_outside(stages, 3 * L_OVER_2, early_z, late_z)  # ZG
        assign_middle(stages, 0, 2 * L_OVER_2, mid_perm)  # XF/ZF
    else:
        raise ValueError("middle must be F or G")
    return stages


def validate_stage_map(stages: Sequence[int]) -> Optional[Dict[str, object]]:
    if len(stages) != NUM_LABELS:
        return {"kind": "shape", "message": f"expected {NUM_LABELS} labels, got {len(stages)}"}

    min_stage = min(stages)
    max_stage = max(stages)
    if min_stage < 0:
        unassigned = [label_name(idx) for idx, stage in enumerate(stages) if stage < 0]
        return {"kind": "unassigned", "labels": unassigned}
    if max_stage >= DEPTH:
        out_of_range = [
            (label_name(idx), stage)
            for idx, stage in enumerate(stages)
            if stage < 0 or stage >= DEPTH
        ]
        return {"kind": "stage_range", "labels": out_of_range, "depth": DEPTH}

    inferred_depth = max_stage + 1
    if inferred_depth != DEPTH:
        return {"kind": "depth", "inferred_depth": inferred_depth, "expected_depth": DEPTH}

    for left, right in _CONFLICT_PAIRS:
        if stages[left] == stages[right]:
            return {
                "kind": "qubit_stage_conflict",
                "left": label_name(left),
                "right": label_name(right),
                "stage": stages[left],
            }

    for constraint in _ORDERING_CONSTRAINTS:
        negative_count = 0
        for x_label, z_label in constraint:
            diff = stages[x_label] - stages[z_label]
            if diff < 0:
                negative_count += 1
        if negative_count % 2 == 1:
            diffs = [
                (label_name(x_label), label_name(z_label), stages[x_label] - stages[z_label])
                for x_label, z_label in constraint
            ]
            return {
                "kind": "ordering",
                "terms": diffs,
            }

    return None


def failure_descriptor(
    task: Task,
    checked: int,
    early_z: Sequence[int],
    late_x: Sequence[int],
    late_z: Sequence[int],
    mid_perm: Sequence[int],
    violation: Dict[str, object],
) -> Dict[str, object]:
    return {
        "task_id": task.task_id,
        "checked_in_task": checked,
        "section_a": task.section_a,
        "section_b": task.section_b,
        "middle": task.middle,
        "early_x": task.early_x,
        "late_x": tuple(late_x),
        "early_z": tuple(early_z),
        "late_z": tuple(late_z),
        "mid_perm": tuple(mid_perm),
        "violation": violation,
    }


def run_task(task: Task) -> TaskResult:
    checked = 0
    invalid_count = 0
    first_failure = None

    late_x_perms = tuple(permutations(task.section_b))
    early_z_perms = tuple(permutations(task.section_b))
    late_z_perms = tuple(permutations(task.section_a))

    for late_x in late_x_perms:
        for early_z in early_z_perms:
            for late_z in late_z_perms:
                for mid_perm in ALL_MID_PERMS:
                    stages = build_stage_map(
                        task.middle,
                        task.early_x,
                        early_z,
                        late_x,
                        late_z,
                        mid_perm,
                    )
                    checked += 1
                    violation = validate_stage_map(stages)
                    if violation is not None:
                        invalid_count += 1
                        if first_failure is None:
                            first_failure = failure_descriptor(
                                task,
                                checked,
                                early_z,
                                late_x,
                                late_z,
                                mid_perm,
                                violation,
                            )
                        if task.stop_on_invalid:
                            return TaskResult(task.task_id, checked, invalid_count, first_failure)

    return TaskResult(task.task_id, checked, invalid_count, first_failure)


def make_tasks(stop_on_invalid: bool) -> List[Task]:
    tasks = []
    task_id = 0
    for section_a in combinations(range(L_OVER_2), MID_SIZE):
        section_a = tuple(section_a)
        section_b = tuple(idx for idx in range(L_OVER_2) if idx not in section_a)
        for middle in ("G", "F"):
            for early_x in permutations(section_a):
                tasks.append(Task(task_id, section_a, section_b, middle, tuple(early_x), stop_on_invalid))
                task_id += 1
    return tasks


def format_int(value: int) -> str:
    return f"{value:,}"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def print_progress(
    checked: int,
    total: int,
    completed_chunks: int,
    total_chunks: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    remaining = max(total - checked, 0)
    rate = checked / elapsed
    eta = remaining / rate if rate > 0 else 0
    percent = (checked / total * 100) if total else 100.0
    print(
        "[progress] "
        f"checked {format_int(checked)} / {format_int(total)} "
        f"({percent:.2f}%); remaining {format_int(remaining)}; "
        f"rate {format_int(int(rate))} sched/s; eta {format_duration(eta)}; "
        f"chunks {completed_chunks}/{total_chunks}",
        flush=True,
    )


def format_violation(violation: Dict[str, object]) -> str:
    kind = violation.get("kind")
    if kind == "qubit_stage_conflict":
        return (
            "qubit-stage conflict: "
            f"{violation['left']} and {violation['right']} both at stage {violation['stage']}"
        )
    if kind == "ordering":
        terms = violation["terms"]
        rendered = ", ".join(f"({x}-{z})={diff}" for x, z, diff in terms)  # type: ignore[misc]
        return f"ordering product is negative: {rendered}"
    if kind == "unassigned":
        labels = ", ".join(violation["labels"])  # type: ignore[arg-type]
        return f"unassigned labels: {labels}"
    if kind == "stage_range":
        return f"stage out of range for depth {violation['depth']}: {violation['labels']}"
    if kind == "depth":
        return f"inferred depth {violation['inferred_depth']} != expected {violation['expected_depth']}"
    return repr(violation)


def print_failure(failure: Dict[str, object]) -> None:
    print("\nINVALID schedule found", file=sys.stderr)
    print(f"  task_id: {failure['task_id']}", file=sys.stderr)
    print(f"  checked_in_task: {failure['checked_in_task']}", file=sys.stderr)
    print(f"  section_a: {failure['section_a']}", file=sys.stderr)
    print(f"  section_b: {failure['section_b']}", file=sys.stderr)
    print(f"  middle: {failure['middle']}", file=sys.stderr)
    print(f"  early_x: {failure['early_x']}", file=sys.stderr)
    print(f"  late_x: {failure['late_x']}", file=sys.stderr)
    print(f"  early_z: {failure['early_z']}", file=sys.stderr)
    print(f"  late_z: {failure['late_z']}", file=sys.stderr)
    print(f"  mid_perm: {failure['mid_perm']}", file=sys.stderr)
    print(f"  violation: {format_violation(failure['violation'])}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify all Kasai schedules from tests/test_kasai.py in parallel."
    )
    parser.add_argument("--workers", type=int, default=14, help="worker processes to use (default: 14)")
    parser.add_argument(
        "--status-interval",
        type=float,
        default=10.0,
        help="seconds between progress reports (default: 10)",
    )
    parser.add_argument(
        "--stop-on-invalid",
        dest="stop_on_invalid",
        action="store_true",
        default=True,
        help="stop after the first invalid schedule (default)",
    )
    parser.add_argument(
        "--no-stop-on-invalid",
        dest="stop_on_invalid",
        action="store_false",
        help="continue checking after invalid schedules and report the first example",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="limit number of chunks for smoke tests; one task is 155,520 schedules",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        print("--workers must be positive", file=sys.stderr)
        return 2
    if args.status_interval <= 0:
        print("--status-interval must be positive", file=sys.stderr)
        return 2
    if args.max_tasks is not None and args.max_tasks <= 0:
        print("--max-tasks must be positive when provided", file=sys.stderr)
        return 2

    print("Building compact Kasai validation constraints...", flush=True)
    constraint_started_at = time.monotonic()
    conflict_pairs, ordering_constraints = build_constraints()
    constraint_elapsed = time.monotonic() - constraint_started_at

    tasks = make_tasks(args.stop_on_invalid)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    total_tasks = len(tasks)
    total_schedules = total_tasks * SCHEDULES_PER_TASK
    expected_total = TOTAL_SCHEDULES if args.max_tasks is None else total_schedules

    print("Kasai schedule verifier")
    print(f"  pid: {os.getpid()}")
    print(f"  workers: {args.workers}")
    print(f"  tasks: {total_tasks}")
    print(f"  schedules/task: {format_int(SCHEDULES_PER_TASK)}")
    print(f"  total schedules to check: {format_int(total_schedules)}")
    print(f"  full schedule count: {format_int(TOTAL_SCHEDULES)}")
    print(f"  conflict constraints: {format_int(len(conflict_pairs))}")
    print(f"  ordering constraints: {format_int(len(ordering_constraints))}")
    print(f"  constraint build time: {constraint_elapsed:.3f}s", flush=True)

    if expected_total != total_schedules:
        print(
            f"Internal count mismatch: expected {expected_total}, got {total_schedules}",
            file=sys.stderr,
        )
        return 2

    checked = 0
    invalid_count = 0
    completed_chunks = 0
    first_failure = None
    started_at = time.monotonic()
    next_report_at = started_at + args.status_interval
    executor: Optional[ProcessPoolExecutor] = None

    try:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(conflict_pairs, ordering_constraints),
        )
        pending = {executor.submit(run_task, task) for task in tasks}

        while pending:
            timeout = max(0.0, next_report_at - time.monotonic())
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)

            now = time.monotonic()
            if not done:
                print_progress(checked, total_schedules, completed_chunks, total_tasks, started_at)
                next_report_at = now + args.status_interval
                continue

            for future in done:
                result = future.result()
                checked += result.checked
                invalid_count += result.invalid_count
                completed_chunks += 1
                if first_failure is None and result.first_failure is not None:
                    first_failure = result.first_failure

            if first_failure is not None and args.stop_on_invalid:
                for future in pending:
                    future.cancel()
                print_progress(checked, total_schedules, completed_chunks, total_tasks, started_at)
                print_failure(first_failure)
                executor.shutdown(wait=False, cancel_futures=True)
                executor = None
                return 1

            if now >= next_report_at:
                print_progress(checked, total_schedules, completed_chunks, total_tasks, started_at)
                next_report_at = now + args.status_interval

    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    elapsed = time.monotonic() - started_at
    print_progress(checked, total_schedules, completed_chunks, total_tasks, started_at)

    if invalid_count:
        if first_failure is not None:
            print_failure(first_failure)
        print(
            f"FAILED: checked {format_int(checked)} / {format_int(total_schedules)} schedules; "
            f"invalid schedules found: {format_int(invalid_count)}; elapsed {format_duration(elapsed)}",
            file=sys.stderr,
        )
        return 1

    bounded = " (bounded run)" if args.max_tasks is not None else ""
    print(
        f"SUCCESS{bounded}: checked {format_int(checked)} / {format_int(total_schedules)} "
        f"schedules; invalid schedules: 0; elapsed {format_duration(elapsed)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
