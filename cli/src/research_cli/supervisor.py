"""Parallel local experiment runs: one worker subprocess per pending group, within a thread and memory budget.

The supervisor process only syncs, plans and reads the database; it never initialises a JAX backend.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

from experiment_definition.db import DatabaseManager, RunBatch
from experiment_definition.records import ExecutionArtifactRow
from experiment_definition.status import ExecutionStatus
from research_runner.types import ExperimentSpec

from research_cli.parallel_policy import (
    Admission,
    MemInfo,
    RunningWorker,
    admit_worker,
    parse_meminfo,
    parse_vm_rss,
    worker_projection,
    worker_threads,
)
from research_cli.reclaim import reclaim_dead

SUPERVISOR_PID_ENV = "RESEARCH_SUPERVISOR_PID"
GIB = 2**30

_PR_SET_PDEATHSIG = 1
_DEFAULT_RESERVE_FRACTION = 0.15
_TICK_SECONDS = 1.0
_TERMINATE_GRACE_SECONDS = 10.0
_COMPILATION_CACHE_DIR = Path("~/.cache/research-jax")


@dataclass(frozen=True)
class PendingGroup:
    key: str
    pending_runs: int


def pending_groups(batches: Sequence[RunBatch]) -> list[PendingGroup]:
    """Pending runs per group key, most pending first."""
    counts: dict[str, int] = {}
    for batch in batches:
        counts[batch.group_key] = counts.get(batch.group_key, 0) + len(batch.run_ids)
    ordered = sorted(counts.items(), key=_most_pending_first)
    return [PendingGroup(key, runs) for key, runs in ordered]


def _most_pending_first(item: tuple[str, int]) -> int:
    return -item[1]


@dataclass(frozen=True)
class SupervisorSettings:
    spec_file: Path
    spec_name: str
    results_root: str | None
    budget: int
    cores_per_worker: int | None
    max_runs: int | None
    worker_memory_bytes: int
    memory_reserve_bytes: int | None
    only: Sequence[str]
    only_clauses: Mapping[str, str] | None


@dataclass(frozen=True)
class _Worker:
    group: PendingGroup
    threads: int
    process: subprocess.Popen[bytes]
    started_at: datetime


class _StopRequest:
    def __init__(self) -> None:
        self.signal: int | None = None
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._request)

    def _request(self, signum: int, _frame: FrameType | None) -> None:
        self.signal = signum


def die_with_supervisor() -> None:
    """Make a worker receive SIGTERM when the supervisor that started it dies, even by SIGKILL."""
    supervisor_pid = os.environ.get(SUPERVISOR_PID_ENV)
    if supervisor_pid is None:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    # A supervisor that died before prctl took effect never signals; its orphan has been reparented.
    if os.getppid() != int(supervisor_pid):
        raise SystemExit("supervisor exited before this worker started")


def supervise(experiment_spec: ExperimentSpec, settings: SupervisorSettings) -> int:
    """Run every pending group of the spec in its own worker process; the exit code is nonzero if any failed."""
    experiment_id = _prepare(experiment_spec, settings)
    with DatabaseManager(experiment_spec.db_path) as database:
        batches = database.list_unsatisfied_run_batches(experiment_id, only=settings.only_clauses)
    return _Supervisor(experiment_spec, settings, experiment_id).run(pending_groups(batches))


def _prepare(experiment_spec: ExperimentSpec, settings: SupervisorSettings) -> int:
    experiment_spec.db_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_spec.sync(experiment_spec.db_path)
    with DatabaseManager(experiment_spec.db_path) as database:
        database.initialize()
        experiment = database.get_experiment(experiment_spec.experiment.name)
        if experiment is None:
            raise RuntimeError(f"Experiment {experiment_spec.experiment.name!r} was not synced.")
        _log(reclaim_dead(database, experiment.id))
    return experiment.id


class _Supervisor:
    def __init__(self, experiment_spec: ExperimentSpec, settings: SupervisorSettings, experiment_id: int) -> None:
        self._db_path = experiment_spec.db_path
        self._settings = settings
        self._experiment_id = experiment_id
        self._running: list[_Worker] = []
        self._failed = 0
        self._projection = self._read_projection()

    def run(self, groups: list[PendingGroup]) -> int:
        _log(f"supervisor: {len(groups)} group(s), {sum(g.pending_runs for g in groups)} pending run(s), budget {self._settings.budget}")
        waiting = deque(groups)
        stop = _StopRequest()
        reserve = self._settings.memory_reserve_bytes
        if reserve is None:
            reserve = int(_DEFAULT_RESERVE_FRACTION * _read_meminfo().total_bytes)

        while (waiting or self._running) and stop.signal is None:
            admission = self._admit(waiting, reserve)
            if admission is Admission.IMPOSSIBLE:
                return 1
            if not self._reap() and admission is not Admission.ADMIT:
                time.sleep(_TICK_SECONDS)

        if stop.signal is not None:
            _log(f"supervisor: received {signal.Signals(stop.signal).name}, stopping {len(self._running)} worker(s)")
            self._terminate()
            return 128 + stop.signal
        _log(f"supervisor: done, {self._failed} worker(s) failed")
        return 1 if self._failed else 0

    def _admit(self, waiting: deque[PendingGroup], reserve: int) -> Admission | None:
        """Start the next waiting group if threads and memory allow; None when the thread budget is spent."""
        threads = worker_threads(
            self._settings.budget,
            [worker.threads for worker in self._running],
            len(waiting),
            fixed=self._settings.cores_per_worker,
        )
        if threads is None:
            return None
        available = _read_meminfo().available_bytes
        running = [RunningWorker(self._projection, _rss_bytes(worker)) for worker in self._running]
        admission = admit_worker(available, reserve, self._projection, running)
        if admission is Admission.IMPOSSIBLE:
            _log(
                f"error: a worker is projected to need {self._projection / GIB:.2f} GiB but only "
                f"{available / GIB:.2f} GiB is available with {reserve / GIB:.2f} GiB reserved",
                err=True,
            )
        if admission is Admission.ADMIT:
            self._running.append(self._start(waiting.popleft(), threads))
        return admission

    def _start(self, group: PendingGroup, threads: int) -> _Worker:
        settings = self._settings
        command = [
            sys.executable, "-m", "research_cli.main", "experiment", "run", str(settings.spec_file.resolve()),
            "--spec", settings.spec_name, "--group", group.key,
        ]
        if settings.max_runs is not None:
            command += ["--max-runs", str(settings.max_runs)]
        if settings.results_root is not None:
            command += ["--results-root", settings.results_root]
        for clause in settings.only:
            command += ["--only", clause]
        env = {
            **os.environ,
            "NPROC": str(threads),
            "JAX_PLATFORMS": "cpu",
            "JAX_COMPILATION_CACHE_DIR": str(_COMPILATION_CACHE_DIR.expanduser()),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1.0",
            SUPERVISOR_PID_ENV: str(os.getpid()),
        }
        started_at = datetime.now(timezone.utc)
        # The worker sets its own parent-death signal at startup, which avoids a preexec_fn in a
        # process whose imported libraries may already run threads.
        process = subprocess.Popen(command, env=env, process_group=0)
        _log(f"worker start group={group.key} pending={group.pending_runs} nproc={threads} pid={process.pid}")
        return _Worker(group, threads, process, started_at)

    def _reap(self) -> bool:
        finished = [worker for worker in self._running if worker.process.poll() is not None]
        for worker in finished:
            self._running.remove(worker)
            self._finish(worker)
        if finished:
            self._projection = self._read_projection()
        return bool(finished)

    def _terminate(self) -> None:
        for worker in self._running:
            _signal_group(worker, signal.SIGTERM)
        deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        for worker in self._running:
            try:
                worker.process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _signal_group(worker, signal.SIGKILL)
                worker.process.wait()
        for worker in self._running:
            self._finish(worker)
        self._running.clear()

    def _finish(self, worker: _Worker) -> None:
        returncode = worker.process.returncode
        line = (
            f"worker exit group={worker.group.key} pending={worker.group.pending_runs} "
            f"nproc={worker.threads} pid={worker.process.pid} status={_describe_exit(returncode)}"
        )
        if returncode != 0:
            self._failed += 1
            with DatabaseManager(self._db_path) as database:
                failed = database.fail_group_executions_started_since(
                    self._experiment_id, worker.group.key, worker.started_at
                )
            line += f" failed_executions={failed}"
        _log(line)

    def _read_projection(self) -> int:
        with DatabaseManager(self._db_path) as database:
            executions = database.find_executions_for_experiment(self._experiment_id)
            artifacts = [
                database.get_execution_artifacts(execution.id)
                for execution in executions
                if execution.status == ExecutionStatus.COMPLETED
            ]
        peaks = [peak for peak in map(_reported_peak_rss, artifacts) if peak is not None]
        return worker_projection(peaks, self._settings.worker_memory_bytes)


def _reported_peak_rss(artifacts: ExecutionArtifactRow | None) -> int | None:
    if artifacts is None or artifacts.metadata_json is None:
        return None
    metadata = json.loads(artifacts.metadata_json)
    runtime = metadata.get("runtime") if isinstance(metadata, dict) else None
    peak = runtime.get("peak_rss_bytes") if isinstance(runtime, dict) else None
    return peak if isinstance(peak, int) else None


def _read_meminfo() -> MemInfo:
    return parse_meminfo(Path("/proc/meminfo").read_bytes())


def _rss_bytes(worker: _Worker) -> int:
    try:
        return parse_vm_rss(Path(f"/proc/{worker.process.pid}/status").read_bytes())
    except FileNotFoundError:
        return 0


def _signal_group(worker: _Worker, signum: signal.Signals) -> None:
    try:
        os.killpg(worker.process.pid, signum)
    except ProcessLookupError:
        pass


def _describe_exit(returncode: int) -> str:
    return f"signal {signal.Signals(-returncode).name}" if returncode < 0 else str(returncode)


def _log(message: str, *, err: bool = False) -> None:
    print(message, file=sys.stderr if err else sys.stdout, flush=True)
