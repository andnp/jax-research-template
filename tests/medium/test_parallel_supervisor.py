"""End-to-end tests of `research experiment run --workers` on a toy spec with three groups.

The supervisor runs as a real subprocess, because process groups, signals and the parent-death
signal are the behavior under test.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
from experiment_definition.db import DatabaseManager

_DEADLINE_SECONDS = 180.0
_BUDGET = 4
_WORKER_START = re.compile(r"worker start group=\S+ pending=\d+ nproc=\d+ pid=(\d+)")


def _write_spec(tmp_path: Path) -> Path:
    spec_file = tmp_path / "toy_spec.py"
    spec_file.write_text(
        textwrap.dedent(f"""\
            import os
            import signal
            import time
            from dataclasses import dataclass
            from pathlib import Path

            from experiment_definition.component import Component, ComponentType
            from experiment_definition.experiment import Experiment
            from research_runner.registry import spec
            from research_runner.types import ExperimentSpec


            @dataclass(frozen=True)
            class Toy:
                learning_rate: float = 0.1


            def train(config: Toy, key):
                # Runs while the batch is traced, so a Python side effect happens once per batch.
                marker = Path(os.environ.get("TOY_KILL_MARKER", "{tmp_path}/unused"))
                if str(config.learning_rate) == os.environ.get("TOY_KILL_LR") and not marker.exists():
                    marker.touch()
                    os.kill(os.getpid(), signal.SIGKILL)
                time.sleep(float(os.environ["TOY_SLEEP"]))
                return None, {{}}


            @spec
            def toy() -> ExperimentSpec:
                experiment = Experiment("toy")
                with experiment.for_component(Component(name="algo", path=Path("algo.py"), type=ComponentType.ALGO)):
                    experiment.add_parameter("learning_rate", [0.1, 0.2, 0.3])
                with experiment.for_component(Component(name="env", path=Path("env.py"), type=ComponentType.ENV)):
                    pass
                experiment.add_parameter("seed", [0, 1])
                return ExperimentSpec(
                    experiment, config=Toy(), train=train, results_root=Path("{tmp_path}/results"), capture_git=False
                )
        """),
    )
    return spec_file


@dataclass
class _Supervisor:
    process: subprocess.Popen[bytes]
    log: Path

    def output(self) -> str:
        return self.log.read_text()

    def worker_pids(self) -> list[int]:
        return [int(pid) for pid in _WORKER_START.findall(self.output())]

    def wait(self) -> int:
        return self.process.wait(timeout=_DEADLINE_SECONDS)


@dataclass(frozen=True)
class _Execution:
    status: str
    run_ids: list[int]
    start_time: str | None
    end_time: str | None
    metadata: dict[str, object]


@pytest.fixture
def launch(tmp_path: Path) -> Iterator[Callable[..., _Supervisor]]:
    launched: list[_Supervisor] = []
    spec_file = _write_spec(tmp_path)

    def start(*, sleep: float, **toy_env: str) -> _Supervisor:
        log = tmp_path / f"supervisor-{len(launched)}.log"
        env = {**os.environ, "TOY_SLEEP": str(sleep), **toy_env}
        command = [
            sys.executable, "-m", "research_cli.main", "experiment", "run", str(spec_file),
            "--workers", str(_BUDGET), "--worker-memory", "0.01", "--memory-reserve", "0",
        ]
        with log.open("wb") as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        supervisor = _Supervisor(process, log)
        launched.append(supervisor)
        return supervisor

    yield start
    for supervisor in launched:
        if supervisor.process.poll() is None:
            supervisor.process.kill()
            supervisor.process.wait()


def _wait_until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + _DEADLINE_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def _alive(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return False
    return "\nState:\tZ" not in status


def _executions(tmp_path: Path) -> list[_Execution]:
    db_path = tmp_path / "results" / "experiments.sqlite"
    if not db_path.exists():
        return []
    with DatabaseManager(db_path) as database:
        experiment = database.get_experiment("toy")
        if experiment is None:
            return []
        executions: list[_Execution] = []
        for row in database.list_executions(experiment.id):
            artifacts = database.get_execution_artifacts(row.id)
            metadata = json.loads(artifacts.metadata_json) if artifacts and artifacts.metadata_json else {}
            run_ids = [run.id for run in database.list_execution_runs(row.id)]
            executions.append(_Execution(row.status, run_ids, row.start_time, row.end_time, metadata))
    return executions


def _completed_runs(tmp_path: Path) -> set[int]:
    return {run for e in _executions(tmp_path) if e.status == "COMPLETED" for run in e.run_ids}


def _running_count(tmp_path: Path) -> int:
    return sum(e.status == "RUNNING" for e in _executions(tmp_path))


def _runtime(execution: _Execution) -> dict[str, object]:
    runtime = execution.metadata["runtime"]
    assert isinstance(runtime, dict)
    return runtime


def _peak_concurrent_nproc(executions: list[_Execution]) -> int:
    intervals = [
        (datetime.fromisoformat(str(e.start_time)), datetime.fromisoformat(str(e.end_time)), int(str(_runtime(e)["nproc"])))
        for e in executions
    ]
    return max(sum(nproc for start, end, nproc in intervals if start <= instant <= end) for instant, _, _ in intervals)


class TestCompletion:
    def test_every_run_completes_within_the_thread_budget(self, launch: Callable[..., _Supervisor], tmp_path: Path) -> None:
        """All three groups run at once, each worker sees its NPROC, and concurrent NPROC never exceeds W.

        W = 4 over three groups spreads to 2, 1 and 1 threads.
        """
        supervisor = launch(sleep=1.0)

        assert supervisor.wait() == 0, supervisor.output()

        executions = _executions(tmp_path)
        assert [e.status for e in executions] == ["COMPLETED"] * 3
        assert len(_completed_runs(tmp_path)) == 6
        assert sorted(int(str(_runtime(e)["nproc"])) for e in executions) == [1, 1, 2]
        assert _peak_concurrent_nproc(executions) <= _BUDGET
        output = supervisor.output()
        assert output.count("worker start") == 3
        assert output.rindex("worker start") < output.index("worker exit"), output


class TestShutdown:
    @pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL], ids=["sigterm", "sigkill"])
    def test_stopping_the_supervisor_leaves_no_worker(
        self, launch: Callable[..., _Supervisor], tmp_path: Path, signum: signal.Signals
    ) -> None:
        """SIGTERM stops the workers through their process groups; SIGKILL through the parent-death signal."""
        supervisor = launch(sleep=600.0)
        _wait_until(lambda: _running_count(tmp_path) == 3, "three running executions")
        workers = supervisor.worker_pids()

        supervisor.process.send_signal(signum)

        assert supervisor.wait() != 0
        _wait_until(lambda: not any(_alive(pid) for pid in workers), "every worker to exit")
        assert len(workers) == 3

    def test_sigterm_fails_the_stopped_workers_executions(self, launch: Callable[..., _Supervisor], tmp_path: Path) -> None:
        """Executions a terminated worker left RUNNING become FAILED, so a relaunch replans them."""
        supervisor = launch(sleep=600.0)
        _wait_until(lambda: _running_count(tmp_path) == 3, "three running executions")

        supervisor.process.send_signal(signal.SIGTERM)

        assert supervisor.wait() != 0
        assert [e.status for e in _executions(tmp_path)] == ["FAILED"] * 3


class TestWorkerDeath:
    def test_a_killed_worker_fails_its_execution_and_a_relaunch_completes(
        self, launch: Callable[..., _Supervisor], tmp_path: Path
    ) -> None:
        """A worker that dies by SIGKILL mid-batch fails the sweep, and a plain relaunch finishes its runs."""
        supervisor = launch(sleep=0.1, TOY_KILL_LR="0.2", TOY_KILL_MARKER=str(tmp_path / "killed"))

        assert supervisor.wait() != 0
        statuses = sorted(e.status for e in _executions(tmp_path))
        assert statuses == ["COMPLETED", "COMPLETED", "FAILED"], supervisor.output()
        assert "status=signal SIGKILL" in supervisor.output()

        relaunch = launch(sleep=0.1)

        assert relaunch.wait() == 0, relaunch.output()
        assert len(_completed_runs(tmp_path)) == 6
