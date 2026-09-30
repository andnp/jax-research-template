import pytest
from research_cli.parallel_policy import (
    Admission,
    MemInfo,
    RunningWorker,
    admit_worker,
    parse_meminfo,
    worker_projection,
    worker_threads,
)


def _start_all(budget: int, groups: int) -> list[int]:
    running: list[int] = []
    for waiting in range(groups, 0, -1):
        c = worker_threads(budget, running, waiting)
        assert c is not None
        running.append(c)
    return running


class TestWorkerThreads:
    def test_budget_is_spread_across_groups_with_remainder(self) -> None:
        assert _start_all(8, 3) == [3, 3, 2]
        assert _start_all(8, 2) == [4, 4]

    def test_many_groups_get_one_thread_each(self) -> None:
        assert set(_start_all(11, 11)) == {1}
        assert worker_threads(11, [], 35) == 1

    def test_free_budget_excludes_running_workers(self) -> None:
        assert worker_threads(8, [3, 3], 1) == 2
        assert worker_threads(8, [6], 2) == 1

    def test_full_budget_or_no_waiting_groups_cannot_start(self) -> None:
        assert worker_threads(8, [4, 4], 1) is None
        assert worker_threads(8, [], 0) is None

    def test_max_threads_caps_a_lone_group(self) -> None:
        assert worker_threads(32, [], 1) == 8
        assert worker_threads(32, [], 1, max_threads=4) == 4

    def test_fixed_threads_apply_when_they_fit(self) -> None:
        assert worker_threads(8, [], 5, fixed=3) == 3
        assert worker_threads(8, [3, 3], 1, fixed=3) is None
        assert worker_threads(8, [3, 3], 1, fixed=2) == 2

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            worker_threads(0, [], 1)
        with pytest.raises(ValueError):
            worker_threads(8, [], 1, fixed=0)


class TestAdmission:
    def test_admits_when_projection_fits_above_reserve(self) -> None:
        assert admit_worker(10, 2, 8, []) is Admission.ADMIT
        assert admit_worker(10, 2, 9, [RunningWorker(4, 4)]) is Admission.WAIT

    def test_running_workers_remaining_growth_is_counted(self) -> None:
        running = [RunningWorker(projection_bytes=4, rss_bytes=1)]
        assert admit_worker(10, 0, 7, running) is Admission.ADMIT
        assert admit_worker(10, 0, 8, running) is Admission.WAIT

    def test_worker_above_its_projection_contributes_no_growth(self) -> None:
        running = [RunningWorker(projection_bytes=4, rss_bytes=9)]
        assert admit_worker(10, 0, 10, running) is Admission.ADMIT

    def test_nothing_running_and_no_fit_is_impossible(self) -> None:
        assert admit_worker(10, 2, 9, []) is Admission.IMPOSSIBLE


class TestWorkerProjection:
    def test_largest_reported_peak_wins(self) -> None:
        assert worker_projection([3, 9, 5], default=4) == 9

    def test_default_when_nothing_reported(self) -> None:
        assert worker_projection([], default=4) == 4


class TestParseMeminfo:
    def test_reads_available_and_total_in_bytes(self) -> None:
        text = b"MemTotal:       16000 kB\nMemFree:          100 kB\nMemAvailable:    8000 kB\n"
        assert parse_meminfo(text) == MemInfo(available_bytes=8000 * 1024, total_bytes=16000 * 1024)

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="MemAvailable"):
            parse_meminfo(b"MemTotal: 1 kB\n")
