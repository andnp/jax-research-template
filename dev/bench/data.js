window.BENCHMARK_DATA = {
  "lastUpdate": 1774744869542,
  "repoUrl": "https://github.com/andnp/jax-research-template",
  "entries": {
    "Env Seam Benchmark": [
      {
        "commit": {
          "author": {
            "email": "andnpatterson@gmail.com",
            "name": "Andy Patterson",
            "username": "andnp"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e40824147e7b0840ec0987844f98ac325baf5363",
          "message": "feat(rl-components): add jaxatari environment substrate (#35)\n\n## Summary\n- add a canonical environment protocol in `rl-components`\n- add a JAXAtari adapter with DeepMind-style preprocessing defaults\n- add real JAXAtari smoke coverage in CI with scripted asset\ninstallation\n- add a Gymnax compatibility bridge for incremental consumer adoption\n- add benchmark-based env-seam regression detection in CI\n\n## Details\n- introduced `EnvSpec`, `EnvReset`, `EnvStep`, and `EnvProtocol`\n- pinned `jaxatari` to GitHub tag `v0.1`\n- added `JAXAtariAdapter` and `JAXAtariConfig` in `rl_components.atari`\n- added a real smoke test plus `scripts/install_jaxatari_assets.py`\n- added `GymnaxCompatibilityBridge` to let current Gymnax-style\nconsumers adopt canonical envs incrementally\n- added a dedicated benchmark job using `pytest-benchmark` and\n`benchmark-action/github-action-benchmark`\n\n## Verification\n- `uv run pytest tests/small/test_rl_components_env_protocol.py -q`\n- `uv run pytest tests/small/test_rl_components_atari.py\ntests/medium/test_rl_components_atari_jit.py -q`\n- `uv run pytest tests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py -q`\n- `uv run pytest tests/regression/test_rl_components_atari_real_smoke.py\n-q`\n- `JAXATARI_CONFIRM_OWNERSHIP=1 XDG_DATA_HOME=<temp> uv run python\nscripts/install_jaxatari_assets.py`\n- `JAXATARI_RUN_SMOKE=1 XDG_DATA_HOME=<same temp> uv run pytest -q\ntests/regression/test_rl_components_atari_real_smoke.py::test_real_jaxatari_adapter_smoke\n-x`\n- `uv run pytest --benchmark-only --benchmark-json output.json\ntests/performance/test_rl_components_gymnax_bridge_bench.py -q`\n- `uv run ruff check\nlibs/rl-components/src/rl_components/env_protocol.py\nlibs/rl-components/src/rl_components/atari.py\nlibs/rl-components/src/rl_components/gymnax_bridge.py\ntests/small/test_rl_components_env_protocol.py\ntests/small/test_rl_components_atari.py\ntests/medium/test_rl_components_atari_jit.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/regression/test_rl_components_atari_real_smoke.py\ntests/performance/test_rl_components_gymnax_bridge_bench.py\nscripts/install_jaxatari_assets.py`\n- `uv run pyright libs/rl-components/src/rl_components/env_protocol.py\nlibs/rl-components/src/rl_components/atari.py\nlibs/rl-components/src/rl_components/gymnax_bridge.py\ntests/small/test_rl_components_env_protocol.py\ntests/small/test_rl_components_atari.py\ntests/medium/test_rl_components_atari_jit.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/regression/test_rl_components_atari_real_smoke.py\ntests/performance/test_rl_components_gymnax_bridge_bench.py\nscripts/install_jaxatari_assets.py`\n- `uv run ty check libs/rl-components/src/rl_components/env_protocol.py\nlibs/rl-components/src/rl_components/atari.py\nlibs/rl-components/src/rl_components/gymnax_bridge.py\ntests/small/test_rl_components_atari.py\ntests/medium/test_rl_components_atari_jit.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/performance/test_rl_components_gymnax_bridge_bench.py`\n\nCloses #10",
          "timestamp": "2026-03-28T18:28:27-06:00",
          "tree_id": "3a33b24afd2cfadda13cb9f7347fc0fbfb545f14",
          "url": "https://github.com/andnp/jax-research-template/commit/e40824147e7b0840ec0987844f98ac325baf5363"
        },
        "date": 1774744869206,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.931968507337367,
            "unit": "iter/sec",
            "range": "stddev: 0.0014308369383294224",
            "extra": "mean: 168.5781033333337 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.0006092032359621,
            "unit": "iter/sec",
            "range": "stddev: 0.005854054825397738",
            "extra": "mean: 999.3911676666655 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04079770579160397,
            "unit": "iter/sec",
            "range": "stddev: 0.026741334552431644",
            "extra": "mean: 24.511182199999993 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.4239184512870633,
            "unit": "iter/sec",
            "range": "stddev: 0.009709656865519447",
            "extra": "mean: 2.3589442661999955 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 31793.671777098723,
            "unit": "iter/sec",
            "range": "stddev: 0.000010957575150028898",
            "extra": "mean: 31.45280001035644 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 39053.651908386266,
            "unit": "iter/sec",
            "range": "stddev: 0.000010752432182486635",
            "extra": "mean: 25.60579999908441 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 26589.66300944416,
            "unit": "iter/sec",
            "range": "stddev: 0.000013548095483913409",
            "extra": "mean: 37.60859999033528 usec\nrounds: 5"
          }
        ]
      }
    ]
  }
}