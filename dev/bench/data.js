window.BENCHMARK_DATA = {
  "lastUpdate": 1774754092012,
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
      },
      {
        "commit": {
          "author": {
            "email": "andnpatterson@gmail.com",
            "name": "Andy Patterson",
            "username": "andnp"
          },
          "committer": {
            "email": "andnpatterson@gmail.com",
            "name": "Andy Patterson",
            "username": "andnp"
          },
          "distinct": true,
          "id": "e073b016e190ade0f62302a2978117d2394e3427",
          "message": "docs(agents): resolve review threads after fixes",
          "timestamp": "2026-03-28T18:29:29-06:00",
          "tree_id": "8631a83b2cece42cc7d698a1ff520c29cbfafcad",
          "url": "https://github.com/andnp/jax-research-template/commit/e073b016e190ade0f62302a2978117d2394e3427"
        },
        "date": 1774744913067,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.085724883917121,
            "unit": "iter/sec",
            "range": "stddev: 0.0006918523959101276",
            "extra": "mean: 164.31896266666968 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.0398359556737415,
            "unit": "iter/sec",
            "range": "stddev: 0.004712943632883759",
            "extra": "mean: 961.6901536666612 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04218924959015812,
            "unit": "iter/sec",
            "range": "stddev: 0.0332754961853468",
            "extra": "mean: 23.70272071 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.4358180878391624,
            "unit": "iter/sec",
            "range": "stddev: 0.009040436154708502",
            "extra": "mean: 2.2945353299999964 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 30746.147512177533,
            "unit": "iter/sec",
            "range": "stddev: 0.000019976983924004307",
            "extra": "mean: 32.52439999528178 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 39380.62157165956,
            "unit": "iter/sec",
            "range": "stddev: 0.000010690780248703867",
            "extra": "mean: 25.39320000778389 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 26280.65638719971,
            "unit": "iter/sec",
            "range": "stddev: 0.000012091646367742316",
            "extra": "mean: 38.05079999779082 usec\nrounds: 5"
          }
        ]
      },
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
          "id": "6ec58dc466e7f376dab0d0668d848c444d3b7d70",
          "message": "Add research doctor CLI health checks (#36)\n\n## Summary\n- define the read-only `research doctor` command contract in the CLI\nspec\n- add typed `research.yaml` loading and validation for doctor config\n- add aggregated config, git, and environment health checks with CLI\ncoverage\n\n## Testing\n- uv run pytest tests/small/test_cli_config.py\ntests/small/test_doctor.py tests/small/test_workspace_init.py\ntests/medium/test_cli_doctor.py -q\n- uv run pyright cli/src/research_cli/config.py\ncli/src/research_cli/doctor.py cli/src/research_cli/main.py\ntests/small/test_cli_config.py tests/small/test_doctor.py\ntests/medium/test_cli_doctor.py\n- uv run ty check cli/src/research_cli/config.py\ncli/src/research_cli/doctor.py cli/src/research_cli/main.py\ntests/small/test_cli_config.py tests/small/test_doctor.py\ntests/medium/test_cli_doctor.py",
          "timestamp": "2026-03-28T19:04:55-06:00",
          "tree_id": "fe492a4eaa799381a5dedb1b12a5740ae999e08b",
          "url": "https://github.com/andnp/jax-research-template/commit/6ec58dc466e7f376dab0d0668d848c444d3b7d70"
        },
        "date": 1774747075931,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.119115435358479,
            "unit": "iter/sec",
            "range": "stddev: 0.0007500372701673495",
            "extra": "mean: 163.422313333335 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.0397187110619044,
            "unit": "iter/sec",
            "range": "stddev: 0.0034956413035032285",
            "extra": "mean: 961.7985993333349 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04206270106541846,
            "unit": "iter/sec",
            "range": "stddev: 0.007951261605168437",
            "extra": "mean: 23.774031973000007 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.43402201251871275,
            "unit": "iter/sec",
            "range": "stddev: 0.011007525561343335",
            "extra": "mean: 2.304030604799993 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 31344.4250742238,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176439941876358",
            "extra": "mean: 31.903600006444318 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 34404.45882237049,
            "unit": "iter/sec",
            "range": "stddev: 0.000016623853290644668",
            "extra": "mean: 29.06599999619175 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 23465.585375447867,
            "unit": "iter/sec",
            "range": "stddev: 0.000018442831919439184",
            "extra": "mean: 42.61559999463316 usec\nrounds: 5"
          }
        ]
      },
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
          "id": "afa1438f97bd45575486458a971c346a23279fa1",
          "message": "Add workspace repair command for broken core submodules (#38)\n\n## Summary\n- define the `research workspace repair` contract and dry-run behavior\n- add repair path/config resolution plus the real submodule repair\nexecution\n- cover dirty submodule recovery with focused small and medium tests\n\n## Testing\n- uv run pytest tests/small/test_workspace_repair.py\ntests/medium/test_cli_workspace_repair.py -q\n- uv run ruff check cli/src/research_cli/workspace.py\ntests/small/test_workspace_repair.py\ntests/medium/test_cli_workspace_repair.py\n- uv run pyright cli/src/research_cli/workspace.py\ntests/small/test_workspace_repair.py\ntests/medium/test_cli_workspace_repair.py\n- uv run ty check cli/src/research_cli/workspace.py\ntests/small/test_workspace_repair.py\ntests/medium/test_cli_workspace_repair.py",
          "timestamp": "2026-03-28T21:11:00-06:00",
          "tree_id": "8d898b7c615f4174050f0a578dd022f9d6e2a43d",
          "url": "https://github.com/andnp/jax-research-template/commit/afa1438f97bd45575486458a971c346a23279fa1"
        },
        "date": 1774754091704,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.840180914632542,
            "unit": "iter/sec",
            "range": "stddev: 0.0006046160898544423",
            "extra": "mean: 171.22757233333394 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9845237826646733,
            "unit": "iter/sec",
            "range": "stddev: 0.011930375758077302",
            "extra": "mean: 1.0157194956666658 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04077394790152377,
            "unit": "iter/sec",
            "range": "stddev: 0.05331801363401701",
            "extra": "mean: 24.525464211000006 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.42312977865545426,
            "unit": "iter/sec",
            "range": "stddev: 0.005283834646366749",
            "extra": "mean: 2.3633411081999953 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 31671.227325728047,
            "unit": "iter/sec",
            "range": "stddev: 0.000015797170111106125",
            "extra": "mean: 31.574399997680302 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 26342.825528787154,
            "unit": "iter/sec",
            "range": "stddev: 0.000030858024875664843",
            "extra": "mean: 37.961000003861045 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 26118.52587360115,
            "unit": "iter/sec",
            "range": "stddev: 0.000011325065059619192",
            "extra": "mean: 38.28699999530727 usec\nrounds: 5"
          }
        ]
      }
    ]
  }
}