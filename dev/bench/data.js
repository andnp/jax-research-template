window.BENCHMARK_DATA = {
  "lastUpdate": 1774825921808,
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
          "id": "45e35f77824b4c70f954eba9cbf01b533ac9284e",
          "message": "feat(rl-agents): add nature dqn preset path (#37)\n\n## Summary\n- add the reusable Nature CNN substrate and legacy DQN initialization\nhelpers in `jax-nn`\n- wire a real `nature_cnn` preset through DQN, add an external env seam,\nand propagate the seam to DQN variants\n- add a JAXAtari-backed DQN example plus a gated regression smoke test\n\n## Testing\n- uv run pytest tests/small/test_jax_nn_initializers.py\ntests/small/test_jax_nn_public_api.py\ntests/small/test_jax_nn_nature_cnn.py\ntests/medium/test_jax_nn_nature_cnn_jit.py\n- uv run pytest tests/small/test_rl_agents_dqn.py\ntests/medium/test_rl_agents_dqn_gradient.py\ntests/medium/test_rl_agents_dqn_nature_env_integration.py\ntests/medium/test_dqn_variants.py\n- uv run pytest tests/small/test_jax_nn_noisy_linear.py\ntests/medium/test_jax_nn_noisy_linear_jit.py\n- uv run pytest tests/regression/test_rl_agents_dqn_atari_real_smoke.py\n-q\n- uv run ruff check .\n- uv run pyright\n\nCloses #11.",
          "timestamp": "2026-03-28T21:24:57-06:00",
          "tree_id": "015b7f8e78cfdd751455718c20144d050315dc04",
          "url": "https://github.com/andnp/jax-research-template/commit/45e35f77824b4c70f954eba9cbf01b533ac9284e"
        },
        "date": 1774754936585,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.8926575701199475,
            "unit": "iter/sec",
            "range": "stddev: 0.00048262967086119433",
            "extra": "mean: 169.70271700000458 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9961750977168474,
            "unit": "iter/sec",
            "range": "stddev: 0.0041289703300077555",
            "extra": "mean: 1.003839588333335 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.041019741939109094,
            "unit": "iter/sec",
            "range": "stddev: 0.01309822539737519",
            "extra": "mean: 24.378505391000004 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.42425075801347445,
            "unit": "iter/sec",
            "range": "stddev: 0.0025422698344685865",
            "extra": "mean: 2.3570965546000027 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 32152.479915852924,
            "unit": "iter/sec",
            "range": "stddev: 0.000015904532808429882",
            "extra": "mean: 31.10180000476248 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 35314.72482642489,
            "unit": "iter/sec",
            "range": "stddev: 0.000013617151053008016",
            "extra": "mean: 28.31680000099368 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 22859.4418645983,
            "unit": "iter/sec",
            "range": "stddev: 0.000024393556568525173",
            "extra": "mean: 43.74559999860139 usec\nrounds: 5"
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
          "id": "454b3c8098cffa3ba2d317f34d81668ca24f1996",
          "message": "Integrate Brax and unify environment API (#39)\n\n## Summary\n- add a canonical Brax adapter in `rl-components`\n- prove Brax environments run through `GymnaxCompatibilityBridge`\n- let PPO and SAC accept injected canonical envs without\nbackend-specific branches\n\n## Testing\n- uv run pytest tests/small/test_rl_agents_sac.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/medium/test_rl_agents_ppo_gradient.py\ntests/medium/test_rl_agents_sac_gradient.py -q\n- uv run ruff check libs/rl-components/src/rl_components/brax.py\nlibs/rl-agents/src/rl_agents/ppo.py libs/rl-agents/src/rl_agents/sac.py\ntests/small/test_rl_agents_sac.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/medium/test_rl_agents_ppo_gradient.py\ntests/medium/test_rl_agents_sac_gradient.py\n- uv run pyright libs/rl-components/src/rl_components/brax.py\nlibs/rl-agents/src/rl_agents/ppo.py libs/rl-agents/src/rl_agents/sac.py\ntests/small/test_rl_agents_sac.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/medium/test_rl_agents_ppo_gradient.py\ntests/medium/test_rl_agents_sac_gradient.py\n- uv run ty check libs/rl-components/src/rl_components/brax.py\nlibs/rl-agents/src/rl_agents/ppo.py libs/rl-agents/src/rl_agents/sac.py\ntests/small/test_rl_agents_sac.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/medium/test_rl_agents_ppo_gradient.py\ntests/medium/test_rl_agents_sac_gradient.py\n\nCloses #16",
          "timestamp": "2026-03-28T22:31:58-06:00",
          "tree_id": "8b25698735475ccf034d9491b50852a5fff6e114",
          "url": "https://github.com/andnp/jax-research-template/commit/454b3c8098cffa3ba2d317f34d81668ca24f1996"
        },
        "date": 1774758998469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.7798588737639855,
            "unit": "iter/sec",
            "range": "stddev: 0.0005513078265074788",
            "extra": "mean: 173.0146049999964 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9831964741866531,
            "unit": "iter/sec",
            "range": "stddev: 0.007680881582496374",
            "extra": "mean: 1.0170907100000004 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04168629770043327,
            "unit": "iter/sec",
            "range": "stddev: 0.10886293350325371",
            "extra": "mean: 23.988697849499992 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.419651421736782,
            "unit": "iter/sec",
            "range": "stddev: 0.004640549156140057",
            "extra": "mean: 2.3829300896000065 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 30226.09115635794,
            "unit": "iter/sec",
            "range": "stddev: 0.000014847464128266829",
            "extra": "mean: 33.08400000605616 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 30011.6445218801,
            "unit": "iter/sec",
            "range": "stddev: 0.000021226064382738235",
            "extra": "mean: 33.32039999577319 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 23033.513766092772,
            "unit": "iter/sec",
            "range": "stddev: 0.000011246226593056629",
            "extra": "mean: 43.41499999327425 usec\nrounds: 5"
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
          "id": "3aabd5fc38b9d293f85f3d0816ac8af0b8ff0514",
          "message": "Implement continuous action space normalization (#40)\n\n## Summary\n- extend the canonical env spec to represent bounded continuous action\nspaces\n- add a thin canonical action-normalization wrapper and teach Brax to\npublish control bounds\n- prove normalized Brax envs work through the bridge under JIT and\nthrough the existing SAC injected-env seam\n\n## Testing\n- uv run pytest tests/small/test_rl_components_env_protocol.py\ntests/small/test_rl_components_action_normalization.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/medium/test_rl_agents_sac_gradient.py -q\n- uv run ruff check libs/rl-components/src/rl_components/env_protocol.py\nlibs/rl-components/src/rl_components/action_normalization.py\nlibs/rl-components/src/rl_components/brax.py\ntests/small/test_rl_components_env_protocol.py\ntests/small/test_rl_components_action_normalization.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/medium/test_rl_agents_sac_gradient.py\n- uv run pyright libs/rl-components/src/rl_components/env_protocol.py\nlibs/rl-components/src/rl_components/action_normalization.py\nlibs/rl-components/src/rl_components/brax.py\ntests/small/test_rl_components_env_protocol.py\ntests/small/test_rl_components_action_normalization.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/medium/test_rl_agents_sac_gradient.py\n- uv run ty check libs/rl-components/src/rl_components/env_protocol.py\nlibs/rl-components/src/rl_components/action_normalization.py\nlibs/rl-components/src/rl_components/brax.py\ntests/small/test_rl_components_env_protocol.py\ntests/small/test_rl_components_action_normalization.py\ntests/small/test_rl_components_gymnax_bridge.py\ntests/medium/test_rl_components_gymnax_bridge_jit.py\ntests/medium/test_rl_agents_sac_gradient.py\n\nCloses #17",
          "timestamp": "2026-03-29T11:54:28-06:00",
          "tree_id": "27b61eccd39ee34c4cc1198d892aef50795bf1d3",
          "url": "https://github.com/andnp/jax-research-template/commit/3aabd5fc38b9d293f85f3d0816ac8af0b8ff0514"
        },
        "date": 1774807159154,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.834394322802938,
            "unit": "iter/sec",
            "range": "stddev: 0.00029102144912437065",
            "extra": "mean: 171.39739699999978 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9993276250589016,
            "unit": "iter/sec",
            "range": "stddev: 0.0032442070406792996",
            "extra": "mean: 1.000672827333337 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04151170222824927,
            "unit": "iter/sec",
            "range": "stddev: 0.004227582141107373",
            "extra": "mean: 24.089592725000003 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.42382704007330513,
            "unit": "iter/sec",
            "range": "stddev: 0.005838570424431761",
            "extra": "mean: 2.3594530443999986 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 23590.135546844398,
            "unit": "iter/sec",
            "range": "stddev: 0.000023900929554057247",
            "extra": "mean: 42.39060000372774 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 35811.74482472008,
            "unit": "iter/sec",
            "range": "stddev: 0.000011133154914682646",
            "extra": "mean: 27.92379999618788 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 22388.995362043937,
            "unit": "iter/sec",
            "range": "stddev: 0.000014179931351600308",
            "extra": "mean: 44.664799997917726 usec\nrounds: 5"
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
          "id": "c16f3128943661a625a60b8a75f86309ff4bc184",
          "message": "feat(rl-agents): scaffold atari dqn baseline (#41)\n\n## Summary\n- add an Atari-specific `rl_agents.dqn_atari` scaffold with DQN\nZoo-style scheduler and optimizer helpers plus a runnable Pong probe\npath\n- add focused smoke/small coverage for the Atari-specific trainer and\nreal JAXAtari Pong path\n- add performance benchmarks that split env-loop, train-update, and\nlearner subphase costs\n- reduce Atari replay storage pressure by storing replay observations as\n`uint8`\n\n## Verification\n- `uv run pytest tests/small/test_rl_agents_dqn_atari.py -q`\n- `JAXATARI_CONFIRM_OWNERSHIP=1 JAXATARI_RUN_SMOKE=1 uv run pytest\ntests/regression/test_rl_agents_dqn_atari_real_smoke.py -q`\n- `JAXATARI_CONFIRM_OWNERSHIP=1 uv run python\nexamples/train_dqn_atari.py`\n- `uv run pytest\ntests/performance/test_rl_agents_dqn_atari_env_loop_bench.py -k fake\n--benchmark-only -q`\n- `JAXATARI_BENCHMARKS=1 uv run pytest\ntests/performance/test_rl_agents_dqn_atari_env_loop_bench.py -k\nreal_pong_env_only_rollout_speed --benchmark-only`\n- `JAXATARI_BENCHMARKS=1 uv run pytest\ntests/performance/test_rl_agents_dqn_atari_env_loop_bench.py -k\nreal_pong_policy_and_env_rollout_speed --benchmark-only`\n- `uv run pytest\ntests/performance/test_rl_agents_dqn_atari_env_loop_bench.py -k\n'fake_replay_sampling_only_speed or fake_loss_and_grad_fixed_batch_speed\nor fake_optimizer_apply_fixed_grads_speed or fake_full_learn_step_speed'\n--benchmark-only -q`\n\n## Notes\n- Short Pong probe showed weak but real signs of life (`max completed\nreturn = -19.00` in a quick run).\n- Benchmark investigation indicates the remaining dominant cost is\nlearner forward/backward compute rather than env stepping or replay\nsampling.\n- Closes #12.",
          "timestamp": "2026-03-29T13:04:39-06:00",
          "tree_id": "0b312c88d7347e7c7228b059322752dd8fb36522",
          "url": "https://github.com/andnp/jax-research-template/commit/c16f3128943661a625a60b8a75f86309ff4bc184"
        },
        "date": 1774811374763,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.817766763892073,
            "unit": "iter/sec",
            "range": "stddev: 0.0010792329323046764",
            "extra": "mean: 171.8872619999985 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9965051385592949,
            "unit": "iter/sec",
            "range": "stddev: 0.004650522420254829",
            "extra": "mean: 1.0035071183333362 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.041400999188204404,
            "unit": "iter/sec",
            "range": "stddev: 0.0036103493389258497",
            "extra": "mean: 24.154006415500014 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.42374179093895314,
            "unit": "iter/sec",
            "range": "stddev: 0.00816475768060281",
            "extra": "mean: 2.359927723399994 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 16930.156332031976,
            "unit": "iter/sec",
            "range": "stddev: 0.00001666829842054833",
            "extra": "mean: 59.06620000359908 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 45.43201200263979,
            "unit": "iter/sec",
            "range": "stddev: 0.0001161287369935652",
            "extra": "mean: 22.010911599994643 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.8044576926069604,
            "unit": "iter/sec",
            "range": "stddev: 0.004026096820536142",
            "extra": "mean: 356.57517766667485 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 3758.390607099107,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072879631050610675",
            "extra": "mean: 266.0713333284548 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 48.801854288253665,
            "unit": "iter/sec",
            "range": "stddev: 0.0006176646549449871",
            "extra": "mean: 20.49102466667326 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 338.8427661130073,
            "unit": "iter/sec",
            "range": "stddev: 0.0003004401905490156",
            "extra": "mean: 2.951221333338102 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 46.617712474447316,
            "unit": "iter/sec",
            "range": "stddev: 0.000649241839401245",
            "extra": "mean: 21.451073999998016 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 30397.539014063263,
            "unit": "iter/sec",
            "range": "stddev: 0.000013754062414892715",
            "extra": "mean: 32.897400001274946 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 26258.573427840078,
            "unit": "iter/sec",
            "range": "stddev: 0.00002924685788369978",
            "extra": "mean: 38.08279999475417 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 21970.00655005989,
            "unit": "iter/sec",
            "range": "stddev: 0.000010989771458241274",
            "extra": "mean: 45.51659999378899 usec\nrounds: 5"
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
          "id": "2a50d28d06530bd7a531111308e25767bd934634",
          "message": "Add repo-shared workflow skills (#44)\n\n## Summary\n- add a repo-shared workflow skill architecture/spec and shared skill\nindex\n- add the initial shell/project lifecycle skills for bootstrap, project\ncreation, diagnosis, repair, change-location decisions, and upstream\ncontribution\n- ground skill content in the current CLI/docs/contracts without\ninventing unsupported automation\n\n## Testing\n- read back all new skill/reference markdown and frontmatter\n- git diff --check\n\nCloses #42",
          "timestamp": "2026-03-29T13:04:55-06:00",
          "tree_id": "e5475fa87d8255a0192043ef58eebdf5725a6a7c",
          "url": "https://github.com/andnp/jax-research-template/commit/2a50d28d06530bd7a531111308e25767bd934634"
        },
        "date": 1774811394778,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.1268706170779215,
            "unit": "iter/sec",
            "range": "stddev: 0.0015082716226402161",
            "extra": "mean: 163.21545900000226 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.0661962460982974,
            "unit": "iter/sec",
            "range": "stddev: 0.0009086701426043399",
            "extra": "mean: 937.9136380000025 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04259484637260257,
            "unit": "iter/sec",
            "range": "stddev: 0.08689060921871876",
            "extra": "mean: 23.477018586999996 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.43165555375573134,
            "unit": "iter/sec",
            "range": "stddev: 0.011778316136407861",
            "extra": "mean: 2.3166619571999947 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 14722.25003159842,
            "unit": "iter/sec",
            "range": "stddev: 0.000024102606289929327",
            "extra": "mean: 67.92439999685485 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 51.5237122892227,
            "unit": "iter/sec",
            "range": "stddev: 0.00010268276417501513",
            "extra": "mean: 19.40853940000693 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.8431889110917155,
            "unit": "iter/sec",
            "range": "stddev: 0.002550624169840667",
            "extra": "mean: 351.71774766665936 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 3930.544655414453,
            "unit": "iter/sec",
            "range": "stddev: 0.000050987708570081886",
            "extra": "mean: 254.41766667692417 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 51.95178078151397,
            "unit": "iter/sec",
            "range": "stddev: 0.00047323728431712843",
            "extra": "mean: 19.248618333326323 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 350.16024500163115,
            "unit": "iter/sec",
            "range": "stddev: 0.00014019716625270924",
            "extra": "mean: 2.8558353333210107 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 50.219695263868104,
            "unit": "iter/sec",
            "range": "stddev: 0.0001449722815715144",
            "extra": "mean: 19.91250633333645 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 33020.518957064436,
            "unit": "iter/sec",
            "range": "stddev: 0.00001145859283650229",
            "extra": "mean: 30.284199993957372 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 35584.402648763426,
            "unit": "iter/sec",
            "range": "stddev: 0.000010682612078370588",
            "extra": "mean: 28.10219999673791 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 24402.147388975904,
            "unit": "iter/sec",
            "range": "stddev: 0.000011485694183339566",
            "extra": "mean: 40.97999999999047 usec\nrounds: 5"
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
          "id": "891f7099636b0afba0d4f6f6f9ec0cb7ff030915",
          "message": "feat(rainbow): add missing baseline components (#49)\n\nCloses #14\n\n## Summary\n- normalize PER importance-sampling weights against the global minimum\npopulated sampling probability\n- harden `NoisyLinear` input dtype and shape handling while keeping\nfactorized-noise behavior unchanged\n- add reusable C51 distributional helpers and a categorical value head\nin `jax-nn`\n\n## Verification\n- `uv run ruff check libs/jax-replay/src/jax_replay/per.py\nlibs/jax-nn/src/jax_nn/layers.py\nlibs/jax-nn/src/jax_nn/distributional.py tests/small/test_sum_tree.py\ntests/small/test_per_buffer.py tests/medium/test_per_jit.py\ntests/small/test_jax_nn_noisy_linear.py\ntests/medium/test_jax_nn_noisy_linear_jit.py\ntests/small/test_jax_nn_c51.py tests/medium/test_jax_nn_c51_jit.py\ntests/small/test_jax_nn_public_api.py`\n- `uv run pyright libs/jax-replay/src/jax_replay/per.py\nlibs/jax-nn/src/jax_nn/layers.py\nlibs/jax-nn/src/jax_nn/distributional.py tests/small/test_sum_tree.py\ntests/small/test_per_buffer.py tests/medium/test_per_jit.py\ntests/small/test_jax_nn_noisy_linear.py\ntests/medium/test_jax_nn_noisy_linear_jit.py\ntests/small/test_jax_nn_c51.py tests/medium/test_jax_nn_c51_jit.py\ntests/small/test_jax_nn_public_api.py`\n- `uv run pytest tests/small/test_sum_tree.py\ntests/small/test_per_buffer.py tests/medium/test_per_jit.py\ntests/small/test_jax_nn_noisy_linear.py\ntests/medium/test_jax_nn_noisy_linear_jit.py\ntests/small/test_jax_nn_c51.py tests/medium/test_jax_nn_c51_jit.py\ntests/small/test_jax_nn_public_api.py -q`",
          "timestamp": "2026-03-29T17:07:07-06:00",
          "tree_id": "7ec5240f2d05ad32dad822b5b4959de244a48d59",
          "url": "https://github.com/andnp/jax-research-template/commit/891f7099636b0afba0d4f6f6f9ec0cb7ff030915"
        },
        "date": 1774825921423,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 4.480936393824749,
            "unit": "iter/sec",
            "range": "stddev: 0.0018919617465291121",
            "extra": "mean: 223.16764000000452 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9083936786798819,
            "unit": "iter/sec",
            "range": "stddev: 0.017756764557268285",
            "extra": "mean: 1.1008442963333305 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.048342499989113615,
            "unit": "iter/sec",
            "range": "stddev: 0.15621417434950194",
            "extra": "mean: 20.685732021000007 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.3143402896857668,
            "unit": "iter/sec",
            "range": "stddev: 0.01787297910137192",
            "extra": "mean: 3.1812657582000043 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 23029.482341253653,
            "unit": "iter/sec",
            "range": "stddev: 0.000019291521784441618",
            "extra": "mean: 43.422600003850675 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 35.09193230019408,
            "unit": "iter/sec",
            "range": "stddev: 0.0003096692840809594",
            "extra": "mean: 28.49657840000077 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 3.3845592849377275,
            "unit": "iter/sec",
            "range": "stddev: 0.0025593809387366136",
            "extra": "mean: 295.4594426666688 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 3685.191304549121,
            "unit": "iter/sec",
            "range": "stddev: 0.000027759719855941586",
            "extra": "mean: 271.35633332401693 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 51.27990453872003,
            "unit": "iter/sec",
            "range": "stddev: 0.0005306817537432277",
            "extra": "mean: 19.50081633332464 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 223.07721936904673,
            "unit": "iter/sec",
            "range": "stddev: 0.0007987254841055274",
            "extra": "mean: 4.482752666670346 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 43.59528578017894,
            "unit": "iter/sec",
            "range": "stddev: 0.00047794502393070786",
            "extra": "mean: 22.938259999998916 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 38368.56847503797,
            "unit": "iter/sec",
            "range": "stddev: 0.000012839632574500151",
            "extra": "mean: 26.06299999570183 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 42980.065837166374,
            "unit": "iter/sec",
            "range": "stddev: 0.000012028082494733145",
            "extra": "mean: 23.2666000044901 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 27031.410504280037,
            "unit": "iter/sec",
            "range": "stddev: 0.00001630657779391707",
            "extra": "mean: 36.99399999277375 usec\nrounds: 5"
          }
        ]
      }
    ]
  }
}