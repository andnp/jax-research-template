window.BENCHMARK_DATA = {
  "lastUpdate": 1780868851422,
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
          "id": "0dd8a569b5f41cbced305bc6b12b3c0f684dd0ab",
          "message": "feat(rl-agents): assemble Rainbow Atari baseline (#51)\n\n## Summary\n- add a Rainbow Atari train path with PER-backed replay, streaming\nN-step accumulation, and focused fake-env integration coverage\n- teach PER to support logical non-power-of-two capacities via padded\ninternal storage so Rainbow can keep a 1,000,000 transition replay size\n- add semantic parity contracts for DQN-Zoo-style schedule/cadence,\nRMSProp wiring, Double-Q distributional targets, and golden C51 target\nmath cases\n\n## Testing\n- uv run pytest tests/small/test_per_buffer.py\ntests/small/test_rl_agents_rainbow.py\ntests/medium/test_rl_agents_rainbow_nature_env_integration.py -q\n- uv run ruff check libs/jax-replay/src/jax_replay/per.py\nlibs/jax-replay/src/jax_replay/types.py\nlibs/rl-agents/src/rl_agents/rainbow.py tests/small/test_per_buffer.py\ntests/small/test_rl_agents_rainbow.py\ntests/medium/test_rl_agents_rainbow_nature_env_integration.py\n- cd libs/jax-replay && uv run pyright src\n- cd libs/rl-agents && uv run pyright src\n- uv run pyright tests/small/test_per_buffer.py\ntests/small/test_rl_agents_rainbow.py\ntests/medium/test_rl_agents_rainbow_nature_env_integration.py\n\nCloses #15",
          "timestamp": "2026-03-29T19:48:39-06:00",
          "tree_id": "e9e0036f23bdf9e36a31dacade993b893c293d79",
          "url": "https://github.com/andnp/jax-research-template/commit/0dd8a569b5f41cbced305bc6b12b3c0f684dd0ab"
        },
        "date": 1774835653976,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.837001658638678,
            "unit": "iter/sec",
            "range": "stddev: 0.0019782352341809984",
            "extra": "mean: 171.32083533332812 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.0007628862183895,
            "unit": "iter/sec",
            "range": "stddev: 0.002301599282183209",
            "extra": "mean: 999.2376953333348 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04121463518986646,
            "unit": "iter/sec",
            "range": "stddev: 0.04969087828722296",
            "extra": "mean: 24.263225803000005 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.4217121067826755,
            "unit": "iter/sec",
            "range": "stddev: 0.005950725947636609",
            "extra": "mean: 2.3712859648000064 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 15916.1157017648,
            "unit": "iter/sec",
            "range": "stddev: 0.00002280695388105927",
            "extra": "mean: 62.82940000801318 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 49.301374381316535,
            "unit": "iter/sec",
            "range": "stddev: 0.00026371563320132946",
            "extra": "mean: 20.283410199999707 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.7647604561222083,
            "unit": "iter/sec",
            "range": "stddev: 0.0007975048993639896",
            "extra": "mean: 361.69498800000116 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 3885.8845243047326,
            "unit": "iter/sec",
            "range": "stddev: 0.00001840761683141278",
            "extra": "mean: 257.3416666772725 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 49.25077585569231,
            "unit": "iter/sec",
            "range": "stddev: 0.0005466523397348774",
            "extra": "mean: 20.304248666661806 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 334.49939823618996,
            "unit": "iter/sec",
            "range": "stddev: 0.00047633585754638106",
            "extra": "mean: 2.9895419999945716 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 46.106145445080074,
            "unit": "iter/sec",
            "range": "stddev: 0.0010748035105304277",
            "extra": "mean: 21.689082666673205 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 34539.68955694075,
            "unit": "iter/sec",
            "range": "stddev: 0.00001207937773526289",
            "extra": "mean: 28.952200000276207 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 35594.78891799159,
            "unit": "iter/sec",
            "range": "stddev: 0.000011152449263679358",
            "extra": "mean: 28.094000003875408 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 24192.455827341793,
            "unit": "iter/sec",
            "range": "stddev: 0.00001207120307161553",
            "extra": "mean: 41.3351999952738 usec\nrounds: 5"
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
          "id": "ec0ad07d5add3c86b2bbc5dc94ac461084e5fa73",
          "message": "feat(cli): improve fresh-shell bootstrap operability (#52)\n\n## Summary\n- make freshly initialized shells include `core/cli` in uv workspace\nmembership so the CLI is available after workspace sync\n- align `workspace init` output and bootstrap docs with the truthful\npost-init flow: `uv sync --all-packages` then `uv run research doctor`\n- add a medium integration test that proves the end-to-end bootstrap\npath using a tiny local git-backed `core` fixture\n\n## Testing\n- uv run pytest tests/small/test_workspace_init.py -q\n- uv run ruff check cli/src/research_cli/workspace.py\ntests/small/test_workspace_init.py\n- cd cli && uv run pyright src\n- uv run pytest tests/medium/test_bootstrap_truthful_path.py -q\n- uv run ruff check tests/medium/test_bootstrap_truthful_path.py\n- uv run ty check tests/medium/test_bootstrap_truthful_path.py\n\nCloses #45",
          "timestamp": "2026-03-29T20:33:19-06:00",
          "tree_id": "35f6a25de741770d4ee4525d366331040260c396",
          "url": "https://github.com/andnp/jax-research-template/commit/ec0ad07d5add3c86b2bbc5dc94ac461084e5fa73"
        },
        "date": 1774838319205,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.226113231225001,
            "unit": "iter/sec",
            "range": "stddev: 0.000983818484183224",
            "extra": "mean: 160.61384733333028 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.0580828182858584,
            "unit": "iter/sec",
            "range": "stddev: 0.005712600924651751",
            "extra": "mean: 945.1056030000042 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04204680618256391,
            "unit": "iter/sec",
            "range": "stddev: 0.06054788772942673",
            "extra": "mean: 23.783019229999994 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.43540259863581454,
            "unit": "iter/sec",
            "range": "stddev: 0.00874606216535581",
            "extra": "mean: 2.296724923399995 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 15006.272621656817,
            "unit": "iter/sec",
            "range": "stddev: 0.00002601796309444751",
            "extra": "mean: 66.63880000132849 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 50.53357185719047,
            "unit": "iter/sec",
            "range": "stddev: 0.0004517894117911373",
            "extra": "mean: 19.788824799996974 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.823046433024451,
            "unit": "iter/sec",
            "range": "stddev: 0.0012586741313809356",
            "extra": "mean: 354.2272590000077 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 3481.643614410142,
            "unit": "iter/sec",
            "range": "stddev: 0.00004022003077708232",
            "extra": "mean: 287.22066665901974 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 50.572662067553374,
            "unit": "iter/sec",
            "range": "stddev: 0.0004580083104644965",
            "extra": "mean: 19.77352900000066 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 327.20657205296504,
            "unit": "iter/sec",
            "range": "stddev: 0.0001955751707224326",
            "extra": "mean: 3.0561733333343 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 48.57187279587902,
            "unit": "iter/sec",
            "range": "stddev: 0.00012253549470436718",
            "extra": "mean: 20.588047000008675 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 34693.31112579968,
            "unit": "iter/sec",
            "range": "stddev: 0.000011483748146599563",
            "extra": "mean: 28.824000003169203 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 35609.23845440529,
            "unit": "iter/sec",
            "range": "stddev: 0.000011427217656774878",
            "extra": "mean: 28.082600005063796 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 24189.997918480392,
            "unit": "iter/sec",
            "range": "stddev: 0.00001232841038782625",
            "extra": "mean: 41.33940000201619 usec\nrounds: 5"
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
          "id": "0381a08ebaddfeffce1abc29c332cbc367a49888",
          "message": "fix(template): make generated projects truthfully smoke-runnable (#57)\n\n## Summary\n- make generated project scaffolds truthful about setup by declaring\nreal starter dependencies and workspace-local package sources\n- align the generated smoke starter with the current explicit-env\ntrainer API and correct the documented workspace-root run command\n- add a medium project-create smoke path that proves the documented\nsetup/run flow works in a hermetic temp workspace\n\n## Testing\n- uv run pytest tests/small/test_project_create.py -q\n- uv run ruff check tests/small/test_project_create.py\ntests/medium/test_project_create_truthful_path.py\n- uv run pyright tests/small/test_project_create.py\n- uv run pytest tests/medium/test_project_create_truthful_path.py -q\n- uv run ty check tests/small/test_project_create.py\ntests/medium/test_project_create_truthful_path.py\ncli/src/research_cli/project.py\n\nCloses #46",
          "timestamp": "2026-03-29T21:25:14-06:00",
          "tree_id": "655ba2a0e383026b571520c948aee8542aecef29",
          "url": "https://github.com/andnp/jax-research-template/commit/0381a08ebaddfeffce1abc29c332cbc367a49888"
        },
        "date": 1774841442820,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.089388929029372,
            "unit": "iter/sec",
            "range": "stddev: 0.0011044572679795913",
            "extra": "mean: 164.22009033333276 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.071424692105881,
            "unit": "iter/sec",
            "range": "stddev: 0.0021425349177557904",
            "extra": "mean: 933.3367126666682 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.03855767553519056,
            "unit": "iter/sec",
            "range": "stddev: 0.0064259240908123415",
            "extra": "mean: 25.93517337650001 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.44471710006366,
            "unit": "iter/sec",
            "range": "stddev: 0.003208361112124878",
            "extra": "mean: 2.248620527199995 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 17900.55169359098,
            "unit": "iter/sec",
            "range": "stddev: 0.000017927909394518077",
            "extra": "mean: 55.864200004407394 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 48.59717782777012,
            "unit": "iter/sec",
            "range": "stddev: 0.00030735481387065677",
            "extra": "mean: 20.57732659999374 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.6730136580011106,
            "unit": "iter/sec",
            "range": "stddev: 0.004409709914947755",
            "extra": "mean: 374.1095736666769 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 5009.375882202038,
            "unit": "iter/sec",
            "range": "stddev: 0.000016550303234657164",
            "extra": "mean: 199.62566665299164 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 49.55422260202796,
            "unit": "iter/sec",
            "range": "stddev: 0.0007406267291362006",
            "extra": "mean: 20.179914999999937 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 338.20925863652224,
            "unit": "iter/sec",
            "range": "stddev: 0.00019187416992522545",
            "extra": "mean: 2.956749333331269 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 46.13488116885466,
            "unit": "iter/sec",
            "range": "stddev: 0.0003233477859977742",
            "extra": "mean: 21.67557333333055 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 40926.577720452806,
            "unit": "iter/sec",
            "range": "stddev: 0.000015036512722098881",
            "extra": "mean: 24.4339999994736 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 40246.63135772843,
            "unit": "iter/sec",
            "range": "stddev: 0.000011435715423655633",
            "extra": "mean: 24.846799999522773 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 30470.897253579773,
            "unit": "iter/sec",
            "range": "stddev: 0.000013522859197547906",
            "extra": "mean: 32.81819999187974 usec\nrounds: 5"
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
          "id": "a55ec555e98b480f6acb59d241d4e941b9c8e3f0",
          "message": "Add execution orchestration facade",
          "timestamp": "2026-03-29T21:37:38-06:00",
          "tree_id": "3b5473a001ceda58e29daeb348dba2935a5c5811",
          "url": "https://github.com/andnp/jax-research-template/commit/a55ec555e98b480f6acb59d241d4e941b9c8e3f0"
        },
        "date": 1774842178399,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.047604566006157,
            "unit": "iter/sec",
            "range": "stddev: 0.002244605956839893",
            "extra": "mean: 165.35472666666115 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.0578656420292913,
            "unit": "iter/sec",
            "range": "stddev: 0.0001603632976386886",
            "extra": "mean: 945.2996299999986 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04208605051345229,
            "unit": "iter/sec",
            "range": "stddev: 0.04302181694389311",
            "extra": "mean: 23.76084207950001 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.43698480654779986,
            "unit": "iter/sec",
            "range": "stddev: 0.012400376251470564",
            "extra": "mean: 2.288409082000004 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 15003.705914049946,
            "unit": "iter/sec",
            "range": "stddev: 0.00001384892937291528",
            "extra": "mean: 66.65020000582444 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 47.156067106678435,
            "unit": "iter/sec",
            "range": "stddev: 0.0003637610373935319",
            "extra": "mean: 21.206178999995018 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.8507549560275454,
            "unit": "iter/sec",
            "range": "stddev: 0.0024632723962739246",
            "extra": "mean: 350.7842713333294 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 4211.96281677128,
            "unit": "iter/sec",
            "range": "stddev: 0.000025133025844718775",
            "extra": "mean: 237.4190000011822 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 49.4827110376465,
            "unit": "iter/sec",
            "range": "stddev: 0.00013185216240448608",
            "extra": "mean: 20.20907866667206 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 332.35234000976703,
            "unit": "iter/sec",
            "range": "stddev: 0.000311201344456255",
            "extra": "mean: 3.0088549999997363 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 45.913588819921834,
            "unit": "iter/sec",
            "range": "stddev: 0.00047905946520724864",
            "extra": "mean: 21.780044333326032 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 34489.894459015944,
            "unit": "iter/sec",
            "range": "stddev: 0.000010664374117849821",
            "extra": "mean: 28.994000001603126 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 36269.204554440454,
            "unit": "iter/sec",
            "range": "stddev: 0.00001129986746230568",
            "extra": "mean: 27.57159999191572 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 24348.317286302463,
            "unit": "iter/sec",
            "range": "stddev: 0.000012051036579743358",
            "extra": "mean: 41.07060000251295 usec\nrounds: 5"
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
          "id": "d326ac5204deb23afdf6f7e5932d1912a094027a",
          "message": "feat(cli): align shell workspace discovery across child project repos (#58)\n\n## Summary\n- resolve the enclosing shell workspace upward for shell-scoped CLI\ncommands when invoked from inside child project repos\n- make shell vs child-project Git ownership explicit for `research\nproject create`\n- align lifecycle commands with the same workspace discovery contract\nalready used by `workspace repair`\n\n## What changed\n- `research project create` now resolves the enclosing shell workspace\nroot upward from `cwd`\n- `research doctor` now resolves the enclosing shell workspace root\nupward from `cwd`\n- `research eject` and `research harvest` now do the same while still\nrequiring explicit project arguments\n- `project create` still initializes each child project as its own Git\nrepo\n- docs/specs now explicitly describe shell-owned files like\n`research.yaml` and `uv.lock` vs child-project Git ownership\n- added focused small/medium coverage for running shell-scoped commands\nfrom inside child project repos\n\n## Verification\n- `uv run --with pytest pytest -q tests/small/test_project_create.py\ntests/medium/test_project_create_render.py\ntests/medium/test_cli_doctor.py`\n- `uv run --with pytest pytest -q tests/small/test_lifecycle.py\ntests/medium/test_cli_eject.py tests/medium/test_cli_harvest.py`\n- `uv run --with pytest pytest -q tests/small/test_project_create.py\ntests/medium/test_project_create_render.py\ntests/medium/test_cli_doctor.py tests/small/test_lifecycle.py\ntests/medium/test_cli_eject.py tests/medium/test_cli_harvest.py`\n- `uv run --with ruff ruff check cli/src/research_cli/project.py\ncli/src/research_cli/doctor.py cli/src/research_cli/workspace.py\ncli/src/research_cli/lifecycle.py tests/small/test_project_create.py\ntests/medium/test_project_create_render.py\ntests/medium/test_cli_doctor.py tests/small/test_lifecycle.py\ntests/medium/test_cli_eject.py tests/medium/test_cli_harvest.py`\n- `uv run --with pyright pyright cli/src/research_cli`\n\nCloses #47\nCloses #48",
          "timestamp": "2026-03-30T20:34:25-06:00",
          "tree_id": "5d74fdd095b48be74663044d62fc2da5f31e29fa",
          "url": "https://github.com/andnp/jax-research-template/commit/d326ac5204deb23afdf6f7e5932d1912a094027a"
        },
        "date": 1774924775695,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.181728437358308,
            "unit": "iter/sec",
            "range": "stddev: 0.00021423574676694964",
            "extra": "mean: 161.76705433332472 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.0558851148001989,
            "unit": "iter/sec",
            "range": "stddev: 0.005798465454027927",
            "extra": "mean: 947.0727316666702 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04247131308078536,
            "unit": "iter/sec",
            "range": "stddev: 0.03402647116412433",
            "extra": "mean: 23.545304523499993 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.43500420566329195,
            "unit": "iter/sec",
            "range": "stddev: 0.014162768972811197",
            "extra": "mean: 2.2988283491999937 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 16567.42777319589,
            "unit": "iter/sec",
            "range": "stddev: 0.000017775523918112367",
            "extra": "mean: 60.359400004017516 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 48.71119339966488,
            "unit": "iter/sec",
            "range": "stddev: 0.00031800546183144093",
            "extra": "mean: 20.52916240000968 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.863136659223717,
            "unit": "iter/sec",
            "range": "stddev: 0.004228309181356185",
            "extra": "mean: 349.26729633336134 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 4854.934555059653,
            "unit": "iter/sec",
            "range": "stddev: 0.000017675996509826367",
            "extra": "mean: 205.9760000179267 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 48.868972429655535,
            "unit": "iter/sec",
            "range": "stddev: 0.000464917913121756",
            "extra": "mean: 20.462881666674093 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 340.9633350747101,
            "unit": "iter/sec",
            "range": "stddev: 0.0005359191645350465",
            "extra": "mean: 2.9328666666780614 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 45.56629264488614,
            "unit": "iter/sec",
            "range": "stddev: 0.0005658160496701358",
            "extra": "mean: 21.94604699999066 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 35833.817094761675,
            "unit": "iter/sec",
            "range": "stddev: 0.000010625940562410362",
            "extra": "mean: 27.906599996185832 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 36059.42591795515,
            "unit": "iter/sec",
            "range": "stddev: 0.000010933220426435214",
            "extra": "mean: 27.7320000122927 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 24260.303346698278,
            "unit": "iter/sec",
            "range": "stddev: 0.000012988492462799643",
            "extra": "mean: 41.21960000702529 usec\nrounds: 5"
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
          "id": "a3028b382581b2fcb2b83419510f629f022c7565",
          "message": "feat(research-analysis): add step-weighted learning curves (#59)\n\n## Summary\n- add step-weighted copy-forward learning curve interpolation in \n- add a Polars DataFrame convenience adapter with explicit column\nvalidation\n- add a medium SQLite-to-Polars-to-learning-curve composition test\n\n## Testing\n- uv run --with pytest pytest -q\ntests/small/test_research_analysis_learning_curve.py\ntests/small/test_research_analysis_bootstrap.py\ntests/small/test_research_analysis_hypothesis.py\ntests/medium/test_research_analysis_loader.py",
          "timestamp": "2026-03-30T21:11:10-06:00",
          "tree_id": "d93b72256709d5b6e607367f10b6796a54555f6a",
          "url": "https://github.com/andnp/jax-research-template/commit/a3028b382581b2fcb2b83419510f629f022c7565"
        },
        "date": 1774926980720,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.2294669113549075,
            "unit": "iter/sec",
            "range": "stddev: 0.0003679979452018074",
            "extra": "mean: 160.5273796666656 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.0627434305489276,
            "unit": "iter/sec",
            "range": "stddev: 0.00068762553294068",
            "extra": "mean: 940.9608860000016 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04236135052631499,
            "unit": "iter/sec",
            "range": "stddev: 0.0075271502715197995",
            "extra": "mean: 23.60642395900001 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.4344794906409506,
            "unit": "iter/sec",
            "range": "stddev: 0.012852897766501935",
            "extra": "mean: 2.3016046132000043 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 16394.625187465994,
            "unit": "iter/sec",
            "range": "stddev: 0.000020037787064063495",
            "extra": "mean: 60.99559999483972 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 48.03396547024775,
            "unit": "iter/sec",
            "range": "stddev: 0.0006088193575988157",
            "extra": "mean: 20.818601799999215 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.853928572522609,
            "unit": "iter/sec",
            "range": "stddev: 0.0007585755329918639",
            "extra": "mean: 350.39419333332944 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 4140.101018466384,
            "unit": "iter/sec",
            "range": "stddev: 0.00001590676923444823",
            "extra": "mean: 241.53999999991052 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 48.37856732065262,
            "unit": "iter/sec",
            "range": "stddev: 0.0007055613048042113",
            "extra": "mean: 20.670310333334402 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 380.44179183403315,
            "unit": "iter/sec",
            "range": "stddev: 0.0005266804292761456",
            "extra": "mean: 2.6285230000079687 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 46.55757192951304,
            "unit": "iter/sec",
            "range": "stddev: 0.0010483027276412028",
            "extra": "mean: 21.478783333331346 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 35927.54133632873,
            "unit": "iter/sec",
            "range": "stddev: 0.000011218010366250378",
            "extra": "mean: 27.833799998688846 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 36916.17076296871,
            "unit": "iter/sec",
            "range": "stddev: 0.000011070305679650914",
            "extra": "mean: 27.088399997410306 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 25316.712074528765,
            "unit": "iter/sec",
            "range": "stddev: 0.000011101601834569176",
            "extra": "mean: 39.499599989767376 usec\nrounds: 5"
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
          "id": "b1f9261d7da119433e462eb189af6bef4f547c43",
          "message": "feat(rl-agents): harden continuous PPO for robotics (#60)\n\n## Summary\n- add a continuous PPO policy path with tanh-squashed Gaussian actions\nfor normalized continuous action spaces\n- add PPO-local observation normalization and reward scaling without\nchanging raw logged returns\n- extend PPO network, gradient, regression, and config coverage for the\nnew continuous and stabilization paths\n\n## Testing\n- PYTHONPATH=libs/rl-components/src:libs/rl-agents/src uv run --with\npytest pytest -q tests/small/test_rl_components_types.py\ntests/small/test_rl_agents_ppo.py\ntests/medium/test_rl_components_networks.py\ntests/medium/test_rl_agents_ppo_gradient.py\ntests/regression/test_ppo_learning.py",
          "timestamp": "2026-03-31T20:02:06-06:00",
          "tree_id": "edd8fa5151b79384f705fbf93f2909f8f8143869",
          "url": "https://github.com/andnp/jax-research-template/commit/b1f9261d7da119433e462eb189af6bef4f547c43"
        },
        "date": 1775009266780,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.868396019855651,
            "unit": "iter/sec",
            "range": "stddev: 0.0009412155862335537",
            "extra": "mean: 170.40431433333936 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.006791609131736,
            "unit": "iter/sec",
            "range": "stddev: 0.003425396610627162",
            "extra": "mean: 993.2542056666591 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.041202647420626515,
            "unit": "iter/sec",
            "range": "stddev: 0.1520098001600734",
            "extra": "mean: 24.270285105500008 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.4196989686668145,
            "unit": "iter/sec",
            "range": "stddev: 0.010423704230938463",
            "extra": "mean: 2.3826601318000087 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 14222.201994706522,
            "unit": "iter/sec",
            "range": "stddev: 0.00003715580459373749",
            "extra": "mean: 70.31260000189832 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 47.157903505240355,
            "unit": "iter/sec",
            "range": "stddev: 0.0002000171031130182",
            "extra": "mean: 21.205353199997035 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.6797515504794656,
            "unit": "iter/sec",
            "range": "stddev: 0.0023174647661411353",
            "extra": "mean: 373.1689229999991 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 4542.749544561993,
            "unit": "iter/sec",
            "range": "stddev: 0.0000381145168000245",
            "extra": "mean: 220.13100000132604 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 49.28117249765062,
            "unit": "iter/sec",
            "range": "stddev: 0.00007569693222647325",
            "extra": "mean: 20.29172500000224 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 270.8624341164848,
            "unit": "iter/sec",
            "range": "stddev: 0.0007322365756001351",
            "extra": "mean: 3.6919109999947373 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 43.204463539531744,
            "unit": "iter/sec",
            "range": "stddev: 0.0008520678975776951",
            "extra": "mean: 23.145756666669588 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 34381.74742055461,
            "unit": "iter/sec",
            "range": "stddev: 0.000013648254244243233",
            "extra": "mean: 29.085199997780364 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 36670.87160526238,
            "unit": "iter/sec",
            "range": "stddev: 0.000011576669703923679",
            "extra": "mean: 27.26959999108658 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 19892.263500915855,
            "unit": "iter/sec",
            "range": "stddev: 0.000023090352667969342",
            "extra": "mean: 50.27079999990747 usec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "test@example.com",
            "name": "Andy Patterson"
          },
          "committer": {
            "email": "test@example.com",
            "name": "Andy Patterson"
          },
          "distinct": true,
          "id": "877688d2e999afc18448ef348b91e66aaea17b40",
          "message": "fix: make the cli installable from the global pyproject.toml",
          "timestamp": "2026-05-27T15:41:53-06:00",
          "tree_id": "b1777f3b3b76e9f742a5109cfece5aa94588ae0d",
          "url": "https://github.com/andnp/jax-research-template/commit/877688d2e999afc18448ef348b91e66aaea17b40"
        },
        "date": 1779918452066,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.839201278079964,
            "unit": "iter/sec",
            "range": "stddev: 0.0016571476958235389",
            "extra": "mean: 171.2562989999924 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 1.024339397393673,
            "unit": "iter/sec",
            "range": "stddev: 0.006494771648148972",
            "extra": "mean: 976.2389326666513 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04201697049177597,
            "unit": "iter/sec",
            "range": "stddev: 0.04233984710167992",
            "extra": "mean: 23.799907234999992 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.42337192745704677,
            "unit": "iter/sec",
            "range": "stddev: 0.0039047536108950823",
            "extra": "mean: 2.3619893884000023 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 16540.189352431975,
            "unit": "iter/sec",
            "range": "stddev: 0.000013716484654318531",
            "extra": "mean: 60.4587999987416 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 49.28337342404806,
            "unit": "iter/sec",
            "range": "stddev: 0.0005079552672785621",
            "extra": "mean: 20.29081879999808 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.8163964457572312,
            "unit": "iter/sec",
            "range": "stddev: 0.002155109859973791",
            "extra": "mean: 355.0636493333362 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 4116.587239416541,
            "unit": "iter/sec",
            "range": "stddev: 0.00001700841087440809",
            "extra": "mean: 242.91966666586026 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 48.16686543529384,
            "unit": "iter/sec",
            "range": "stddev: 0.0007534967248848518",
            "extra": "mean: 20.761159999987438 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 323.49136030068644,
            "unit": "iter/sec",
            "range": "stddev: 0.000842423522100219",
            "extra": "mean: 3.0912726666656454 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 47.31484396456883,
            "unit": "iter/sec",
            "range": "stddev: 0.0006997186253352443",
            "extra": "mean: 21.13501633332741 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 24754.190880109883,
            "unit": "iter/sec",
            "range": "stddev: 0.000023812621914296014",
            "extra": "mean: 40.397200007191714 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 32405.247056519707,
            "unit": "iter/sec",
            "range": "stddev: 0.000014925180157507458",
            "extra": "mean: 30.85920000103215 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 23639.208936321324,
            "unit": "iter/sec",
            "range": "stddev: 0.000014808933561593424",
            "extra": "mean: 42.30260000213093 usec\nrounds: 5"
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
          "id": "2232a5c7b5832f894b8cad90eb3cdda6839cea45",
          "message": "fix: resolve pre-existing CI failures (#68)\n\n- Cherry-picks SAC dict fix for flax TrainState compatibility (fixes\ntest-medium)\n- Marks test-jaxatari-smoke as non-blocking since upstream\nk4ntz/JAXAtari removed all releases",
          "timestamp": "2026-05-29T22:48:44-06:00",
          "tree_id": "508424f2a5aaed3ef5287132a49d907c4bbffa95",
          "url": "https://github.com/andnp/jax-research-template/commit/2232a5c7b5832f894b8cad90eb3cdda6839cea45"
        },
        "date": 1780116842168,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 7.080028113658958,
            "unit": "iter/sec",
            "range": "stddev: 0.004229268384826119",
            "extra": "mean: 141.24237699999753 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9951388878426234,
            "unit": "iter/sec",
            "range": "stddev: 0.003733317180491225",
            "extra": "mean: 1.0048848579999874 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.03898609333873433,
            "unit": "iter/sec",
            "range": "stddev: 0.17411713304938092",
            "extra": "mean: 25.650172006500014 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.5865010442019961,
            "unit": "iter/sec",
            "range": "stddev: 0.007902318600401442",
            "extra": "mean: 1.7050268024000161 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 18732.90622353054,
            "unit": "iter/sec",
            "range": "stddev: 0.000014193299729207026",
            "extra": "mean: 53.381999998691754 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 44.51671426337536,
            "unit": "iter/sec",
            "range": "stddev: 0.0007471641270919116",
            "extra": "mean: 22.463472799984174 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.649828275184662,
            "unit": "iter/sec",
            "range": "stddev: 0.002170837293897776",
            "extra": "mean: 377.3829456666628 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 4532.036968886876,
            "unit": "iter/sec",
            "range": "stddev: 0.000040849383939944393",
            "extra": "mean: 220.65133335521145 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 43.80082282473139,
            "unit": "iter/sec",
            "range": "stddev: 0.0002748373682403577",
            "extra": "mean: 22.830621333336392 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 557.561019938567,
            "unit": "iter/sec",
            "range": "stddev: 0.0003392545263770592",
            "extra": "mean: 1.793525666679822 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 37.83806513895718,
            "unit": "iter/sec",
            "range": "stddev: 0.0005512327337786899",
            "extra": "mean: 26.428412666651486 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 40698.3842833097,
            "unit": "iter/sec",
            "range": "stddev: 0.000013077045667716728",
            "extra": "mean: 24.570999994466547 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 41243.23611965573,
            "unit": "iter/sec",
            "range": "stddev: 0.00001267503161338282",
            "extra": "mean: 24.246399993899104 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 27398.161026808997,
            "unit": "iter/sec",
            "range": "stddev: 0.00001544796700814383",
            "extra": "mean: 36.49880001148631 usec\nrounds: 5"
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
          "id": "0c699bc3cd8353f3f99c216b36a8f0ae5eb14e82",
          "message": "chore: migrate type-checker from ty to pyrefly (#64)\n\nReplace the Astral `ty` type-checker with Meta's `pyrefly` across the\nmonorepo core.\n\n## Changes\n\n- **`pyproject.toml`**: Replace `ty~=0.0` dep with `pyrefly>=1.0`;\nreplace `[tool.ty.environment]` + `extra-paths` with `[tool.pyrefly]` +\n`search-path` + `replace-imports-with-any` for flax/chex (suppresses\nfalse positives from flax's internal `nn.Module.__init__` scope\nparameter)\n- **`CONTRIBUTING.md`**: Update dev workflow command\n- **`ROADMAP.md`**: Update technical debt item\n- **`scripts/copilot_hooks.py`**: Replace `ty` invocation with `pyrefly\ncheck`\n- **`cli/src/research_cli/lifecycle.py`**: Harvest automation now\nupdates `[tool.pyrefly].search-path`\n- **`tests/medium/test_cli_harvest.py`**: Update fixture pyproject\ntemplate\n- **`libs/experiment-definition/src/experiment_definition/db.py`**: Fix\ndict invariance — `params`/`jax_config` changed to `Mapping[str,\nobject]`\n- **`tests/small/test_rl_agents_dqn_atari.py`**: Add `# pyrefly: ignore`\non intentional-error test call\n- **`libs/rl-agents/src/rl_agents/sac.py`**: Fix pre-existing SAC test\nfailure — wrap `log_alpha` in a dict for flax `TrainState` compatibility\n(newer flax's `apply_gradients` does `OVERWRITE_WITH_GRADIENT in grads`\nwhich requires a dict-like pytree, not a raw array)\n\n## Verification\n\n```\nuv run pyrefly check  # 0 errors\nuv run pytest tests/small tests/medium  # 618/618 pass\n```",
          "timestamp": "2026-05-29T22:55:18-06:00",
          "tree_id": "2d6da080ff1598ec12c59c2602edf6f9a5e2d7c9",
          "url": "https://github.com/andnp/jax-research-template/commit/0c699bc3cd8353f3f99c216b36a8f0ae5eb14e82"
        },
        "date": 1780117239681,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.267949854454546,
            "unit": "iter/sec",
            "range": "stddev: 0.04224626031321763",
            "extra": "mean: 159.54179966665075 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.914834755657732,
            "unit": "iter/sec",
            "range": "stddev: 0.0025465231396552585",
            "extra": "mean: 1.0930935819999945 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04250685092565429,
            "unit": "iter/sec",
            "range": "stddev: 0.020196740266052768",
            "extra": "mean: 23.525619475999974 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.5691395287987601,
            "unit": "iter/sec",
            "range": "stddev: 0.042404018053574605",
            "extra": "mean: 1.757038387600005 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 16502.902858071273,
            "unit": "iter/sec",
            "range": "stddev: 0.000017014624781785",
            "extra": "mean: 60.59540000933339 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 49.23341117133044,
            "unit": "iter/sec",
            "range": "stddev: 0.0008657509562945665",
            "extra": "mean: 20.31141000001071 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.8224339275344836,
            "unit": "iter/sec",
            "range": "stddev: 0.007169385573331967",
            "extra": "mean: 354.30413099999214 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 4059.4355491302267,
            "unit": "iter/sec",
            "range": "stddev: 0.000030324180945960363",
            "extra": "mean: 246.33966665987828 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 42.21375220484707,
            "unit": "iter/sec",
            "range": "stddev: 0.001654462503101869",
            "extra": "mean: 23.68896266665388 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 314.3790698993514,
            "unit": "iter/sec",
            "range": "stddev: 0.0015266946131605552",
            "extra": "mean: 3.1808733333302066 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 37.86326680779515,
            "unit": "iter/sec",
            "range": "stddev: 0.0009787000847899414",
            "extra": "mean: 26.4108220000215 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 31968.900650176525,
            "unit": "iter/sec",
            "range": "stddev: 0.000011535214570699947",
            "extra": "mean: 31.28040000319743 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 32266.183111577746,
            "unit": "iter/sec",
            "range": "stddev: 0.000011519758825075095",
            "extra": "mean: 30.99219999285197 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 22631.39790296834,
            "unit": "iter/sec",
            "range": "stddev: 0.000012563366837045051",
            "extra": "mean: 44.18639998675644 usec\nrounds: 5"
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
          "id": "678a8e45223e85d765618976ae6aa392588067f7",
          "message": "refactor: replace TYPE_CHECKING apply stubs with TypedApply generic mixin (#69)\n\nReplace the repetitive `if TYPE_CHECKING: def apply(...)` pattern across\n`jax-nn` and `rl-agents` modules with a `TypedApply[T]` generic mixin\nthat provides type-safe `apply()` signatures at both runtime and\ntype-check time.\n\n### Changes\n- **New**: `libs/jax-nn/src/jax_nn/typed_module.py` — `TypedApply[T_co]`\nmixin using `Generic[T_co]` with a typed `apply()` that delegates to\n`nn.Module.apply()`\n- **Simplified**: `heads.py`, `layers.py`, `dqn.py`, `dueling_dqn.py`,\n`rainbow.py`, `networks.py` — replaced `if TYPE_CHECKING` stubs with\n`TypedApply[jax.Array]` in class bases\n- **Updated**: `libs/rl-components/pyproject.toml` — added `jax-nn`\ndependency for the mixin import\n\nNet result: **-98 lines**, **+56 lines** — removes boilerplate while\npreserving full type safety.",
          "timestamp": "2026-05-29T23:02:51-06:00",
          "tree_id": "f4ff3728fdd3693a530caf9cdefb5aca453be6de",
          "url": "https://github.com/andnp/jax-research-template/commit/678a8e45223e85d765618976ae6aa392588067f7"
        },
        "date": 1780117691562,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 4.878500989765732,
            "unit": "iter/sec",
            "range": "stddev: 0.001784390325909763",
            "extra": "mean: 204.98099766666655 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9225101912883357,
            "unit": "iter/sec",
            "range": "stddev: 0.006400495494382054",
            "extra": "mean: 1.0839988646666825 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.0419468665697709,
            "unit": "iter/sec",
            "range": "stddev: 0.09906403016311073",
            "extra": "mean: 23.839682955499995 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.5713538055714195,
            "unit": "iter/sec",
            "range": "stddev: 0.045800486830812476",
            "extra": "mean: 1.7502290003999974 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 17160.076051802185,
            "unit": "iter/sec",
            "range": "stddev: 0.000013617122401070826",
            "extra": "mean: 58.27480000561991 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 50.08161600636045,
            "unit": "iter/sec",
            "range": "stddev: 0.00018333040607369729",
            "extra": "mean: 19.967406799992204 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.768213175770915,
            "unit": "iter/sec",
            "range": "stddev: 0.0075012224298685376",
            "extra": "mean: 361.2438553333277 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 3792.1976797802145,
            "unit": "iter/sec",
            "range": "stddev: 0.000035815780334742036",
            "extra": "mean: 263.69933332640966 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 42.27932401956621,
            "unit": "iter/sec",
            "range": "stddev: 0.001026259730086567",
            "extra": "mean: 23.65222299999914 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 324.47505614112987,
            "unit": "iter/sec",
            "range": "stddev: 0.0008671631377937574",
            "extra": "mean: 3.0819010000110816 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 37.222778702748165,
            "unit": "iter/sec",
            "range": "stddev: 0.0009920254062540342",
            "extra": "mean: 26.865270000011304 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 32812.059075792546,
            "unit": "iter/sec",
            "range": "stddev: 0.000011564161730417852",
            "extra": "mean: 30.476600011297705 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 31795.693582393953,
            "unit": "iter/sec",
            "range": "stddev: 0.000013260017709059972",
            "extra": "mean: 31.450800008769875 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 22470.193288862785,
            "unit": "iter/sec",
            "range": "stddev: 0.00001316359624779432",
            "extra": "mean: 44.50339999948483 usec\nrounds: 5"
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
          "id": "03f02736f3cefb88a0a5ec65bcab5b9343cfc347",
          "message": "feat: migrate atari from jaxatari to ale-py; add PythonEnvBridge and FrameStackWrapper (#67)\n\n## Summary\n\nReplaces `jaxatari` (pure-JAX Atari reimplementation) with `ale-py`\n(real Atari via ALE) and introduces composable non-JIT environment\nbridge tooling.\n\n## New modules in `rl-components`\n\n- `frame_stack.py` — composable `FrameStackWrapper` + `FrameStackState`;\nworks with any `EnvProtocol`; handles episode-boundary reset via\n`lax.cond`\n- `python_env_bridge.py` — `PythonEnvBridge` using\n`jax.experimental.io_callback(ordered=True)`; serializes ALE emulator\nstate as `jnp.uint8[N]` bytes so it can live inside `lax.scan`;\nauto-resets on episode end inside the callback\n- `atari_ale.py` — `AleAtariConfig` + `make_atari_adapter()` composing\nthe two above\n\n## Infrastructure\n\n- Remove `jaxatari` from all deps; add `ale-py>=0.10`,\n`gymnasium[atari]>=1.0`\n- Delete `scripts/install_jaxatari_assets.py` and the two old JAXAtari\nregression smoke tests\n- Replace `test-jaxatari-smoke` CI job with `test-ale-atari-smoke` (no\ncustom asset-download script; uses `autorom[accept-rom-license]` for ROM\ninstall)",
          "timestamp": "2026-05-29T23:31:25-06:00",
          "tree_id": "725419a59648fe7c7c8be63e171058e21af84b81",
          "url": "https://github.com/andnp/jax-research-template/commit/03f02736f3cefb88a0a5ec65bcab5b9343cfc347"
        },
        "date": 1780119390625,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.295681041928167,
            "unit": "iter/sec",
            "range": "stddev: 0.02972162831016242",
            "extra": "mean: 158.8390506666665 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9980528351811515,
            "unit": "iter/sec",
            "range": "stddev: 0.0026617793058088123",
            "extra": "mean: 1.001950963666663 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.03926819340130192,
            "unit": "iter/sec",
            "range": "stddev: 0.03738284966752349",
            "extra": "mean: 25.46590289449999 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.5845739685677733,
            "unit": "iter/sec",
            "range": "stddev: 0.013643683013292825",
            "extra": "mean: 1.7106475035999893 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 16568.69047656623,
            "unit": "iter/sec",
            "range": "stddev: 0.000030358598552640072",
            "extra": "mean: 60.35480000150528 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 45.57231218080091,
            "unit": "iter/sec",
            "range": "stddev: 0.0000870055004472155",
            "extra": "mean: 21.94314820000045 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.661168847948637,
            "unit": "iter/sec",
            "range": "stddev: 0.00375443842299456",
            "extra": "mean: 375.7747280000103 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 5052.2827055309845,
            "unit": "iter/sec",
            "range": "stddev: 0.000035109380241053425",
            "extra": "mean: 197.93033333333673 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 43.1742223642554,
            "unit": "iter/sec",
            "range": "stddev: 0.0014894728802256581",
            "extra": "mean: 23.161969000000227 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 362.4206072362531,
            "unit": "iter/sec",
            "range": "stddev: 0.000662951012523884",
            "extra": "mean: 2.759224999995998 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 37.740381700174495,
            "unit": "iter/sec",
            "range": "stddev: 0.0006814759739146851",
            "extra": "mean: 26.49681733333864 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 39783.26079065795,
            "unit": "iter/sec",
            "range": "stddev: 0.000013334326742032273",
            "extra": "mean: 25.136200002862097 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 40649.08456772517,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134811929263754",
            "extra": "mean: 24.600800009011436 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 27692.85303451841,
            "unit": "iter/sec",
            "range": "stddev: 0.00001547142887033157",
            "extra": "mean: 36.11039999213972 usec\nrounds: 5"
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
          "id": "266879ff5c3a8fda9543996ba09eb4d9dad43261",
          "message": "feat: add research-runner plan-execute-record lifecycle library (#65)\n\n## Summary\n\nExtracts the experiment execution lifecycle from missingness-rl into a\nreusable research-runner library in core/libs/research-runner/.\n\n## Changes\n\n### experiment-definition\n- Expose Experiment.name property for downstream consumers\n\n### research-runner (new library)\n- **types.py**: ExecutionContext and ExecutionResult frozen dataclasses\n— the contract between runner and user-provided training callbacks\n- **runner.py**: Core lifecycle — run_experiment() (batch loop until all\nruns satisfied), execute_batch() (single execution lifecycle: create\ndir, call train_fn, record artifacts)\n- **git.py**: capture_git_metadata() for recording commit SHA and\nworking-tree diff at execution time\n- **pyproject.toml**: Package definition depending on\nexperiment-definition\n\n### Workspace registration\n- Added research-runner to core pyproject.toml dependencies\n- Updated uv.lock\n\n### Tests\n- 9 integration tests covering: happy-path batch execution, failure\nrecording, already-satisfied experiments, on_batch_complete callback,\nmetrics DB path derivation, and git metadata capture\n\n## Design\n\nThe runner accepts a train_fn(ctx: ExecutionContext) -> ExecutionResult\ncallback, keeping all project-specific training logic in the consumer.\nThe runner handles:\n1. Resolving experiments and hyperparameters from the database\n2. Planning execution batches\n3. Creating execution directories\n4. Calling the training callback with full context\n5. Recording execution artifacts (including git metadata and metrics DB\npath)\n\nBuilt on ADR-007 (database-centric orchestration) and ADR-008\n(relational experiment schema).",
          "timestamp": "2026-05-30T12:19:28-06:00",
          "tree_id": "a6ae62a2991526a7df18938605f1fee3a3868a95",
          "url": "https://github.com/andnp/jax-research-template/commit/266879ff5c3a8fda9543996ba09eb4d9dad43261"
        },
        "date": 1780165496087,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.567149732517274,
            "unit": "iter/sec",
            "range": "stddev: 0.01861978676147975",
            "extra": "mean: 179.62513099999455 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9878453834561014,
            "unit": "iter/sec",
            "range": "stddev: 0.008012220309738109",
            "extra": "mean: 1.0123041689999848 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.03928217605131156,
            "unit": "iter/sec",
            "range": "stddev: 0.029000205549699374",
            "extra": "mean: 25.456838203000004 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.5848018558918963,
            "unit": "iter/sec",
            "range": "stddev: 0.015834281844120186",
            "extra": "mean: 1.7099808933999951 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 19063.959579444276,
            "unit": "iter/sec",
            "range": "stddev: 0.000014857120740632045",
            "extra": "mean: 52.45500001365144 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 45.46426075374767,
            "unit": "iter/sec",
            "range": "stddev: 0.00007028128223290294",
            "extra": "mean: 21.995298800004548 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.6173234130041725,
            "unit": "iter/sec",
            "range": "stddev: 0.004371005510338796",
            "extra": "mean: 382.06971099998555 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 5084.849183849857,
            "unit": "iter/sec",
            "range": "stddev: 0.00001156905914686717",
            "extra": "mean: 196.6626666482322 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 42.699809309789885,
            "unit": "iter/sec",
            "range": "stddev: 0.0009868091020424179",
            "extra": "mean: 23.41930833332147 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 425.64535994961443,
            "unit": "iter/sec",
            "range": "stddev: 0.0004533843777648917",
            "extra": "mean: 2.349373666656144 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 38.24434051641892,
            "unit": "iter/sec",
            "range": "stddev: 0.00040364701369090245",
            "extra": "mean: 26.147659666681495 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 40174.5181163781,
            "unit": "iter/sec",
            "range": "stddev: 0.000013663130327639012",
            "extra": "mean: 24.891399993975938 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 40434.75452679379,
            "unit": "iter/sec",
            "range": "stddev: 0.000013837360103639993",
            "extra": "mean: 24.731199971483875 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 24208.15136863407,
            "unit": "iter/sec",
            "range": "stddev: 0.000024719151956880047",
            "extra": "mean: 41.30840000016178 usec\nrounds: 5"
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
          "id": "a73c0682a8253e1f4115041be4233f4f95616d7f",
          "message": "feat(types): enable pyrefly strict mode with full annotation pass (#73)\n\n## Summary\n\nEnables pyrefly strict mode (preset = \"strict\") across the entire core\ncodebase, achieving **0 type errors** across all 13 libraries.\n\n## What Changed\n\n### Pyrefly config\n- `preset = \"strict\"` with `replace-imports-with-any = [\"chex.*\",\n\"flax.linen.*\"]`\n- Suppressed `implicit-any-empty-container` and\n`implicit-any-type-argument` (inferred type args on JAX containers)\n\n### Annotation pass (14 commits)\nFull type annotation across: rl-components, rl-agents (DQN family, PPO,\nSAC), jax-replay, research-cluster, research-store, all examples and\ntests.\n\n### Quality improvements (5 commits)\n\n**Option A — Shared GymEnv Protocol** (`96278cc`)  \nExtracted duplicated `_ObservationSpace`, `_ActionSpace`, `_EnvLike`\nProtocols from 8 agents into a single `rl_components.gym_env` module\nwith `ObservationSpace`, `DiscreteActionSpace`, `ContinuousActionSpace`,\n`GymEnv[ActionSpaceT]`.\n\n**Option B — TrainOutput TypedDicts** (`ae0c930`)  \nAdded typed output classes for each agent (`DQNTrainOutput`,\n`PPOTrainOutput`, `SACTrainOutput`, etc.) replacing `dict[str, Any]`\nreturn types.\n\n**Option C — FrameStack ObsT bound** (`6e78b06`)  \nBounded `FrameStackWrapper[ObsT: jax.Array, ...]` so array-stacking\noperations are type-safe without `# type: ignore`.\n\n**Option D — ALEInterface Protocol** (`c1181a3`)  \nDefined `_ALEInterface` and `_ALEEnv` Protocols in\n`python_env_bridge.py` to remove 4 `attr-defined` ignores on ALE state\nmanagement.\n\n**Option E — SAC alpha cast** (`e667c12`)  \nReplaced `# type: ignore[assignment]` with explicit `cast(jax.Array,\n...)` for alpha parameter extraction.\n\n## Test Results\n- 455 small tests passing\n- 229 medium tests passing\n- 0 pyrefly errors (rl-* code)",
          "timestamp": "2026-05-30T16:20:16-06:00",
          "tree_id": "89c87a6dbb35a584f5a7048abb5d254dd28d66b4",
          "url": "https://github.com/andnp/jax-research-template/commit/a73c0682a8253e1f4115041be4233f4f95616d7f"
        },
        "date": 1780179940557,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.586764109615593,
            "unit": "iter/sec",
            "range": "stddev: 0.009137888658364502",
            "extra": "mean: 151.81961633333194 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9926786426098314,
            "unit": "iter/sec",
            "range": "stddev: 0.00408212572130276",
            "extra": "mean: 1.007375355000003 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.03899857481123998,
            "unit": "iter/sec",
            "range": "stddev: 0.16185596722455176",
            "extra": "mean: 25.641962683000017 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.5864311584593249,
            "unit": "iter/sec",
            "range": "stddev: 0.007452302945261032",
            "extra": "mean: 1.7052299925999932 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 16664.389199993067,
            "unit": "iter/sec",
            "range": "stddev: 0.000028904530258105928",
            "extra": "mean: 60.008200000538636 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 45.568549265369406,
            "unit": "iter/sec",
            "range": "stddev: 0.00005721238261396381",
            "extra": "mean: 21.944960199994057 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.6495507852682993,
            "unit": "iter/sec",
            "range": "stddev: 0.003000222359899747",
            "extra": "mean: 377.4224693333205 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 5066.283880384837,
            "unit": "iter/sec",
            "range": "stddev: 0.000035865343948675046",
            "extra": "mean: 197.38333334847388 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 44.842950793545874,
            "unit": "iter/sec",
            "range": "stddev: 0.0003486848014237694",
            "extra": "mean: 22.30004899998524 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 488.3002447542898,
            "unit": "iter/sec",
            "range": "stddev: 0.0003989190568035166",
            "extra": "mean: 2.047920333325237 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 40.25799145956013,
            "unit": "iter/sec",
            "range": "stddev: 0.00019608727963334405",
            "extra": "mean: 24.839788666668028 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 40342.752020669155,
            "unit": "iter/sec",
            "range": "stddev: 0.000013393262358961491",
            "extra": "mean: 24.787600000308885 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 32793.33637194159,
            "unit": "iter/sec",
            "range": "stddev: 0.000015809885212174555",
            "extra": "mean: 30.494000020553358 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 26896.180739513162,
            "unit": "iter/sec",
            "range": "stddev: 0.000016823148805927922",
            "extra": "mean: 37.180000003900204 usec\nrounds: 5"
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
          "id": "62196f04a05850d5518d59d53e9d00847829b60c",
          "message": "feat(analysis): extract statistical primitives and reporting API (#75)\n\n## Summary\n\nExtract reusable non-parametric statistical primitives and add a\nhigh-level reporting API for experiment analysis.\n\n## New Modules\n- **`statistics.py`**: Non-parametric tolerance interval and median-run\nstatistical primitives\n- **`reporting.py`**: High-level API for pairwise comparisons, bakeoff\nreports, and hyperparameter sensitivity analysis\n- `compare_pairwise()` — A/B comparison with bootstrap CI and effect\nsizes\n  - `compare_bakeoff()` — multi-algorithm tournament ranking\n  - `analyze_hypers()` — hyperparameter sensitivity analysis\n\n## Supporting Changes\n- Research reporting API specification (design doc)\n- Research-plot setuptools package discovery fix\n- Result artifact examples (comparison plots)",
          "timestamp": "2026-06-07T13:56:14-06:00",
          "tree_id": "9489477062d25fada9fdbbd1432074ef1d6ff264",
          "url": "https://github.com/andnp/jax-research-template/commit/62196f04a05850d5518d59d53e9d00847829b60c"
        },
        "date": 1780862493712,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.885968137429774,
            "unit": "iter/sec",
            "range": "stddev: 0.036055641671706315",
            "extra": "mean: 169.89558500000138 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.8411946133779874,
            "unit": "iter/sec",
            "range": "stddev: 0.013035517534534946",
            "extra": "mean: 1.1887855486666723 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.04162777743847762,
            "unit": "iter/sec",
            "range": "stddev: 0.08102413174975286",
            "extra": "mean: 24.022421122000004 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.5625657388585051,
            "unit": "iter/sec",
            "range": "stddev: 0.02317590900551736",
            "extra": "mean: 1.7775700347999988 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 16208.190973174444,
            "unit": "iter/sec",
            "range": "stddev: 0.00001632155792829236",
            "extra": "mean: 61.69719999320478 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 46.05171054473659,
            "unit": "iter/sec",
            "range": "stddev: 0.0005667665553563646",
            "extra": "mean: 21.714719999999943 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.6987410092523394,
            "unit": "iter/sec",
            "range": "stddev: 0.0026290496257521567",
            "extra": "mean: 370.5431519999915 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 3906.428027347849,
            "unit": "iter/sec",
            "range": "stddev: 0.000045497444408602666",
            "extra": "mean: 255.98833333143983 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 41.91775928450213,
            "unit": "iter/sec",
            "range": "stddev: 0.0011687870262168556",
            "extra": "mean: 23.856236999999208 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 325.02747294797797,
            "unit": "iter/sec",
            "range": "stddev: 0.0006535401718713725",
            "extra": "mean: 3.076662999992171 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 35.32597336302804,
            "unit": "iter/sec",
            "range": "stddev: 0.0015876246066375537",
            "extra": "mean: 28.307783333341757 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 30421.582278942693,
            "unit": "iter/sec",
            "range": "stddev: 0.000013188362106399032",
            "extra": "mean: 32.871400009071294 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 27149.71438654514,
            "unit": "iter/sec",
            "range": "stddev: 0.00002400278307773404",
            "extra": "mean: 36.832799997910115 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 21347.633403697873,
            "unit": "iter/sec",
            "range": "stddev: 0.00001508919085128586",
            "extra": "mean: 46.84359999487242 usec\nrounds: 5"
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
          "id": "daabe7234af3587f6f512777c60b2abf401de698",
          "message": "feat(#27): add CI workflow template and TD3 algorithm choice (#72)\n\nCloses #27\nCloses #61\n\n## Summary\nAdds a CI workflow to the copier project template so every new student\nproject gets automated checks immediately.\n\n### Changes:\n- **`templates/{{project_name}}/.github/workflows/ci.yml.jinja`** (new):\n3 parallel jobs — lint (ruff), typecheck (pyrefly), test (pytest). Uses\n`{{ python_version }}` from copier config.\n- **`templates/copier.yml`**: Added `td3` to algorithm choices (now that\nTD3 is implemented). Removed `3.12` from python_version choices since\ncore requires `>=3.13,<3.14`.\n- **`templates/{{project_name}}/train.py.jinja`**: Added TD3 training\ntemplate block.\n- **`docs/getting_started.md`**: Updated Python prerequisite from \"3.12\nor 3.13\" to \"3.13\".",
          "timestamp": "2026-06-07T14:37:52-06:00",
          "tree_id": "b0f7c558a611963ef4ecc4d6dbb94276b43102ab",
          "url": "https://github.com/andnp/jax-research-template/commit/daabe7234af3587f6f512777c60b2abf401de698"
        },
        "date": 1780864992705,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 5.403006198069383,
            "unit": "iter/sec",
            "range": "stddev: 0.01460931704517436",
            "extra": "mean: 185.08214933333278 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.9946430791542955,
            "unit": "iter/sec",
            "range": "stddev: 0.015423399195600797",
            "extra": "mean: 1.0053857720000015 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.03913402150892104,
            "unit": "iter/sec",
            "range": "stddev: 0.06354311568698838",
            "extra": "mean: 25.553213328000005 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.5882091748622384,
            "unit": "iter/sec",
            "range": "stddev: 0.013587491235567377",
            "extra": "mean: 1.7000754880000044 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 18838.706761404428,
            "unit": "iter/sec",
            "range": "stddev: 0.000013593796709713618",
            "extra": "mean: 53.0821999973341 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 42.81059611915888,
            "unit": "iter/sec",
            "range": "stddev: 0.0004912378259553816",
            "extra": "mean: 23.35870299999101 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.646712716021355,
            "unit": "iter/sec",
            "range": "stddev: 0.0034586848624304813",
            "extra": "mean: 377.82717933332793 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 4550.032153278535,
            "unit": "iter/sec",
            "range": "stddev: 0.000028981703861822714",
            "extra": "mean: 219.77866668028886 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 44.46422937968521,
            "unit": "iter/sec",
            "range": "stddev: 0.0004575776283212456",
            "extra": "mean: 22.489988333338335 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 454.8386876672401,
            "unit": "iter/sec",
            "range": "stddev: 0.0006238070501696487",
            "extra": "mean: 2.1985816666756364 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 39.18971194815175,
            "unit": "iter/sec",
            "range": "stddev: 0.0006510153354999826",
            "extra": "mean: 25.51690100001262 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 39366.35909956811,
            "unit": "iter/sec",
            "range": "stddev: 0.000012974113314020317",
            "extra": "mean: 25.40239999007099 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 40685.468773124696,
            "unit": "iter/sec",
            "range": "stddev: 0.000011812424566853926",
            "extra": "mean: 24.578800002927892 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 27266.084262708868,
            "unit": "iter/sec",
            "range": "stddev: 0.00001546963828487223",
            "extra": "mean: 36.67560000053527 usec\nrounds: 5"
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
          "id": "5a0883197f6160c32deef719d98068e8138cd7b8",
          "message": "feat(#20): implement TD3 (Twin Delayed DDPG) (#70)\n\nCloses #20\n\n## Summary\nImplements the TD3 (Twin Delayed Deep Deterministic Policy Gradient)\nalgorithm in `rl-agents`.\n\n### Key components:\n- **TD3Config**: chex dataclass with TD3-specific hyperparameters\n(policy delay, target noise, noise clip, exploration noise)\n- **Actor**: deterministic policy with tanh-bounded output scaled to\naction space\n- **Critic**: twin Q-networks for reducing overestimation bias\n- **make_train**: JIT-compatible training loop with:\n  - Delayed policy updates (every `POLICY_DELAY` steps)\n  - Target policy smoothing (clipped noise on target actions)\n  - Soft target network updates (Polyak averaging)\n  - Configurable exploration noise\n\n### Tests:\n- 9 small tests: config defaults, network shapes, actor output bounds,\nsoft update correctness\n- 5 medium tests: make_train JIT compilation, gradient flow verification",
          "timestamp": "2026-06-07T15:41:47-06:00",
          "tree_id": "3fc4bd5108f8c35b1f31df1f8558584f1ebf8874",
          "url": "https://github.com/andnp/jax-research-template/commit/5a0883197f6160c32deef719d98068e8138cd7b8"
        },
        "date": 1780868851159,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_all_bench.py::test_ppo_speed",
            "value": 6.815915077188092,
            "unit": "iter/sec",
            "range": "stddev: 0.010679750271219812",
            "extra": "mean: 146.7154429999956 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_dqn_speed",
            "value": 0.8566289199300864,
            "unit": "iter/sec",
            "range": "stddev: 0.017601551198642908",
            "extra": "mean: 1.167366611999995 sec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_all_bench.py::test_sac_speed",
            "value": 0.03427273521420608,
            "unit": "iter/sec",
            "range": "stddev: 0.5904409886996154",
            "extra": "mean: 29.17771207199999 sec\nrounds: 2"
          },
          {
            "name": "tests/performance/test_ppo_bench.py::test_ppo_speed",
            "value": 0.5851953144071751,
            "unit": "iter/sec",
            "range": "stddev: 0.018367841150177817",
            "extra": "mean: 1.7088311805999978 sec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_env_only_rollout_speed",
            "value": 10217.051032178966,
            "unit": "iter/sec",
            "range": "stddev: 0.00006550635677051471",
            "extra": "mean: 97.87559999949735 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_policy_and_env_rollout_speed",
            "value": 43.67047475535864,
            "unit": "iter/sec",
            "range": "stddev: 0.0007138117561769875",
            "extra": "mean: 22.89876639999875 msec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_micro_train_replay_and_update_speed",
            "value": 2.348326852342606,
            "unit": "iter/sec",
            "range": "stddev: 0.009414609049617367",
            "extra": "mean: 425.83510000000047 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_replay_sampling_only_speed",
            "value": 3318.283076177717,
            "unit": "iter/sec",
            "range": "stddev: 0.000033194171881290756",
            "extra": "mean: 301.3606666589415 usec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_loss_and_grad_fixed_batch_speed",
            "value": 37.60467260619693,
            "unit": "iter/sec",
            "range": "stddev: 0.0010913139269743944",
            "extra": "mean: 26.592440000001716 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_optimizer_apply_fixed_grads_speed",
            "value": 493.6836468579911,
            "unit": "iter/sec",
            "range": "stddev: 0.0003026919716489869",
            "extra": "mean: 2.025588666678383 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py::test_fake_full_learn_step_speed",
            "value": 31.89363888175844,
            "unit": "iter/sec",
            "range": "stddev: 0.0007250311505014591",
            "extra": "mean: 31.35421466667291 msec\nrounds: 3"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_canonical_env_rollout_speed",
            "value": 34682.00074162156,
            "unit": "iter/sec",
            "range": "stddev: 0.00001972508745724116",
            "extra": "mean: 28.833399994709907 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_rollout_speed",
            "value": 37565.740037246054,
            "unit": "iter/sec",
            "range": "stddev: 0.00001637873008170244",
            "extra": "mean: 26.620000005550537 usec\nrounds: 5"
          },
          {
            "name": "tests/performance/test_rl_components_gymnax_bridge_bench.py::test_gymnax_bridge_log_wrapper_rollout_speed",
            "value": 25452.287147540635,
            "unit": "iter/sec",
            "range": "stddev: 0.000021809257168190464",
            "extra": "mean: 39.289199992254 usec\nrounds: 5"
          }
        ]
      }
    ]
  }
}