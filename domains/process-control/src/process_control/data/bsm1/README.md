# BSM1 weather influent files

`dryinfluent.csv`, `raininfluent.csv` and `storminfluent.csv` are the IWA Benchmark
Simulation Model no. 1 dynamic influent profiles: 14 days at 15-minute samples of dry
weather, dry weather with a long rain event, and dry weather with two storm events.
They are described in Alex et al. (2008), *Benchmark Simulation Model no. 1 (BSM1)*,
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf, and distributed with
the benchmark at http://www.benchmarkwwtp.org. These copies come from the RL Core
`rl_env` BSM1 environment.

Each row has 22 comma-separated numbers and no header:

| Columns | Content | Units |
|---|---|---|
| 1 | time | d |
| 2-14 | S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK | g COD/m³, g N/m³, mol/m³ |
| 15 | TSS, 0.75 (X_I + X_S + X_BH + X_BA + X_P) | g SS/m³ |
| 16 | flow | m³/d |
| 17 | temperature, 15 throughout | °C |
| 18-22 | unused, zero | |
