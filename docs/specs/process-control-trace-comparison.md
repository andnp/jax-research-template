# Process-control trace comparison

## Purpose

Compare a reference process trace with a candidate trace after both have been
canonicalized onto the same fixed time grid. The comparison layer reports
process-relevant errors without knowing how either trace was stored or produced.

## Comparison specification

A comparison specification contains named signal pairs. Each pair declares:

- the reference signal name;
- the candidate signal name;
- the expected unit;
- an optional normalization scale.

Reference and candidate signals may use different names, but their units must
match the declared unit. The traces must have equal lengths, equal sample
periods, and equal timestamps within an explicit tolerance. Comparison never
resamples implicitly.

The specification also contains a leading warmup count. Warmup rows are
excluded from every metric while remaining available in the source traces.

## Validity and segments

A row contributes to a signal pair only when both values are valid and finite.
The reference trace supplies the segment identity. Invalid rows and segment
boundaries are never bridged when computing a metric.

Results include one row per requested signal and segment, plus an overall
summary. Basic error metrics are weighted by valid sample count. Correlation
and R² use accumulated within-segment sufficient statistics so a gap cannot
create an artificial trajectory between two observations.

Pairs with no valid samples report zero count and undefined metrics. Pairs with
constant reference or candidate values report their error metrics normally but
mark correlation and R² as undefined. Undefined values use a stable numeric
placeholder in the JAX payload and are identified by explicit boolean flags.

## Initial metrics

For each signal pair and segment, compute:

- valid sample count;
- mean signed error (bias);
- mean absolute error;
- root mean squared error;
- maximum absolute error;
- correlation;
- R²;
- normalized RMSE when a scale is supplied.

The first version does not compute lag searches, frequency-response metrics,
or dynamic model-fit statistics. Those belong in later system-identification
consumers and must use segment-aware windows as well.

## Scope boundary

The comparison API consumes typed traces and returns typed numeric results. It
does not load TSDB data, infer units from customer tags, select model families,
or write reports. Project adapters may project replay diagnostics into a
simulated `UniformTrace` before calling it.
