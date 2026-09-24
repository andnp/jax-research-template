"""Statistical reports for RL experiments: bakeoffs, bias-corrected champions, and sensitivity slices.

These functions only compute. Wrap results in a report document (research_analysis.report) to serialize,
render, or plot them.
"""

from __future__ import annotations

from research_analysis._reporting_bakeoff import bakeoff_report as bakeoff_report
from research_analysis._reporting_hypers import best_hypers as best_hypers
from research_analysis._reporting_hypers import slice_hypers as slice_hypers
from research_analysis._reporting_models import BakeoffReport as BakeoffReport
from research_analysis._reporting_models import BlockKey as BlockKey
from research_analysis._reporting_models import CellSummary as CellSummary
from research_analysis._reporting_models import HyperparameterConfigReport as HyperparameterConfigReport
from research_analysis._reporting_models import HyperparameterDifferenceReport as HyperparameterDifferenceReport
from research_analysis._reporting_models import HyperparameterSliceReport as HyperparameterSliceReport
from research_analysis._reporting_models import PairedDifference as PairedDifference
from research_analysis._reporting_models import RunScore as RunScore
from research_analysis._reporting_models import StatisticalTestDetails as StatisticalTestDetails
