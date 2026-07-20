"""WP4 analyzers: fixed, versioned code that turns RawRun artifacts into
unit-checked ObservationBundles (PLAN.md stage 7, 'analyze').

Analyzers never judge validity — that is the gates' job. Each analyzer
returns (ObservationBundle, diagnostics); property gates consume the
diagnostics, and only gated bundles ever reach scoring or narration.
"""
from analyzers.eos import ANALYZER_EOS, analyze_eos
from analyzers.elastic import ANALYZER_ELASTIC, analyze_elastic

__all__ = ["ANALYZER_EOS", "analyze_eos", "ANALYZER_ELASTIC", "analyze_elastic"]
