"""WP5: legacy Callaway engine behind the generic execute/analyze
interface. The physics (engines/callaway.py) is untouched — this wrapper
produces a contracts.RawRun so the v2 loop treats Callaway exactly like
LAMMPS. PLAN.md: 'the current run(plan) interface will be retained behind
an adapter during migration.'
"""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from contracts import ArtifactRef, EnvironmentLock, RawRun
from engines.callaway import CallawayEngine

CALLAWAY_ENGINE_VERSION = "callaway_rta_si/1.0"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class CallawayAdapter:
    engine = "callaway_rta_si"

    def execute(self, plan, workdir, seed=0) -> RawRun:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        t0 = time.time()
        plan_json = json.dumps(plan.to_dict(), sort_keys=True).encode()
        try:
            results = CallawayEngine().run(plan, np.random.default_rng(seed))
            raw = {"callaway": results, "engine_version": CALLAWAY_ENGINE_VERSION}
            status = "completed"
        except Exception as e:
            (workdir / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
            raw, status = {"error": str(e)}, "failed"
        (workdir / "raw_outputs.json").write_text(json.dumps(raw, indent=2))

        artifacts = [ArtifactRef(path=p.name, sha256=_sha(p.read_bytes()),
                                 kind="json" if p.suffix == ".json" else "log")
                     for p in sorted(workdir.iterdir()) if p.is_file()]
        return RawRun(
            run_id=f"{plan.plan_id}_callaway", plan_id=plan.plan_id,
            plan_sha256=_sha(plan_json),
            capability_id="si_kappa_callaway", engine=self.engine, seed=seed,
            started_utc=started, finished_utc=datetime.now(timezone.utc),
            exit_status=status, artifacts=artifacts,
            environment=EnvironmentLock(python=platform.python_version(),
                                        platform=platform.platform(),
                                        engine_version=CALLAWAY_ENGINE_VERSION),
            resource_usage={"walltime_s": time.time() - t0})
