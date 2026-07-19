"""WP3 ASE/LAMMPS adapter: execute(plan) -> contracts.RawRun.

Discipline:
  * fixed reviewed templates only (templates.py); plans fill allowlisted
    numeric placeholders; pair/model blocks come from the pinned registry;
  * subprocess isolation with hard timeouts from the plan resource budget;
  * every artifact checksummed into the RawRun; structure lineage recorded;
  * fail closed: nonzero exit, timeout, missing artifacts, or missing
    sentinel outputs -> RawRun with exit_status failed/timeout, never a
    silent partial success. Unexpected exceptions also land in a failed
    RawRun (with error.txt) rather than escaping the adapter.
  * NO physics interpretation here: EOS/elastic fitting is WP4 analyzer
    work; this module only produces raw, checksummed evidence.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from contracts import ArtifactRef, EnvironmentLock, RawRun
from engines.lammps_adapter import structures as st
from engines.lammps_adapter.models import get_model, pair_blocks
from engines.lammps_adapter.templates import parse_sentinels, render

ADAPTER_VERSION = "0.1"

_KIND_BY_SUFFIX = {".in": "other", ".data": "structure", ".json": "json",
                   ".lammps": "log", ".txt": "log"}

_REQUIRED_SENTINELS = {
    "relax_v1": {"natoms", "pe_eV", "vol_A3", "lx_A"},
    "static_v1": {"natoms", "pe_eV", "vol_A3", "pxx_bar", "pyy_bar",
                  "pzz_bar", "pxy_bar", "pxz_bar", "pyz_bar"},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class LammpsAdapter:
    engine = "lammps"

    def __init__(self, binary=None, allow_unverified=False):
        self.binary = binary or os.environ.get("AIPHYS_LMP_BIN", "lmp")
        self.allow_unverified = allow_unverified

    # ------------------------------------------------------------ probing

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def version_info(self) -> dict:
        """Best-effort parse of `lmp -h`: version line + installed packages."""
        try:
            out = subprocess.run([self.binary, "-h"], capture_output=True,
                                 text=True, timeout=30).stdout
        except Exception as e:
            return {"binary": self.binary, "error": str(e)}
        version = next((ln.strip() for ln in out.splitlines()
                        if "Large-scale Atomic" in ln or ln.startswith("LAMMPS")), "")
        packages, in_pkg = [], False
        for ln in out.splitlines():
            if "Installed packages" in ln:
                in_pkg = True
                continue
            if in_pkg:
                if ln.strip() == "" and packages:
                    break
                packages += ln.split()
        return {"binary": self.binary,
                "binary_path": shutil.which(self.binary),
                "version_line": version, "packages": packages}

    # ------------------------------------------------------------ subruns

    def _run_one(self, workdir: Path, label: str, template: str,
                 values: dict, model, data_text: str, timeout_s: float):
        """One isolated engine invocation. Returns (status, sentinels)."""
        d = workdir / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "structure.data").write_text(data_text)
        if model.file_path is not None:      # potential file: run-local copy
            shutil.copyfile(model.file_path, d / model.file_name)
        init_block, interaction_block = pair_blocks(model)
        script = render(template, {**values, "data_file": "structure.data"},
                        {"init_block": init_block,
                         "interaction_block": interaction_block})
        (d / "input.in").write_text(script)

        cmd = [self.binary, "-in", "input.in", "-log", "log.lammps"]
        try:
            with open(d / "stdout.txt", "w") as so, open(d / "stderr.txt", "w") as se:
                proc = subprocess.run(cmd, cwd=d, stdout=so, stderr=se,
                                      timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return "timeout", {}
        except FileNotFoundError:
            (d / "stderr.txt").write_text(f"binary not found: {self.binary}\n")
            return "failed", {}
        if proc.returncode != 0:
            return "failed", {}

        log = d / "log.lammps"
        text = log.read_text() if log.exists() else ""
        if not text and (d / "stdout.txt").exists():
            text = (d / "stdout.txt").read_text()
        sentinels = parse_sentinels(text)
        if not _REQUIRED_SENTINELS[template] <= set(sentinels):
            return "failed", sentinels          # missing outputs: fail closed
        return "completed", sentinels

    # ---------------------------------------------------------- protocols

    def execute(self, plan, workdir) -> RawRun:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        t0 = time.time()
        try:
            status, raw = self._execute_inner(plan, workdir)
        except Exception as e:                   # fail closed, keep evidence
            (workdir / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
            status, raw = "failed", {"error": str(e)}
        (workdir / "raw_outputs.json").write_text(json.dumps(raw, indent=2))

        artifacts = []
        for p in sorted(workdir.rglob("*")):
            if p.is_file():
                artifacts.append(ArtifactRef(
                    path=str(p.relative_to(workdir)), sha256=_sha256(p),
                    kind=_KIND_BY_SUFFIX.get(p.suffix, "other")))

        return RawRun(
            run_id=f"{plan.plan_id}_{plan.protocol}",
            plan_id=plan.plan_id,
            plan_sha256=hashlib.sha256(
                plan.model_dump_json().encode()).hexdigest(),
            capability_id=plan.capability_id,
            engine=self.engine, seed=plan.seed,
            started_utc=started, finished_utc=datetime.now(timezone.utc),
            exit_status=status, artifacts=artifacts,
            environment=EnvironmentLock(
                python=platform.python_version(), platform=platform.platform(),
                engine_version=self.version_info().get("version_line") or None),
            resource_usage={"walltime_s": time.time() - t0})

    def _execute_inner(self, plan, workdir: Path):
        model = get_model(plan.model_key, self.allow_unverified)
        budget = plan.resource_budget.max_walltime_s
        t0 = time.time()

        def left():
            return budget - (time.time() - t0)

        min_vals = {"etol": plan.etol, "ftol": plan.ftol,
                    "maxiter": plan.maxiter, "maxeval": plan.maxeval}
        atoms, lin0 = st.diamond_si(plan.a0_A, plan.supercell)

        # 1. relax (all protocols start here)
        status, relax = self._run_one(
            workdir, "relax", "relax_v1",
            {**min_vals, "out_data": "relaxed.data"},
            model, st.to_lammps_data(atoms), max(left(), 1.0))
        raw = {"adapter_version": ADAPTER_VERSION, "model_key": plan.model_key,
               "structure_lineage": [lin0], "relax": relax}
        if status != "completed":
            return status, raw
        if plan.protocol == "relax_v1":
            return "completed", raw

        relaxed = ase_read(workdir / "relax" / "relaxed.data",
                           format="lammps-data")
        relaxed_lin = {"op": "engine_relax", "parent_sha256": lin0["sha256"],
                       "sha256": st.atoms_sha256(relaxed)}
        raw["structure_lineage"].append(relaxed_lin)

        # 2a. EOS volume sweep
        if plan.protocol == "eos_sweep_v1":
            scales = np.linspace(plan.scale_min, plan.scale_max, plan.n_volumes)
            points = []
            for i, s in enumerate(scales):
                if left() <= 0:
                    return "timeout", {**raw, "eos_points": points}
                cell, lin = st.scale_volume(relaxed, float(s), relaxed_lin)
                stat, sen = self._run_one(workdir, f"eos_{i:02d}", "static_v1",
                                          {}, model, st.to_lammps_data(cell),
                                          max(left(), 1.0))
                if stat != "completed":
                    return stat, {**raw, "eos_points": points, "failed_at": i}
                points.append({"volume_scale": float(s), "lineage": lin,
                               "sentinels": sen})
            raw["eos_points"] = points
            return "completed", raw

        # 2b. finite-strain stress sweep
        if plan.protocol == "finite_strain_v1":
            amps = np.linspace(plan.max_strain / plan.n_strains_per_side,
                               plan.max_strain, plan.n_strains_per_side)
            points = []
            for v in range(1, 7):
                for a in np.concatenate([-amps[::-1], amps]):
                    if left() <= 0:
                        return "timeout", {**raw, "strain_points": points}
                    F = st.strain_matrix(v, float(a))
                    cell, lin = st.apply_deformation(relaxed, F, relaxed_lin)
                    stat, sen = self._run_one(
                        workdir, f"strain_v{v}_{a:+.5f}", "static_v1", {},
                        model, st.to_lammps_data(cell), max(left(), 1.0))
                    if stat != "completed":
                        return stat, {**raw, "strain_points": points,
                                      "failed_at": [v, float(a)]}
                    points.append({"voigt": v, "amplitude": float(a),
                                   "F": np.asarray(F).tolist(),
                                   "lineage": lin, "sentinels": sen})
            raw["strain_points"] = points
            raw["conventions"] = {
                "strain": "engineering; shear amplitude = gamma, eps_ij = gamma/2",
                "stress_units": "bar (LAMMPS metal units); sign: pressure tensor"}
            return "completed", raw

        raise ValueError(f"unknown protocol {plan.protocol!r}")
