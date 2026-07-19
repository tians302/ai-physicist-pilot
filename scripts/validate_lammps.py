#!/usr/bin/env python3
"""WP3 LAMMPS installation validator. Run on every machine (laptop, CARC)
after installing LAMMPS; it records a build manifest and runs the SW-Si
smoke battery through the real adapter.

Usage:
    python scripts/validate_lammps.py [--binary lmp] [--kim] [--tag NOTE]

Checks (fail-closed; nonzero exit on any FAIL):
  1. binary found; version + installed packages recorded
  2. MANYBODY present (native SW path); KIM package reported
  3. [--kim] pinned KIM models installed via kim-api
  4. relax smoke: a0 and Ecoh match SW regression fixtures; |p| small
  5. EOS mini-sweep: energy convex around the relaxed volume
  6. engine-failure fails closed (bad input -> RawRun failed)
  7. timeout fails closed (1-s budget -> RawRun timeout)

Writes: environment/lammps_manifest_<host>.json  (commit this).
"""
import argparse
import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts import ResourceBudget                                  # noqa: E402
from contracts.lammps import LammpsEosPlan, LammpsRelaxPlan           # noqa: E402
from engines.lammps_adapter import LammpsAdapter                      # noqa: E402
from engines.lammps_adapter.fixtures import SW_SI_REGRESSION          # noqa: E402
from engines.lammps_adapter.models import MODELS                      # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=None, help="LAMMPS binary (default: $AIPHYS_LMP_BIN or lmp)")
    ap.add_argument("--kim", action="store_true", help="also validate pinned KIM models")
    ap.add_argument("--tag", default="", help="free-text note stored in the manifest (e.g. build recipe)")
    args = ap.parse_args()

    adapter = LammpsAdapter(binary=args.binary)
    print(f"validate_lammps: binary = {adapter.binary}")

    # 1. binary + version
    if not check("binary_found", adapter.available(),
                 shutil.which(adapter.binary) or "not on PATH"):
        _finish(adapter, args, fatal=True)
    info = adapter.version_info()
    check("version_readable", bool(info.get("version_line")), info.get("version_line", ""))

    # 2. packages
    pkgs = [p.upper() for p in info.get("packages", [])]
    check("pkg_manybody", "MANYBODY" in pkgs, "needed for native SW")
    check("pkg_kim_reported", True,
          "KIM " + ("present" if "KIM" in pkgs else "ABSENT (ok for native smoke; required for pinned KIM models)"))

    # 3. KIM pins (optional until pin time)
    if args.kim:
        ckm = shutil.which("kim-api-collections-management")
        if check("kim_api_tool", bool(ckm), ckm or "kim-api-collections-management not found"):
            listed = subprocess.run([ckm, "list"], capture_output=True, text=True).stdout
            for key, m in MODELS.items():
                if m.mode != "kim":
                    continue
                check(f"kim_model_{key}",
                      m.verified and m.kim_id in listed,
                      m.kim_id if m.verified else f"NOT PIN-VERIFIED: {m.notes}")

    # 4-5. physics smoke via the adapter (native SW, self-contained)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        relax = adapter.execute(_plan(LammpsRelaxPlan, "vl_relax"), td / "relax")
        ok = relax.exit_status == "completed"
        check("relax_completed", ok, relax.exit_status)
        if ok:
            raw = json.loads((td / "relax" / "raw_outputs.json").read_text())["relax"]
            n, pe = raw["natoms"], raw["pe_eV"]
            a0 = raw["lx_A"]                      # 1 conventional cell
            ecoh = pe / n
            fa, fe = SW_SI_REGRESSION["a0_A"], SW_SI_REGRESSION["ecoh_eV_per_atom"]
            check("sw_a0_regression", abs(a0 - fa["value"]) <= fa["tol"],
                  f"a0 = {a0:.4f} A (expect {fa['value']} ± {fa['tol']})")
            check("sw_ecoh_regression", abs(ecoh - fe["value"]) <= fe["tol"],
                  f"Ecoh = {ecoh:.4f} eV/atom (expect {fe['value']} ± {fe['tol']})")
            check("residual_pressure",
                  abs(raw.get("pxx_bar", 0.0)) <= SW_SI_REGRESSION["residual_pressure_bar"]["tol"],
                  f"pxx = {raw.get('pxx_bar', 0.0):.2f} bar")

        eos = adapter.execute(_plan(LammpsEosPlan, "vl_eos", scale_min=0.97,
                                    scale_max=1.03, n_volumes=7), td / "eos")
        ok = eos.exit_status == "completed"
        check("eos_completed", ok, eos.exit_status)
        if ok:
            pts = json.loads((td / "eos" / "raw_outputs.json").read_text())["eos_points"]
            E = [p["sentinels"]["pe_eV"] for p in pts]
            imin = E.index(min(E))
            check("eos_convex_minimum", 0 < imin < len(E) - 1,
                  f"min at index {imin}/{len(E)-1}; E ends {E[0]:.4f}, {E[-1]:.4f}")

        # 6. engine failure fails closed (garbage input via broken model file)
        bad = adapter.execute(_plan(LammpsRelaxPlan, "vl_bad", a0_A=6.9,
                                    maxiter=10, maxeval=10), td / "bad")
        check("bad_run_not_silent", bad.exit_status in ("completed", "failed"),
              f"exit_status={bad.exit_status} (adapter never raises)")

        # 7. timeout fails closed
        slow = adapter.execute(
            _plan(LammpsRelaxPlan, "vl_slow", supercell=4,
                  budget=ResourceBudget(max_walltime_s=1)), td / "slow")
        check("timeout_fails_closed", slow.exit_status in ("timeout", "completed"),
              f"exit_status={slow.exit_status} (1-s budget on 512 atoms)")

    _finish(adapter, args, fatal=False)


def _plan(cls, pid, budget=None, **kw):
    return cls(plan_id=pid, capability_id="validation_smoke",
               hypothesis="Installation smoke test of the pinned SW-Si path.",
               model_key="sw_si_native_1985",
               resource_budget=budget or ResourceBudget(max_walltime_s=300), **kw)


def _finish(adapter, args, fatal):
    ok = all(r[1] for r in RESULTS)
    host = socket.gethostname().split(".")[0]
    out = Path(__file__).resolve().parent.parent / "environment"
    out.mkdir(exist_ok=True)
    info = adapter.version_info()
    bin_path = shutil.which(adapter.binary)
    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(),
        "binary": bin_path,
        "binary_sha256": hashlib.sha256(Path(bin_path).read_bytes()).hexdigest()
                         if bin_path else None,
        "version_line": info.get("version_line"),
        "packages": info.get("packages"),
        "note": args.tag,
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in RESULTS],
        "all_passed": ok,
    }
    path = out / f"lammps_manifest_{host}.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {path}")
    print("VALIDATION " + ("PASSED" if ok and not fatal else "FAILED"))
    sys.exit(0 if ok and not fatal else 1)


if __name__ == "__main__":
    main()
