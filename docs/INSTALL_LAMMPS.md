# Installing and pinning LAMMPS (WP3)

Goal (from `PHASE1B2_EXECUTION_PLAN.md`): **one pinned stable LAMMPS release used everywhere** (laptop + CARC), with `MANYBODY` and `KIM` packages, kim-api, and two pinned OpenKIM silicon models — all recorded in a committed build manifest. The adapter refuses unpinned models by design, so finishing the pin steps below is part of the install, not optional polish.

The native SW path (`sw_si_native_1985`, parameters shipped in-repo and checksummed) works without kim-api, so you can validate the toolchain first and pin KIM models second.

## 1. Laptop (macOS, conda-forge)

Install miniforge if you don't have a conda:

```bash
brew install miniforge        # or: curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh && bash Miniforge3-*.sh
```

See what LAMMPS versions conda-forge currently offers, then create the env **with an explicit version** (pick the latest stable; write it down — this is the pin):

```bash
conda search -c conda-forge lammps | tail -5
conda create -n ai-physicist -c conda-forge python=3.10 "lammps=<VERSION>"
conda activate ai-physicist
pip install -r requirements.txt
```

Notes:

- The conda-forge build ships the `lmp` binary with most packages enabled (including MANYBODY and KIM) and pulls in `kim-api`. Verify rather than trust: `lmp -h | grep -A40 "Installed packages"`.
- Apple Silicon and Intel both have native builds; no Rosetta needed.
- If you prefer a serial laptop build, that's fine — engine parity across machines is by *release version*, not by MPI flavor; record both in the manifest.

## 2. Validate the toolchain (native SW path)

From the repo root, inside the env:

```bash
python scripts/validate_lammps.py --binary lmp --tag "laptop conda-forge lammps=<VERSION>"
```

This runs the full smoke battery **through the real adapter** (relax → a0/Ecoh regression fixtures, EOS convexity, failure + timeout fail-closed checks) and writes `environment/lammps_manifest_<host>.json`. Expected: `VALIDATION PASSED` and a manifest listing the version line, installed packages, and binary checksum. **Commit the manifest.**

## 3. Pin the two OpenKIM models (pin-time verification)

Do this once, deliberately, with openkim.org open:

1. On openkim.org, find the **canonical Stillinger–Weber (1985) Si model** and copy its full current ID including the version suffix (it looks like `SW_StillingerWeber_1985_Si__MO_405512056662_XXX` — verify the suffix, do not trust any transcription including the placeholder in `models.py`).
2. Choose the **second, independently parameterized Si model** (a Tersoff variant, e.g. Erhart–Albe or Tersoff T3 family) and copy its full ID.
3. Install both into the user collection:

   ```bash
   kim-api-collections-management install user <SW_MODEL_ID>
   kim-api-collections-management install user <TERSOFF_MODEL_ID>
   kim-api-collections-management list
   ```

4. Edit `engines/lammps_adapter/models.py`: set the verified `kim_id`s, `verified=True`, `pinned_date`, and paste the models' applicability facts (elements, regimes, known failure modes) from their OpenKIM pages into the manifest cards / registry at WP4.
5. Re-run validation including KIM checks:

   ```bash
   python scripts/validate_lammps.py --binary lmp --kim --tag "laptop + KIM pins"
   ```

6. Commit `models.py` + the refreshed manifest together (this commit *is* the pin).

## 4. CARC (same release, Slurm)

1. Check the module tree first: `module spider lammps`. If a module matches the laptop-pinned release **and** reports MANYBODY+KIM, use it and record `module list` output in the manifest tag.
2. Otherwise build the same release yourself (recommended for control):

   ```bash
   git clone -b <RELEASE_TAG> --depth 1 https://github.com/lammps/lammps.git
   cd lammps && mkdir build && cd build
   module load gcc openmpi cmake        # record exact versions
   cmake ../cmake -D BUILD_MPI=on -D PKG_MANYBODY=on -D PKG_KIM=on \
         -D DOWNLOAD_KIM=on -D CMAKE_INSTALL_PREFIX=$HOME/opt/lammps-<RELEASE_TAG>
   make -j 8 && make install
   export PATH=$HOME/opt/lammps-<RELEASE_TAG>/bin:$PATH
   ```

3. Repeat KIM model installs (step 3.3) on CARC — kim-api collections are per-machine.
4. Validate on a login node (the smoke battery is seconds of compute) or via `salloc`:

   ```bash
   python scripts/validate_lammps.py --binary lmp --kim --tag "CARC build <RELEASE_TAG>, gcc X, openmpi Y, cmake flags above"
   ```

5. Commit the CARC manifest. Slurm/sbatch templates with walltime caps come with the Phase-2 battery work (WP10), not now.

## 5. Definition of done (WP3 install checklist)

- [ ] One LAMMPS release chosen; **same version green on laptop and CARC** (`validate_lammps.py` PASSED on both; manifests committed).
- [ ] `lmp -h` shows MANYBODY + KIM on both machines.
- [ ] Two OpenKIM Si models pin-verified in `models.py` (IDs with version suffix, `pinned_date`, `verified=True`) and installed on both machines.
- [ ] `pytest tests/ -q` still fully green in the same env.
- [ ] Regression fixtures reproduced: a0 = 5.431 ± 0.01 Å, Ecoh = −4.3364 ± 0.01 eV/atom (same-model fixtures only — never a physics answer key).

Troubleshooting quick hits: `lmp` not found after conda install → `conda deactivate && conda activate ai-physicist`; KIM model "not found" at runtime → collections are per-user/per-machine, re-run step 3.3; box/relax warnings about triclinic cells → harmless for the cubic smoke tests; validation failing only `sw_ecoh_regression` → almost always a wrong/unpinned potential file or model version, check the manifest's checksums.
