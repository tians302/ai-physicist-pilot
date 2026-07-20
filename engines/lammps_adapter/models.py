"""Pinned physical models for the LAMMPS adapter (WP3).

Pin discipline (PHASE1B2 checklist): KIM model IDs/versions must be
verified on openkim.org AT PIN TIME and checksummed; until a human flips
`verified=True` (with date + checksum), the adapter refuses the model in
production mode. The native SW file ships in-repo from the published 1985
parameter set and is checksummed here, so laptop/sandbox smoke tests are
self-contained even without kim-api.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_POTENTIALS_DIR = Path(__file__).resolve().parent / "potentials"


@dataclass(frozen=True)
class PinnedModel:
    key: str
    mode: str                        # "kim" | "native_sw"
    elements: tuple
    kim_id: Optional[str] = None
    file_name: Optional[str] = None  # relative to potentials/
    file_sha256: Optional[str] = None
    verified: bool = False           # flip ONLY at human pin time
    pinned_date: Optional[str] = None
    notes: str = ""

    @property
    def file_path(self) -> Optional[Path]:
        return _POTENTIALS_DIR / self.file_name if self.file_name else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


MODELS: dict[str, PinnedModel] = {m.key: m for m in [
    PinnedModel(
        key="sw_si_native_1985",
        mode="native_sw",
        elements=("Si",),
        file_name="Si.sw",
        # checksum of the in-repo file; verified because the file itself is
        # the pin (published SW 1985 parameters, no external fetch).
        file_sha256="3876f9574a7f8f8b5c2a6519f250fe888854c8b436da53a2a293194aedbb156c",
        verified=True,
        pinned_date="2026-07-19",
        notes="Stillinger-Weber 1985 Si, PRB 31 5262. Regression fixtures only."),
    PinnedModel(
        key="sw_si_kim",
        mode="kim",
        elements=("Si",),
        kim_id="SW_StillingerWeber_1985_Si__MO_405512056662_006",
        verified=False,
        notes="VERIFY exact current ID/version on openkim.org at pin time "
              "(WP3 checklist), record checksum + date, then set verified."),
    PinnedModel(
        key="lj_argon",
        mode="native_lj",
        elements=("Ar",),
        verified=True,
        pinned_date="2026-07-19",
        notes="Parameter-free reduced-unit Lennard-Jones (eps=sigma=1, "
              "rc=2.5). WP8 transport ladder; literature comparisons via "
              "reduced state points."),
    PinnedModel(
        key="tersoff_si_native_1988",
        mode="native_tersoff",
        elements=("Si",),
        file_name="Si.tersoff",
        file_sha256="5b55dc7be1569193f468b9f56128a36e67feedd8e145561f61aec4b49a33d723",
        verified=True,
        pinned_date="2026-07-19",
        notes="Tersoff Si, PRB 37, 6991 (1988). Pinned from the official "
              "LAMMPS 2025-07-22 release potentials/ (wheel 2025.7.22.4.0). "
              "Second, independently parameterized Si model for WP12 "
              "cross-model sensitivity. Regression fixtures only."),
    PinnedModel(
        key="tersoff_si_kim",
        mode="kim",
        elements=("Si",),
        kim_id="PENDING_PIN_VERIFY_ON_OPENKIM",
        verified=False,
        notes="Optional KIM twin of tersoff_si_native_1988; verify on "
              "openkim.org at pin time if KIM provenance is wanted."),
]}


class UnpinnedModelError(RuntimeError):
    pass


def get_model(key: str, allow_unverified: bool = False) -> PinnedModel:
    if key not in MODELS:
        raise UnpinnedModelError(f"model {key!r} not in pinned registry {sorted(MODELS)}")
    m = MODELS[key]
    if not m.verified and not allow_unverified:
        raise UnpinnedModelError(
            f"model {key!r} is not pin-verified ({m.notes}); "
            "refusing production use (fail-closed)")
    return m


def pair_blocks(model: PinnedModel) -> tuple[str, str]:
    """Trusted (init_block, interaction_block) template blocks. Only this
    module may construct them — plans never inject engine commands."""
    els = " ".join(model.elements)
    if model.mode == "kim":
        return (f"kim init {model.kim_id} metal",
                f"kim interactions {els}")
    if model.mode == "native_sw":
        # file is copied into the run dir by the adapter (checksummed
        # artifact; also avoids LAMMPS tokenization issues with paths)
        return ("units metal",
                f"pair_style sw\npair_coeff * * {model.file_name} {els}")
    if model.mode == "native_lj":
        return ("units lj",
                "pair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0")
    if model.mode == "native_tersoff":
        return ("units metal",
                f"pair_style tersoff\npair_coeff * * {model.file_name} {els}")
    raise UnpinnedModelError(f"unknown model mode {model.mode!r}")
