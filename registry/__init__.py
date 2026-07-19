"""WP2 capability registry (trusted control plane).

The registry is the closed catalog of executable capabilities. The router
(registry.router) may only match hypotheses against this catalog; if no
compatible capability exists the outcome is `unsupported_hypothesis` —
the engine or model is never silently changed (PLAN.md, target
architecture). LLM routing suggestions are advisory; the registry and
validator are authoritative.
"""
from contracts import CapabilityManifest


class DuplicateCapabilityError(ValueError):
    pass


class CapabilityRegistry:
    def __init__(self):
        self._by_id: dict[str, CapabilityManifest] = {}

    def register(self, manifest: CapabilityManifest) -> None:
        cid = manifest.capability_id
        if cid in self._by_id:
            raise DuplicateCapabilityError(f"capability {cid!r} already registered")
        self._by_id[cid] = manifest

    def get(self, capability_id: str) -> CapabilityManifest:
        return self._by_id[capability_id]

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def all(self) -> list[CapabilityManifest]:
        """Deterministic order: sorted by capability_id."""
        return [self._by_id[c] for c in self.ids()]

    def supports_property(self, name: str) -> list[CapabilityManifest]:
        return [m for m in self.all()
                if name == m.property_name or name in m.property_aliases]

    def __len__(self):
        return len(self._by_id)

    def __contains__(self, capability_id: str):
        return capability_id in self._by_id


def default_registry() -> CapabilityRegistry:
    from registry.builtin import callaway_capability
    reg = CapabilityRegistry()
    reg.register(callaway_capability())
    return reg
