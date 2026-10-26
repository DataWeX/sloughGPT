"""Contract gate for ``datasets.search_federated`` — the federated discovery descriptor.

Companion to the behavioral tests in ``tests/server/test_datasets_search_router.py``
(route envelope, partial-failure semantics, validation). This file pins the
descriptor itself: the contract gate (``scripts/check_contract.py``) requires
every declared capability to have a test under ``apps/api/server/tests/`` —
a contract without a test is a wish.
"""

from infrastructure.contract import REGISTRY, derive_method
from routers.datasets_search import SPEC


def test_descriptor_name_and_tier():
    assert SPEC.name == "datasets.search_federated"
    # site route of the app — never a model-facing chat tool
    assert SPEC.auth_scope == "public"
    assert SPEC.idempotent is True


def test_descriptor_projects_to_get():
    assert derive_method(SPEC) == "GET"


def test_query_is_required_and_others_optional():
    by_name = {p.name: p for p in SPEC.parameters}
    assert by_name["q"].required is True
    assert by_name["limit"].required is False
    assert by_name["sources"].required is False


def test_route_is_registered_in_contract_registry():
    assert ("GET", "/datasets/search/all") in REGISTRY
