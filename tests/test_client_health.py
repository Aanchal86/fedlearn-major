
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from federated.networking.registry import ClientRegistry


@pytest.fixture
def registry():
    return ClientRegistry()


def test_new_client_is_active(registry):
    registry.register("client_1")

    assert registry.is_active("client_1")


def test_unknown_client_is_inactive(registry):
    assert not registry.is_active("unknown")


def test_heartbeat_refreshes_last_seen(registry):
    original = registry.register("client_1")

    updated = registry.heartbeat("client_1")

    assert updated.last_seen >= original.last_seen
    assert updated.registered_at == original.registered_at


def test_expired_client_is_inactive(registry):
    registry.register("client_1")

    expired_time = (
        datetime.now(timezone.utc)
        - timedelta(seconds=120)
    )

    with registry._lock:
        registry._clients["client_1"] = replace(
            registry._clients["client_1"],
            last_seen=expired_time,
        )

    assert not registry.is_active(
        "client_1",
        timeout_seconds=60,
    )


def test_heartbeat_reactivates_client(registry):
    registry.register("client_1")

    expired_time = (
        datetime.now(timezone.utc)
        - timedelta(seconds=120)
    )

    with registry._lock:
        registry._clients["client_1"] = replace(
            registry._clients["client_1"],
            last_seen=expired_time,
        )

    assert not registry.is_active("client_1")

    registry.heartbeat("client_1")

    assert registry.is_active("client_1")


def test_active_and_inactive_lists(registry):
    registry.register("client_1")
    registry.register("client_2")

    expired_time = (
        datetime.now(timezone.utc)
        - timedelta(seconds=120)
    )

    with registry._lock:
        registry._clients["client_2"] = replace(
            registry._clients["client_2"],
            last_seen=expired_time,
        )

    active = registry.list_active_clients()
    inactive = registry.list_inactive_clients()

    assert [client.client_id for client in active] == [
        "client_1"
    ]

    assert [client.client_id for client in inactive] == [
        "client_2"
    ]


def test_health_summary(registry):
    registry.register("client_1")
    registry.register("client_2")

    summary = registry.health_summary()

    assert summary["total_clients"] == 2
    assert summary["active_clients"] == 2
    assert summary["inactive_clients"] == 0


def test_invalid_timeout_rejected(registry):
    with pytest.raises(ValueError):
        registry.is_active(
            "client_1",
            timeout_seconds=0,
        )

    with pytest.raises(ValueError):
        registry.health_summary(
            timeout_seconds=-1,
        )
