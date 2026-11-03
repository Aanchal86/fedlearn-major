
from unittest.mock import MagicMock, patch

import httpx
import pytest

from federated.networking.http_client import FederatedHTTPClient


@pytest.fixture
def client():
    return FederatedHTTPClient(
        client_id="client_1",
        max_retries=2,
        backoff_factor=0,
    )


def make_response(status_code, payload=None):
    request = httpx.Request(
        "GET",
        "http://127.0.0.1:8000/test",
    )

    return httpx.Response(
        status_code,
        request=request,
        json=payload or {"status": "ok"},
    )


def test_successful_request(client):
    with patch("httpx.Client") as mock_class:
        session = mock_class.return_value.__enter__.return_value
        session.request.return_value = make_response(200)

        result = client._request("GET", "/test")

        assert result == {"status": "ok"}
        assert session.request.call_count == 1


def test_retry_after_server_error(client):
    with patch("httpx.Client") as mock_class:
        session = mock_class.return_value.__enter__.return_value

        session.request.side_effect = [
            make_response(503),
            make_response(200),
        ]

        result = client._request("GET", "/test")

        assert result == {"status": "ok"}
        assert session.request.call_count == 2


def test_retry_after_connection_error(client):
    with patch("httpx.Client") as mock_class:
        session = mock_class.return_value.__enter__.return_value

        session.request.side_effect = [
            httpx.ConnectError("Connection failed"),
            make_response(200),
        ]

        result = client._request("GET", "/test")

        assert result == {"status": "ok"}
        assert session.request.call_count == 2


def test_no_retry_on_404(client):
    with patch("httpx.Client") as mock_class:
        session = mock_class.return_value.__enter__.return_value
        session.request.return_value = make_response(404)

        with pytest.raises(httpx.HTTPStatusError):
            client._request("GET", "/test")

        assert session.request.call_count == 1


def test_retry_limit_respected(client):
    with patch("httpx.Client") as mock_class:
        session = mock_class.return_value.__enter__.return_value
        session.request.return_value = make_response(503)

        with pytest.raises(httpx.HTTPStatusError):
            client._request("GET", "/test")

        assert session.request.call_count == 3


def test_submission_does_not_retry(client):
    with patch("httpx.Client") as mock_class:
        session = mock_class.return_value.__enter__.return_value
        session.request.return_value = make_response(503)

        with pytest.raises(httpx.HTTPStatusError):
            client._request(
                "POST",
                "/api/network/clients/client_1/updates",
                retry=False,
            )

        assert session.request.call_count == 1


def test_invalid_retry_settings():
    with pytest.raises(ValueError):
        FederatedHTTPClient(
            "client_1",
            max_retries=-1,
        )

    with pytest.raises(ValueError):
        FederatedHTTPClient(
            "client_1",
            backoff_factor=-1,
        )

    with pytest.raises(ValueError):
        FederatedHTTPClient(
            "client_1",
            timeout=0,
        )


def test_exponential_backoff():
    client = FederatedHTTPClient(
        "client_1",
        max_retries=2,
        backoff_factor=0.5,
    )

    with (
        patch("httpx.Client") as mock_class,
        patch("time.sleep") as mock_sleep,
    ):
        session = mock_class.return_value.__enter__.return_value
        session.request.side_effect = [
            make_response(503),
            make_response(503),
            make_response(200),
        ]

        result = client._request("GET", "/test")

        assert result == {"status": "ok"}
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0].args[0] == 0.5
        assert mock_sleep.call_args_list[1].args[0] == 1.0
