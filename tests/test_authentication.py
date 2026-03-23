from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ServerDisconnectedError

from deebot_client.authentication import Authenticator, _AuthClient, create_rest_config
from deebot_client.exceptions import ApiError
from deebot_client.models import Credentials

if TYPE_CHECKING:
    from aiohttp import ClientSession

    from deebot_client.authentication import RestConfiguration


async def test_authenticator_authenticate(rest_config: RestConfiguration) -> None:
    on_changed_called = asyncio.Event()

    async def on_changed(_: Credentials) -> None:
        if on_changed_called.is_set():
            pytest.fail("Event was already set")
        on_changed_called.set()

    with patch("deebot_client.authentication._AuthClient", spec_set=True) as api_client:
        login_mock: AsyncMock = api_client.return_value.login
        login_mock.return_value = Credentials(
            "token", "user_id", int(time.time() + 123456789)
        )
        authenticator = Authenticator(rest_config, "test", "test")

        unsub = authenticator.subscribe(on_changed)

        assert (await authenticator.authenticate()) == login_mock.return_value
        login_mock.assert_awaited_once()
        async with asyncio.timeout(0.1):
            await on_changed_called.wait()
            on_changed_called.clear()

        login_mock.reset_mock()

        # re-authenticate but this time we can use the cached credentials
        assert (await authenticator.authenticate()) == login_mock.return_value
        login_mock.assert_not_called()
        assert not on_changed_called.is_set()

        # Unsubscribe from authenticator
        unsub()

        # re-authenticate with force=True should call again the api
        assert (await authenticator.authenticate(force=True)) == login_mock.return_value
        login_mock.assert_awaited_once()
        assert not on_changed_called.is_set()


@pytest.mark.parametrize(
    (
        "country",
        "override_rest_url",
        "expected_portal_url",
        "expected_login_url",
        "expected_auth_code_url",
    ),
    [
        (
            "CN",
            "http://example.com",
            "http://example.com",
            "http://example.com",
            "http://example.com",
        ),
        (
            "CN",
            None,
            "https://portal.ecouser.net",
            "https://gl-cn-api.ecovacs.cn",
            "https://gl-cn-openapi.ecovacs.cn",
        ),
        (
            "IT",
            "http://example.com",
            "http://example.com",
            "http://example.com",
            "http://example.com",
        ),
        (
            "IT",
            None,
            "https://portal-eu.ecouser.net",
            "https://gl-it-api.ecovacs.com",
            "https://gl-it-openapi.ecovacs.com",
        ),
    ],
)
def test_config_override_rest_url(
    session: ClientSession,
    country: str,
    override_rest_url: str | None,
    expected_portal_url: str,
    expected_login_url: str,
    expected_auth_code_url: str,
) -> None:
    """Test rest configuration."""
    config = create_rest_config(
        session=session,
        device_id="123",
        alpha_2_country=country,
        override_rest_url=override_rest_url,
    )
    assert config.portal_url == expected_portal_url
    assert config.login_url == expected_login_url
    assert config.auth_code_url == expected_auth_code_url


@pytest.mark.parametrize(
    ("num_disconnects", "should_succeed"),
    [
        (1, True),  # 1 disconnect then success
        (2, True),  # 2 disconnects then success
        (3, False),  # all retries exhausted → ApiError
    ],
    ids=["one_disconnect", "two_disconnects", "all_retries_exhausted"],
)
async def test_post_retries_on_server_disconnected(
    rest_config: RestConfiguration,
    num_disconnects: int,
    should_succeed: bool,
) -> None:
    """ServerDisconnectedError should be retried; ApiError raised when all retries fail."""
    client = _AuthClient(rest_config, "test_account", "test_password")

    call_count = 0

    def make_post_cm(*_args: object, **_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        cm = MagicMock()
        if call_count <= num_disconnects:
            cm.__aenter__ = AsyncMock(side_effect=ServerDisconnectedError())
        else:
            response = MagicMock()
            response.status = 200
            response.raise_for_status = MagicMock()
            response.json = AsyncMock(return_value={"result": "ok"})
            cm.__aenter__ = AsyncMock(return_value=response)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    with patch.object(rest_config.session, "post", side_effect=make_post_cm):
        if should_succeed:
            result = await client.post("some/path", {})
            assert result == {"result": "ok"}
        else:
            with pytest.raises(ApiError):
                await client.post("some/path", {})


async def test_post_raises_api_error_on_session_closed(
    rest_config: RestConfiguration,
) -> None:
    """RuntimeError('Session is closed') should be converted to ApiError immediately (no retry)."""
    client = _AuthClient(rest_config, "test_account", "test_password")

    call_count = 0

    def make_post_cm(*_args: object, **_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=RuntimeError("Session is closed"))
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    with patch.object(rest_config.session, "post", side_effect=make_post_cm):
        with pytest.raises(ApiError, match="Session is closed"):
            await client.post("some/path", {})

    # Session is closed is not retried; only one attempt should be made
    assert call_count == 1


async def test_post_reraises_other_runtime_errors(
    rest_config: RestConfiguration,
) -> None:
    """RuntimeErrors unrelated to a closed session should propagate unchanged."""
    client = _AuthClient(rest_config, "test_account", "test_password")

    def make_post_cm(*_args: object, **_kwargs: object) -> MagicMock:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=RuntimeError("Some other error"))
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    with patch.object(rest_config.session, "post", side_effect=make_post_cm):
        with pytest.raises(RuntimeError, match="Some other error"):
            await client.post("some/path", {})
