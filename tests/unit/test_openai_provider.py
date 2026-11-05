from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider


@pytest.mark.asyncio
async def test_http_error_still_writes_ledger(tmp_path) -> None:
    ledger = UsageLedger(path=tmp_path / "ledger.jsonl", task_id="http-err")
    provider = OpenAIChatProvider(api_key="test-key", ledger=ledger, test_run_id="err")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.headers = {}
    mock_resp.json.return_value = {
        "usage": {"prompt_tokens": 3, "completion_tokens": 0},
        "error": "unauthorized",
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    provider._client = mock_client

    turn = await provider.complete([{"role": "user", "content": "hi"}], max_tokens=1)
    assert turn.http_status == 401
    assert ledger.read_cumulative() == 3
    await provider.aclose()
