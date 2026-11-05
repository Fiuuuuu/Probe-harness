"""Connectivity check for api.deepseek.com."""

from __future__ import annotations

import asyncio
import os
import socket
import ssl

import httpx

from probe_harness.providers.ledger import UsageLedger
from probe_harness.providers.openai_chat import OpenAIChatProvider
from probe_harness.providers.usage import estimate_chat_input_tokens, normalise_openai_usage

HOST = "api.deepseek.com"
BASE_URL = f"https://{HOST}"

# ok: True=pass, False=fail, None=skipped (N1)
NetRow = tuple[str, bool | None, str]


async def check_dns() -> tuple[bool, str]:
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.run_in_executor(None, lambda: socket.getaddrinfo(HOST, 443))
        addrs = sorted({i[4][0] for i in infos})
        return True, ", ".join(addrs[:3])
    except OSError as exc:
        return False, str(exc)


async def check_tcp() -> tuple[bool, str]:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(HOST, 443), timeout=10.0)
        writer.close()
        await writer.wait_closed()
        return True, "tcp_ok"
    except OSError as exc:
        return False, str(exc)


async def check_tls() -> tuple[bool, str]:
    try:
        ctx = ssl.create_default_context()

        def _handshake() -> str:
            with socket.create_connection((HOST, 443), timeout=10) as sock:
                with ctx.wrap_socket(sock, server_hostname=HOST) as ssock:
                    return ssock.version() or "tls_ok"

        version = await asyncio.get_running_loop().run_in_executor(None, _handshake)
        return True, version
    except OSError as exc:
        return False, str(exc)


async def check_unauthenticated_http() -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
            resp = await client.post(
                f"{BASE_URL}/chat/completions",
                json={
                    "model": "deepseek-flash",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "thinking": {"type": "disabled"},
                },
            )
        if resp.status_code in (401, 403):
            return True, f"http_{resp.status_code}_auth_error_expected"
        if resp.status_code >= 500:
            return False, f"http_{resp.status_code}"
        return True, f"http_{resp.status_code}"
    except httpx.HTTPError as exc:
        return False, str(exc)


async def check_authenticated_live(ledger: UsageLedger) -> tuple[bool, str]:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return False, "skipped_no_key"
    if os.environ.get("RUN_LIVE") != "1":
        return False, "skipped_run_live_not_set"

    provider = OpenAIChatProvider(
        api_key=key,
        ledger=ledger,
        thinking_enabled=False,
        max_tokens=1,
        test_run_id="net-check",
    )
    turn = None
    try:
        messages = [{"role": "user", "content": "Reply with the single letter x."}]
        turn = await provider.complete(
            messages,
            max_tokens=1,
            thinking_enabled=False,
            estimated_input_tokens=estimate_chat_input_tokens(messages),
        )
    except Exception as exc:
        return False, str(exc)
    finally:
        await provider.aclose()

    if turn is None or turn.http_status >= 400:
        return False, f"http_{turn.http_status}"
    norm = normalise_openai_usage(turn.usage)
    return True, f"live_ok usage_in={norm['input_total']} out={norm['output']}"


async def run_net_check() -> tuple[list[NetRow], bool, int]:
    rows: list[NetRow] = []
    ledger = UsageLedger(task_id="net-check")

    for name, coro in (
        ("dns", check_dns()),
        ("tcp_443", check_tcp()),
        ("tls", check_tls()),
        ("https_unauthenticated", check_unauthenticated_http()),
    ):
        ok, detail = await coro
        rows.append((name, ok, detail))

    core_ok = all(r[1] is True for r in rows)

    if os.environ.get("RUN_LIVE") == "1" and os.environ.get("DEEPSEEK_API_KEY"):
        live_ok, live_detail = await check_authenticated_live(ledger)
        rows.append(("live_authenticated", live_ok, live_detail))
    else:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            detail = "skipped_no_key"
        elif os.environ.get("RUN_LIVE") != "1":
            detail = "skipped_run_live_not_set"
        else:
            detail = "skipped"
        rows.append(("live_authenticated", None, detail))

    return rows, core_ok, ledger.total_tokens_used()


def print_net_check_table(rows: list[NetRow], reachable: bool, cumulative: int) -> None:
    print("| check | ok | detail |")
    print("|---|---|---|")
    for name, ok, detail in rows:
        if ok is None:
            ok_str = "skip"
        else:
            ok_str = "yes" if ok else "no"
        print(f"| {name} | {ok_str} | {detail} |")
    print(f"| reachable | {'yes' if reachable else 'no'} | |")
    print(f"| ledger_cumulative_tokens | | {cumulative} |")
