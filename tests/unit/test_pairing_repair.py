from __future__ import annotations

from probe_harness.providers.guards import repair_tool_pairing


def pairing_ok(msgs: list[dict]) -> str | list[str]:
    i = 0
    problems: list[str] = []
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "tool":
            problems.append(f"stray tool result {m.get('tool_call_id')} at {i}")
            i += 1
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            want = [tc["id"] for tc in m["tool_calls"]]
            j = i + 1
            got: list[str] = []
            while j < len(msgs) and msgs[j].get("role") == "tool":
                got.append(msgs[j].get("tool_call_id", ""))
                j += 1
            if got != want:
                problems.append(f"assistant@{i} wants {want} got {got}")
            i = j
            continue
        i += 1
    return problems or "OK"


def A(*ids: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": x,
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
            for x in ids
        ],
    }


def R(call_id: str, content: str = "r") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def U(content: str = "u") -> dict:
    return {"role": "user", "content": content}


def test_c1_repair_mismatched_id() -> None:
    msgs = [U(), A("c1"), R("WRONG"), U("next")]
    out, repairs = repair_tool_pairing(msgs, reason="probe_b")
    assert any(
        r["action"] in ("drop_mismatched_tool_result", "drop_orphan_tool_result") for r in repairs
    )
    assert any(r["action"] == "synthetic_tool_result" for r in repairs)
    assert pairing_ok(out) == "OK"


def test_c1_repair_non_adjacent_result() -> None:
    msgs = [U(), A("c1"), U("interject"), R("c1")]
    out, repairs = repair_tool_pairing(msgs, reason="probe_c")
    assert any(r["action"] == "relocate_non_adjacent_tool_result" for r in repairs)
    assert pairing_ok(out) == "OK"


def test_c1_repair_duplicate_result() -> None:
    msgs = [U(), A("c1"), R("c1", "x"), R("c1", "y"), U("next")]
    out, repairs = repair_tool_pairing(msgs, reason="probe_d")
    assert any(r["action"] == "drop_duplicate_tool_result" for r in repairs)
    assert pairing_ok(out) == "OK"


def test_c1_repair_stray_result_no_call() -> None:
    msgs = [U(), R("ghost"), U("next")]
    out, repairs = repair_tool_pairing(msgs, reason="probe_e")
    assert any(r["action"] == "drop_orphan_tool_result" for r in repairs)
    assert pairing_ok(out) == "OK"
