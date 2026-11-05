"""Follow-up probes for g (thinking). Continues budget from results.json."""
import json, re, probes as P
prev = json.load(open(P.OUT)); P.STATE.update(prev["state"]); P.RESULTS[:] = prev["results"]
Q = "First compute 37*43 and 29*53 and decide which is larger (think it through briefly), then call get_weather for Paris."
# MiniMax Anthropic, adaptive
s, j = P.call(P.MA, "g4_thinking_reasoning_prompt", {"thinking": {"type": "adaptive"}, "tools": P.WEATHER_A, "messages": [{"role": "user", "content": Q}]})
if s == 200 and any(b["type"] == "thinking" for b in j["content"]) and any(b["type"] == "tool_use" for b in j["content"]):
    tu = [b for b in j["content"] if b["type"] == "tool_use"][0]
    tr = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu["id"], "content": "18C sunny"}]}
    stripped = [b for b in j["content"] if b["type"] != "thinking"]
    P.call(P.MA, "g5_drop_thinking", {"thinking": {"type": "adaptive"}, "tools": P.WEATHER_A, "messages": [{"role": "user", "content": Q}, {"role": "assistant", "content": stripped}, tr]}, note="thinking removed")
# MiniMax OpenAI, thinking omitted (default)
s, j = P.call(P.MO, "g4_default_thinking_reasoning_prompt", {"tools": P.WEATHER_O, "messages": [{"role": "user", "content": Q}]}, note="thinking omitted")
if s == 200:
    m = j["choices"][0]["message"]
    has_r = bool(m.get("reasoning_content") or m.get("reasoning_details") or "<think>" in (m.get("content") or ""))
    if has_r and m.get("tool_calls"):
        tr = {"role": "tool", "tool_call_id": m["tool_calls"][0]["id"], "content": "18C sunny"}
        st = {"role": "assistant", "content": re.sub(r"<think>.*?</think>", "", m.get("content") or "", flags=re.S).strip(), "tool_calls": m["tool_calls"]}
        P.call(P.MO, "g5_drop_reasoning", {"tools": P.WEATHER_O, "messages": [{"role": "user", "content": Q}, st, tr]}, note="reasoning removed")
# DeepSeek: later user turn, earlier tool turn without reasoning_content
tc = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_00_x", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}
P.call(P.DS, "g6_next_user_turn_no_reasoning", {"tools": P.WEATHER_O, "messages": [
    {"role": "user", "content": "Weather in Paris?"}, tc, {"role": "tool", "tool_call_id": "call_00_x", "content": "18C sunny"},
    {"role": "assistant", "content": "Paris: 18C sunny."}, {"role": "user", "content": "Say OK."}]}, note="thinking default, no reasoning_content anywhere in history")
P.call(P.DS, "g7_explicit_enabled_no_reasoning", {"thinking": {"type": "enabled"}, "tools": P.WEATHER_O, "messages": [
    {"role": "user", "content": "Weather in Paris?"}, tc, {"role": "tool", "tool_call_id": "call_00_x", "content": "18C sunny"}]}, note="thinking explicitly enabled, fabricated tool turn w/o reasoning_content")
P.save(); print("FINAL", P.STATE)
