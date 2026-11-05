import json, probes as P
prev = json.load(open(P.OUT)); P.STATE.update(prev["state"]); P.RESULTS[:] = prev["results"]
Q = "First compute 37*43 and 29*53 and decide which is larger (think it through briefly), then call get_weather for Paris."
P.call(P.MA, "g8_thinking_omitted_reasoning_prompt", {"tools": P.WEATHER_A, "messages": [{"role": "user", "content": Q}]}, note="thinking param omitted on Anthropic endpoint")
tc = {"role": "assistant", "content": "", "reasoning_content": "I should call get_weather for Paris.", "tool_calls": [{"id": "call_00_x", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}
P.call(P.DS, "g9_explicit_enabled_with_reasoning", {"thinking": {"type": "enabled"}, "tools": P.WEATHER_O, "messages": [
    {"role": "user", "content": "Weather in Paris?"}, tc, {"role": "tool", "tool_call_id": "call_00_x", "content": "18C sunny"}]}, note="control for g7: reasoning_content passed")
P.save(); print("FINAL", P.STATE)
