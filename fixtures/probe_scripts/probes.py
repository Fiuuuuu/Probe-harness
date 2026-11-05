"""Minimal API constraint probes. Never prints/saves keys. Budget-guarded."""
import os, json, time, re, sys, httpx

MAX_REQ, MAX_TOK = 60, 60000
STATE = {"requests": 2, "tokens": 0}  # 2 = earlier GET /models calls (0 tokens)
RESULTS = []
OUT = os.path.join(os.path.dirname(__file__), "results.json")
KEYS = [os.environ["MINIMAX_API_KEY"], os.environ["DEEPSEEK_API_KEY"]]

MA = dict(name="minimax-anthropic", url="https://api.minimaxi.com/anthropic/v1/messages", model="MiniMax-M3", kind="anth",
          headers={"x-api-key": KEYS[0], "Authorization": "Bearer " + KEYS[0], "anthropic-version": "2023-06-01"})
MO = dict(name="minimax-openai", url="https://api.minimaxi.com/v1/chat/completions", model="MiniMax-M3", kind="oai",
          headers={"Authorization": "Bearer " + KEYS[0]})
DS = dict(name="deepseek", url="https://api.deepseek.com/chat/completions", model="deepseek-flash", kind="oai",
          headers={"Authorization": "Bearer " + KEYS[1]})

def redact(s):
    for k in KEYS:
        s = s.replace(k, "***")
    return s

def usage_tokens(kind, u):
    if not u: return 0
    if kind == "anth":
        return sum(int(u.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    return int(u.get("total_tokens") or 0)

def guard():
    if STATE["requests"] >= MAX_REQ or STATE["tokens"] >= MAX_TOK:
        save(); print("BUDGET EXCEEDED, stop", STATE); sys.exit(2)

def save():
    json.dump({"state": STATE, "results": RESULTS}, open(OUT, "w"), ensure_ascii=False, indent=1)

def call(p, probe, body, note=""):
    guard()
    body = dict(body); body.setdefault("model", p["model"]); body.setdefault("max_tokens", 64)
    assert body["max_tokens"] <= 64
    STATE["requests"] += 1
    t = time.time()
    try:
        r = httpx.post(p["url"], headers=p["headers"], json=body, timeout=120)
        status, text = r.status_code, r.text
    except Exception as e:
        status, text = -1, repr(e)
    j = None
    try: j = json.loads(text)
    except Exception: pass
    u = (j or {}).get("usage") if isinstance(j, dict) else None
    tok = usage_tokens(p["kind"], u); STATE["tokens"] += tok
    rec = {"provider": p["name"], "probe": probe, "note": note, "status": status, "secs": round(time.time()-t, 1),
           "usage": u, "tokens": tok, "cum": dict(STATE)}
    if j and status == 200:
        if p["kind"] == "anth":
            rec["stop_reason"] = j.get("stop_reason")
            rec["content"] = [{k: (str(v)[:200]) for k, v in b.items() if k != "signature"} | ({"signature_len": len(b.get("signature",""))} if "signature" in b else {}) for b in j.get("content", [])]
        else:
            ch = (j.get("choices") or [{}])[0]; m = ch.get("message", {})
            rec["finish_reason"] = ch.get("finish_reason")
            rec["message"] = {k: str(v)[:300] for k, v in m.items()}
    else:
        rec["error_body"] = redact(text[:600])
    RESULTS.append(rec); save()
    print(f"[{p['name']}] {probe} {note} -> {status} tok={tok} cum={STATE}")
    return status, j

def stream(p, probe, body, note=""):
    guard()
    body = dict(body); body.setdefault("model", p["model"]); body.setdefault("max_tokens", 64); body["stream"] = True
    STATE["requests"] += 1
    events, usage_seen, frags, final_usage = [], [], [], None
    with httpx.stream("POST", p["url"], headers=p["headers"], json=body, timeout=120) as r:
        status = r.status_code
        if status != 200:
            txt = r.read().decode()
            RESULTS.append({"provider": p["name"], "probe": probe, "note": note, "status": status, "error_body": redact(txt[:600])}); save()
            print(probe, status); return
        for line in r.iter_lines():
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": events.append("[DONE]"); continue
            try: ev = json.loads(d)
            except Exception: continue
            if p["kind"] == "anth":
                typ = ev.get("type"); events.append(typ)
                if typ == "content_block_delta" and ev["delta"].get("type") == "input_json_delta":
                    frags.append(ev["delta"]["partial_json"])
                if typ == "message_start": usage_seen.append(("message_start", ev["message"].get("usage")))
                if typ == "message_delta": usage_seen.append(("message_delta", ev.get("usage"))); final_usage = {**(usage_seen[0][1] or {}), **(ev.get("usage") or {})}
            else:
                ch = ev.get("choices") or []
                events.append("chunk(choices=%d)" % len(ch))
                for c in ch:
                    for tc in (c.get("delta", {}).get("tool_calls") or []):
                        a = tc.get("function", {}).get("arguments")
                        if a is not None: frags.append(a)
                if ev.get("usage") is not None:
                    usage_seen.append(("chunk", ev["usage"], len(ch))); final_usage = ev["usage"]
    tok = usage_tokens(p["kind"], final_usage); STATE["tokens"] += tok
    # compress events
    comp = []
    for e in events:
        if comp and comp[-1][0] == e: comp[-1][1] += 1
        else: comp.append([e, 1])
    RESULTS.append({"provider": p["name"], "probe": probe, "note": note, "status": status, "events": comp,
                    "arg_fragments": frags, "usage_seen": usage_seen, "tokens": tok, "cum": dict(STATE)}); save()
    print(f"[{p['name']}] {probe} {note} -> {status} frags={len(frags)} usage_events={len(usage_seen)} tok={tok} cum={STATE}")

# ---------- fixtures ----------
WEATHER_A = [{"name": "get_weather", "description": "Get weather for a city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
WEATHER_O = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
U0 = "What's the weather in Paris? Use the tool."
A_TU = {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}]}
def A_TR(i="toolu_01"): return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": "18C sunny"}]}
O_TC = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_01", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}
def O_TR(i="call_01"): return {"role": "tool", "tool_call_id": i, "content": "18C sunny"}
OFF = {"thinking": {"type": "disabled"}}  # for OpenAI-compatible providers in structural probes

def structural():
    # MiniMax Anthropic: thinking omitted (doc: off by default for M3) -> also records default content types
    p = MA
    call(p, "a_no_tool_result", {"tools": WEATHER_A, "messages": [{"role": "user", "content": U0}, A_TU, {"role": "user", "content": "never mind, say hi"}]})
    call(p, "b_mismatched_id", {"tools": WEATHER_A, "messages": [{"role": "user", "content": U0}, A_TU, A_TR("toolu_99")]})
    call(p, "c_result_not_adjacent", {"tools": WEATHER_A, "messages": [{"role": "user", "content": U0}, A_TU, {"role": "user", "content": "hold on"}, {"role": "assistant", "content": "ok"}, A_TR()]})
    call(p, "d_two_consecutive_user", {"messages": [{"role": "user", "content": "Say A."}, {"role": "user", "content": "Now say B."}]})
    call(p, "e_system_in_messages", {"messages": [{"role": "user", "content": "Say hello."}, {"role": "system", "content": "Always answer in French."}]})
    call(p, "f_name_dot_slash", {"tools": [dict(WEATHER_A[0], name="mcp.server/tool")], "messages": [{"role": "user", "content": "hi"}]})
    call(p, "f_name_70chars", {"tools": [dict(WEATHER_A[0], name="t" * 70)], "messages": [{"role": "user", "content": "hi"}]})
    for p in (MO, DS):
        call(p, "a_no_tool_result", {**OFF, "tools": WEATHER_O, "messages": [{"role": "user", "content": U0}, O_TC, {"role": "user", "content": "never mind, say hi"}]})
        call(p, "b_mismatched_id", {**OFF, "tools": WEATHER_O, "messages": [{"role": "user", "content": U0}, O_TC, O_TR("call_99")]})
        call(p, "c_result_not_adjacent", {**OFF, "tools": WEATHER_O, "messages": [{"role": "user", "content": U0}, O_TC, {"role": "user", "content": "hold on"}, {"role": "assistant", "content": "ok"}, O_TR()]})
        call(p, "d_two_consecutive_user", {**OFF, "messages": [{"role": "user", "content": "Say A."}, {"role": "user", "content": "Now say B."}]})
        call(p, "f_name_dot_slash", {**OFF, "tools": [{"type": "function", "function": dict(WEATHER_O[0]["function"], name="mcp.server/tool")}], "messages": [{"role": "user", "content": "hi"}]})
        call(p, "f_name_70chars", {**OFF, "tools": [{"type": "function", "function": dict(WEATHER_O[0]["function"], name="t" * 70)}], "messages": [{"role": "user", "content": "hi"}]})

def thinking():
    G = "Call get_weather for Paris now. Do not deliberate."
    # MiniMax Anthropic, adaptive thinking on
    p = MA
    s, j = call(p, "g1_thinking_toolcall", {"thinking": {"type": "adaptive"}, "tools": WEATHER_A, "messages": [{"role": "user", "content": G}]})
    if s == 200:
        content = j["content"]; tus = [b for b in content if b["type"] == "tool_use"]
        if tus:
            tr = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tus[0]["id"], "content": "18C sunny"}]}
            stripped = [b for b in content if b["type"] not in ("thinking", "redacted_thinking")]
            call(p, "g2_drop_thinking", {"thinking": {"type": "adaptive"}, "tools": WEATHER_A, "messages": [{"role": "user", "content": G}, {"role": "assistant", "content": stripped}, tr]}, note="thinking blocks removed")
            call(p, "g3_keep_thinking", {"thinking": {"type": "adaptive"}, "tools": WEATHER_A, "messages": [{"role": "user", "content": G}, {"role": "assistant", "content": content}, tr]}, note="control: full content")
    # OpenAI-compatible: omit thinking param -> observe default
    for p in (MO, DS):
        s, j = call(p, "g1_default_thinking_toolcall", {"tools": WEATHER_O, "messages": [{"role": "user", "content": G}]}, note="thinking param omitted")
        if s != 200: continue
        m = j["choices"][0]["message"]
        if not m.get("tool_calls"): continue
        tr = {"role": "tool", "tool_call_id": m["tool_calls"][0]["id"], "content": "18C sunny"}
        content = m.get("content") or ""
        stripped = {"role": "assistant", "content": re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip(), "tool_calls": m["tool_calls"]}
        call(p, "g2_drop_reasoning", {"tools": WEATHER_O, "messages": [{"role": "user", "content": G}, stripped, tr]}, note="reasoning_content / <think> removed")
        full = {"role": "assistant", "content": content, "tool_calls": m["tool_calls"]}
        if m.get("reasoning_content") is not None: full["reasoning_content"] = m["reasoning_content"]
        if m.get("reasoning_details") is not None: full["reasoning_details"] = m["reasoning_details"]
        call(p, "g3_keep_reasoning", {"tools": WEATHER_O, "messages": [{"role": "user", "content": G}, full, tr]}, note="control: passed back")

PREFIX = "You are a meticulous code-review assistant. Follow these numbered house rules exactly.\n" + "\n".join(
    f"Rule {i}: When reviewing module m{i:03d}, check that function f{i}_{(i*7)%13} validates argument x{i%5} and logs at level {['debug','info','warn'][i%3]} before returning code {1000+i}." for i in range(1, 60))

def caching():
    msgs = [{"role": "user", "content": "Reply with just OK."}]
    call(MA, "h1_cache_plain", {"system": PREFIX, "messages": msgs, "max_tokens": 8})
    call(MA, "h2_cache_plain", {"system": PREFIX, "messages": msgs, "max_tokens": 8}, note="identical repeat, no cache_control")
    sysb = [{"type": "text", "text": PREFIX, "cache_control": {"type": "ephemeral"}}]
    call(MA, "h3_cache_control", {"system": sysb, "messages": msgs, "max_tokens": 8})
    call(MA, "h4_cache_control", {"system": sysb, "messages": msgs, "max_tokens": 8}, note="identical repeat with cache_control")
    om = [{"role": "system", "content": PREFIX}] + msgs
    call(MO, "h1_cache", {**OFF, "messages": om, "max_tokens": 8})
    call(MO, "h2_cache", {**OFF, "messages": om, "max_tokens": 8}, note="identical repeat")
    for i in range(1, 4):
        call(DS, f"h{i}_cache", {**OFF, "messages": om, "max_tokens": 8}, note="identical repeat" if i > 1 else "")
        time.sleep(8)

def truncation():
    W_A = [{"name": "write_file", "description": "Write a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}]
    W_O = [{"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": W_A[0]["input_schema"]}}]
    ask = "Use write_file to create poem.txt containing a 20-line poem about the sea. Call the tool immediately."
    call(MA, "i_maxtokens_in_toolcall", {"tools": W_A, "messages": [{"role": "user", "content": ask}], "max_tokens": 40})
    call(MO, "i_maxtokens_in_toolcall", {**OFF, "tools": W_O, "messages": [{"role": "user", "content": ask}], "max_tokens": 40})
    call(DS, "i_maxtokens_in_toolcall", {**OFF, "tools": W_O, "messages": [{"role": "user", "content": ask}], "max_tokens": 40})

def streaming():
    G = "Call get_weather for Paris now."
    stream(MA, "j_stream_toolcall", {"tools": WEATHER_A, "messages": [{"role": "user", "content": G}]})
    for p in (MO, DS):
        stream(p, "j_stream_no_include_usage", {**OFF, "tools": WEATHER_O, "messages": [{"role": "user", "content": G}]})
        stream(p, "j_stream_include_usage", {**OFF, "tools": WEATHER_O, "messages": [{"role": "user", "content": G}], "stream_options": {"include_usage": True}})

if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
    save(); print("FINAL", STATE)
