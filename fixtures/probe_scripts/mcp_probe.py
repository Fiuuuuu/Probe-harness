import subprocess, json, sys, time, threading, queue, os
HERE = os.path.dirname(os.path.abspath(__file__))
def run(seq, label):
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "mcp_server.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l) for l in p.stdout], daemon=True).start()
    out = {"label": label, "exchanges": []}
    for raw, wait in seq:
        try:
            p.stdin.write(raw + "\n"); p.stdin.flush()
        except BrokenPipeError:
            out["exchanges"].append({"sent": raw[:200], "error": "BrokenPipe: server process already exited"}); break
        got = []
        t = time.time()
        while time.time() - t < wait:
            try: got.append(q.get(timeout=0.2).strip())
            except queue.Empty: pass
        out["exchanges"].append({"sent": raw[:200], "received": [g[:400] for g in got]})
    try: p.stdin.close()
    except BrokenPipeError: pass
    try: p.wait(timeout=5)
    except Exception: p.kill()
    out["exit_code"] = p.returncode
    out["stderr_tail"] = p.stderr.read()[-800:]
    return out
INIT = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "probe", "version": "0"}}})
INITED = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
LIST = lambda i: json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/list", "params": {}})
res = []
res.append(run([(LIST(1), 3)], "1_tools_list_before_initialize"))
res.append(run([(INIT, 3), (LIST(2), 3)], "2_initialize_then_list_without_initialized_notification"))
res.append(run([(INIT, 3), (INITED, 1), (LIST(2), 3)], "3_correct_order"))
pretty = json.dumps(json.loads(INIT), indent=2)
res.append(run([(pretty, 3)], "4_multiline_pretty_json_initialize"))
json.dump(res, open(os.path.join(HERE, "mcp_results.json"), "w"), indent=1, ensure_ascii=False)
for r in res: print(json.dumps(r, ensure_ascii=False)[:1500]); print()
