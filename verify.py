"""Quick verification script — starts server, tests all endpoints, writes results to file."""
import subprocess, time, sys, json, os

# Start server with suppressed output
server = subprocess.Popen(
    [sys.executable, "main.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(3)

results = []
try:
    import httpx
    c = httpx.Client(base_url="http://localhost:7860", timeout=10)

    # 1. Health
    h = c.get("/health").json()
    results.append(f"[OK] GET /health -> {h}")

    # 2. Tasks
    t = c.get("/tasks").json()
    results.append(f"[OK] GET /tasks  -> {len(t['tasks'])} tasks, {len(t['action_types'])} action types")

    # 3. Reset
    r = c.post("/reset?task_idx=0").json()
    results.append(f"[OK] POST /reset -> {len(r['open_tickets'])} open tickets")

    # 4. Step (classify correctly)
    s = c.post("/step", json={"ticket_id": "T001", "action_type": "classify", "target": "it_support"}).json()
    results.append(f"[OK] POST /step  -> reward={s['reward']}, done={s['done']}")
    results.append(f"     Feedback: {s['info']['feedback']}")

    # 5. Grader
    g = c.get("/grader").json()
    results.append(f"[OK] GET /grader -> score={g['score']}")

    # 6. State
    st = c.get("/state").json()
    results.append(f"[OK] GET /state  -> step_count={st['step_count']}, done={st['done']}")

    c.close()
    results.append("")
    results.append("ALL 6 ENDPOINTS VERIFIED SUCCESSFULLY")

except Exception as e:
    results.append(f"[FAIL] Error: {e}")

finally:
    server.terminate()
    server.wait()

# Write to file
with open("verify_results.txt", "w") as f:
    f.write("\n".join(results))

# Print results
print("\n".join(results))
