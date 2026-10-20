"""Validate an actual public health/CORS/SSE path, never an HTML-200 surrogate."""
import argparse
import datetime
import json
from pathlib import Path
import time

import httpx


def check(base, origin, expected_engine="continuous_cached_greedy"):
    base = base.rstrip("/")
    with httpx.Client(timeout=180) as client:
        health_response = client.get(base + "/health", headers={"Origin":origin})
        health_response.raise_for_status()
        assert "application/json" in health_response.headers.get("content-type", ""), "Health is not JSON"
        health = health_response.json()
        assert health["status"] == "ok" and health["engine"] == expected_engine
        assert health["serving_cache"] is (expected_engine == "continuous_cached_greedy")
        cors = {"Origin":origin, "Access-Control-Request-Method":"POST", "Access-Control-Request-Headers":"content-type"}
        preflight = client.options(base + "/generate/stream", headers=cors)
        preflight.raise_for_status()
        assert preflight.headers.get("access-control-allow-origin") == origin
        denied = client.options(base + "/generate/stream", headers={**cors,"Origin":"https://untrusted.example"})
        assert "access-control-allow-origin" not in denied.headers
        body = {"prompt":"The capital of France is", "max_new_tokens":8}
        events = []
        started = time.perf_counter()
        with client.stream("POST",base + "/generate/stream",json=body,headers={"Origin":origin}) as response:
            response.raise_for_status()
            assert "text/event-stream" in response.headers.get("content-type", "")
            assert response.headers.get("access-control-allow-origin") == origin
            kind, data = "message", []
            for line in response.iter_lines():
                if not line:
                    if data:
                        events.append({"event":kind,"arrival_ms":round((time.perf_counter()-started)*1000,3),"data":json.loads("\n".join(data))})
                    kind, data = "message", []
                elif line.startswith("event:"):
                    kind = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
        tokens = [e for e in events if e["event"] == "token"]
        done = [e for e in events if e["event"] == "done"]
        assert len(tokens)==8 and len(done)==1 and not any(e["event"]=="error" for e in events)
        assert [e["data"]["token_index"] for e in tokens] == list(range(1,9))
        assert [e["data"]["token_id"] for e in tokens] == done[0]["data"]["generated_token_ids"]
        snapshot_response=client.get(base+"/runtime/snapshot",headers={"Origin":origin})
        snapshot_response.raise_for_status()
        snapshot=snapshot_response.json()
        assert snapshot["engine"]==health["engine"] and snapshot["serving_cache"] is health["serving_cache"]
        return {"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"backend_url":base,
                "health_url":base+"/health","health":health,"allowed_origin":origin,
                "cors_preflight_status":preflight.status_code,"untrusted_origin_preflight_status":denied.status_code,
                "request":body,"events":events,"token_events":len(tokens),"done_events":len(done),"error_events":0,
                "generated_text":done[0]["data"]["generated_text"],"snapshot_engine":snapshot["engine"],
                "scope":"Real public HTTP check. Browser Playground/console/network signoff is separate."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",required=True)
    parser.add_argument("--origin",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-engine", choices=["continuous_cached_greedy", "continuous_uncached_greedy"], default="continuous_cached_greedy")
    args=parser.parse_args()
    result=check(args.base_url,args.origin,args.expected_engine)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps({key:result[key] for key in ("backend_url","health","token_events","done_events","error_events","generated_text")},indent=2))
