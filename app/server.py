"""HTTP server (standard library). Run: python3 -m app.server"""
import json
import mimetypes
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import ai, engine, ledger, model, service
from .db import connect, iso, now as real_now

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
LOCK = threading.Lock()
TOKEN = os.environ.get("KEPT_TOKEN", "")           # if set, every /api route except the public ones needs it
PUBLIC = {"/api/scorecard", "/api/health", "/api/eta"}   # aggregates only; CORS-enabled so a shop page can embed them
MAX_BODY = 2_000_000


def _int(q, key, default, lo, hi):
    """Integer query parameter, clamped to [lo, hi]."""
    try:
        return max(lo, min(hi, int(q.get(key, [default])[0])))
    except ValueError:
        raise ValueError(f"{key} must be a whole number")


def _routes():
    def g(conn, now, q, body, *m):
        return None
    R = []

    def route(method, pattern):
        def deco(fn):
            R.append((method, re.compile("^" + pattern + "$"), fn))
            return fn
        return deco

    @route("GET", "/api/health")
    def health(conn, now, q, b): return {"ok": True, "ai": ai.status()}

    @route("GET", "/api/summary")
    def summary(conn, now, q, b): return service.summary(conn, now)

    @route("GET", "/api/scorecard")
    def scorecard(conn, now, q, b): return engine.scorecard(conn, now, _int(q, "days", 90, 1, 3650))

    @route("GET", "/api/trend")
    def trend(conn, now, q, b): return {"weeks": engine.trend(conn, now, _int(q, "weeks", 12, 2, 104))}

    @route("GET", "/api/promises")
    def promises_get(conn, now, q, b): return {"config": engine.get_config(conn), "regions": engine.regions(conn)}

    @route("POST", "/api/promises")
    def promises_set(conn, now, q, b): return {"config": engine.set_config(conn, b, now)}

    @route("GET", "/api/eta")
    def eta(conn, now, q, b): return engine.quote_eta(conn, q.get("region", [""])[0], now)

    @route("POST", "/api/orders")
    def order_new(conn, now, q, b): return service.create_order(conn, now, b)

    @route("POST", r"/api/orders/(?P<oid>[\w-]+)/deliver")
    def order_deliver(conn, now, q, b, oid):
        return {"outcomes": service.deliver_order(conn, now, oid, b.get("final_total"), b.get("delivered_at"), b.get("after_days"))}

    @route("GET", "/api/tickets")
    def tickets(conn, now, q, b): return {"tickets": service.list_tickets(conn, now, q.get("status", ["open"])[0])}

    @route("POST", "/api/tickets")
    def ticket_new(conn, now, q, b): return service.create_ticket(conn, now, b)

    @route("POST", r"/api/tickets/(?P<tid>\d+)/triage")
    def ticket_triage(conn, now, q, b, tid): return service.triage_ticket(conn, now, int(tid))

    @route("POST", r"/api/tickets/(?P<tid>\d+)/reply")
    def ticket_reply(conn, now, q, b, tid): return {"outcomes": service.reply_ticket(conn, now, int(tid))}

    @route("POST", r"/api/tickets/(?P<tid>\d+)/resolve")
    def ticket_resolve(conn, now, q, b, tid): return service.resolve_ticket(conn, now, int(tid))

    @route("GET", "/api/radar")
    def radar(conn, now, q, b): return model.score_all(conn, now, _int(q, "limit", 30, 1, 500))

    @route("POST", r"/api/radar/(?P<cid>[\w-]+)/draft")
    def radar_draft(conn, now, q, b, cid): return service.draft_for_customer(conn, now, cid)

    @route("POST", "/api/model/retrain")
    def retrain(conn, now, q, b): return model.train(conn, now)

    @route("GET", "/api/customers")
    def customers(conn, now, q, b): return {"customers": service.list_customers(conn, q.get("q", [""])[0])}

    @route("GET", r"/api/customers/(?P<cid>[\w-]+)")
    def customer(conn, now, q, b, cid): return service.customer_detail(conn, now, cid)

    @route("GET", "/api/ledger")
    def ledger_list(conn, now, q, b):
        lim = _int(q, "limit", 30, 1, 200)
        rows = [dict(r) for r in conn.execute("SELECT seq,ts,kind,ref,payload,hash FROM ledger ORDER BY seq DESC LIMIT ?", (lim,))]
        return {"entries": rows, "verify": ledger.verify(conn)}

    @route("GET", "/api/ledger/verify")
    def ledger_verify(conn, now, q, b): return ledger.verify(conn)

    @route("POST", "/api/import/orders")
    def imp_o(conn, now, q, b): return service.import_orders(conn, now, b.get("csv", ""))

    @route("POST", "/api/import/tickets")
    def imp_t(conn, now, q, b): return service.import_tickets(conn, now, b.get("csv", ""))

    @route("POST", "/api/ask")
    def ask(conn, now, q, b):
        question = (b.get("question") or "").strip()
        if not question:
            raise ValueError("question is required")
        return service.ask(conn, now, question)

    @route("GET", "/api/ai/log")
    def ai_log(conn, now, q, b):
        return {"log": [dict(r) for r in conn.execute("SELECT id,ts,feature,source,note,output FROM ai_log ORDER BY id DESC LIMIT 20")],
                "status": ai.status()}
    return R


ROUTES = _routes()


class Handler(BaseHTTPRequestHandler):
    server_version = "Kept/1.0"

    def log_message(self, fmt, *a):
        if os.environ.get("KEPT_QUIET") != "1":
            super().log_message(fmt, *a)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if getattr(self, "public", False):
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, method):
        u = urlparse(self.path)
        self.public = u.path in PUBLIC and method == "GET"
        if not u.path.startswith("/api/"):
            return self._static(u.path) if method == "GET" else self._send(405, {"error": "method not allowed"})
        if TOKEN and u.path not in PUBLIC and self.headers.get("X-Kept-Token") != TOKEN:
            return self._send(401, {"error": "token required"})
        body = {}
        if method == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._send(413, {"error": "body too large"})
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON"})
            if not isinstance(body, dict):
                return self._send(400, {"error": "JSON object expected"})
        for m, rx, fn in ROUTES:
            mt = rx.match(u.path)
            if m == method and mt:
                with LOCK:
                    conn = connect(self.server.db_path)
                    try:
                        return self._send(200, fn(conn, self.server.clock(), parse_qs(u.query), body, **mt.groupdict()))
                    except KeyError as e:
                        return self._send(404, {"error": str(e).strip("'\"")})
                    except (ValueError, TypeError) as e:
                        return self._send(400, {"error": str(e)})
                    finally:
                        conn.close()
        self._send(404, {"error": "not found"})

    def _static(self, path):
        path = "/index.html" if path in ("/", "") else path
        full = os.path.normpath(os.path.join(STATIC, path.lstrip("/")))
        if not full.startswith(STATIC) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        with open(full, "rb") as f:
            self._send(200, f.read(), mimetypes.guess_type(full)[0] or "application/octet-stream")

    def do_GET(self): self._handle("GET")
    def do_POST(self): self._handle("POST")


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, db_path=None, clock=real_now):
        super().__init__(addr, Handler)
        self.db_path = db_path
        self.clock = clock


def _sweeper(srv, every=60):
    while True:
        time.sleep(every)
        try:
            with LOCK:
                c = connect(srv.db_path)
                engine.sweep(c, srv.clock())
                c.close()
        except Exception as e:      # never let the background job die silently
            print("sweep error:", e)


def main():
    host = os.environ.get("KEPT_HOST", "127.0.0.1")
    port = int(os.environ.get("KEPT_PORT", "8000"))
    c = connect()
    empty = c.execute("SELECT COUNT(*) n FROM customers").fetchone()["n"] == 0
    c.close()
    if empty and os.environ.get("KEPT_NO_SEED") != "1":
        print("Empty database: generating synthetic demo data (about 3 seconds)...")
        from . import seed
        seed.build().close()
    srv = Server((host, port))
    threading.Thread(target=_sweeper, args=(srv,), daemon=True).start()
    print(f"Kept running at http://{host}:{port}   AI: {ai.status()['mode']}")
    if TOKEN:
        print("Token protection ON (send X-Kept-Token).")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
