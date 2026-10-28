"""Promise engine: evaluates promises, issues capped credits, quotes honest ETAs,
builds the scorecard from the ledger, and computes discount-free retention."""
import json
import math

from . import ledger
from .db import days, add_days, iso, parse

DEFAULT_PROMISES = {
    "delivery": {"enabled": True, "credit_pct": 10, "max_credit": 150},
    "price": {"enabled": True},
    "reply": {"enabled": True, "hours": 4, "credit": 30},
    "caps": {"per_customer_90d": 300},
    "margin_pct": 25,
    "target_kept_rate": 0.90,
}


def get_config(conn):
    row = conn.execute("SELECT value FROM config WHERE key='promises'").fetchone()
    cfg = json.loads(json.dumps(DEFAULT_PROMISES))
    if row:
        stored = json.loads(row["value"])
        for k, v in stored.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def set_config(conn, updates, ts):
    """Validate, store and log a rule change on the ledger (rules are part of the record)."""
    cfg = get_config(conn)
    def num(v, lo, hi, name):
        v = float(v)
        if not (lo <= v <= hi):
            raise ValueError(f"{name} must be between {lo} and {hi}")
        return v
    for k, v in (updates or {}).items():
        if k == "delivery":
            if "enabled" in v: cfg["delivery"]["enabled"] = bool(v["enabled"])
            if "credit_pct" in v: cfg["delivery"]["credit_pct"] = num(v["credit_pct"], 0, 50, "credit_pct")
            if "max_credit" in v: cfg["delivery"]["max_credit"] = num(v["max_credit"], 0, 5000, "max_credit")
        elif k == "price":
            if "enabled" in v: cfg["price"]["enabled"] = bool(v["enabled"])
        elif k == "reply":
            if "enabled" in v: cfg["reply"]["enabled"] = bool(v["enabled"])
            if "hours" in v: cfg["reply"]["hours"] = num(v["hours"], 0.5, 72, "reply hours")
            if "credit" in v: cfg["reply"]["credit"] = num(v["credit"], 0, 1000, "reply credit")
        elif k == "caps":
            if "per_customer_90d" in v: cfg["caps"]["per_customer_90d"] = num(v["per_customer_90d"], 0, 10000, "cap")
        elif k == "margin_pct":
            cfg["margin_pct"] = num(v, 1, 90, "margin_pct")
        elif k == "target_kept_rate":
            cfg["target_kept_rate"] = num(v, 0.5, 0.999, "target_kept_rate")
        else:
            raise ValueError(f"unknown setting: {k}")
    conn.execute("INSERT OR REPLACE INTO config(key,value) VALUES('promises',?)", (json.dumps(cfg),))
    ledger.append(conn, "config_changed", "promises", cfg, iso(ts))
    conn.commit()
    return cfg


# ---------------------------------------------------------------- credits

def _credit_room(conn, customer_id, now, cfg):
    since = iso(add_days(now, -90))
    spent = conn.execute(
        "SELECT COALESCE(SUM(amount),0) s FROM credits WHERE customer_id=? AND ts>=? AND promise!='price'",
        (customer_id, since),
    ).fetchone()["s"]
    return max(0.0, cfg["caps"]["per_customer_90d"] - spent)


def _issue_credit(conn, customer_id, order_id, promise, amount, now, cfg, capped=True):
    amount = round(float(amount), 2)
    if capped:
        room = _credit_room(conn, customer_id, now, cfg)
        if amount > room:
            ledger.append(conn, "credit_capped", order_id or "", {
                "customer": customer_id, "promise": promise, "wanted": amount, "paid": round(room, 2)}, iso(now))
            amount = round(room, 2)
    if amount <= 0:
        return 0.0
    seq = ledger.append(conn, "credit_issued", order_id or "", {
        "customer": customer_id, "promise": promise, "amount": amount}, iso(now))
    conn.execute(
        "INSERT INTO credits(customer_id,order_id,promise,amount,ts,ledger_seq) VALUES(?,?,?,?,?,?)",
        (customer_id, order_id, promise, amount, iso(now), seq))
    return amount


def _record(conn, promise, ref, kept, payload, now):
    """Idempotent: each (promise, ref) is judged once."""
    try:
        conn.execute("INSERT INTO evaluations(promise,ref) VALUES(?,?)", (promise, ref))
    except Exception:
        return False
    ledger.append(conn, "promise_kept" if kept else "promise_broken", ref, dict(payload, promise=promise), iso(now))
    return True


# ------------------------------------------------------------ evaluation

def evaluate_order(conn, order_id, now):
    """Judge delivery and price promises for one order as of `now`. Returns list of outcomes."""
    cfg = get_config(conn)
    o = conn.execute("SELECT o.*, c.grp FROM orders o JOIN customers c ON c.id=o.customer_id WHERE o.id=?",
                     (order_id,)).fetchone()
    if not o:
        raise KeyError("order not found")
    out = []
    holdout = o["grp"] == "holdout"
    delivered = parse(o["delivered_at"])
    promised = parse(o["promised_at"])
    base = {"customer": o["customer_id"], "region": o["region"], "holdout": holdout}

    if cfg["delivery"]["enabled"]:
        decided = None
        if delivered and delivered <= now:
            decided = delivered <= promised
            detail = {"late_by_days": round(max(0, days(promised, delivered)), 2)}
        elif promised < now:                      # overdue and still undelivered: detected, not complained about
            decided = False
            detail = {"detected_before_delivery": True, "overdue_days": round(days(promised, now), 2)}
        if decided is not None and _record(conn, "delivery", order_id, decided, dict(base, **detail), now):
            out.append({"promise": "delivery", "kept": decided})
            if not decided and not holdout:
                amt = min(o["quoted_total"] * cfg["delivery"]["credit_pct"] / 100.0, cfg["delivery"]["max_credit"])
                paid = _issue_credit(conn, o["customer_id"], order_id, "delivery", amt, now, cfg)
                out[-1]["credit"] = paid

    if cfg["price"]["enabled"] and delivered and delivered <= now and o["final_total"] is not None:
        extra = round(o["final_total"] - o["quoted_total"], 2)
        kept = extra <= 0.005
        if _record(conn, "price", order_id, kept, dict(base, extra=max(0, extra)), now):
            out.append({"promise": "price", "kept": kept})
            if not kept and not holdout:          # a surprise charge is refunded in full, outside the cap
                out[-1]["credit"] = _issue_credit(conn, o["customer_id"], order_id, "price", extra, now, cfg, capped=False)
    conn.commit()
    return out


def ticket_due(ticket, cfg):
    return add_days(parse(ticket["created_at"]), cfg["reply"]["hours"] / 24.0)


def evaluate_ticket(conn, ticket_id, now):
    cfg = get_config(conn)
    if not cfg["reply"]["enabled"]:
        return []
    t = conn.execute("SELECT t.*, c.grp FROM tickets t JOIN customers c ON c.id=t.customer_id WHERE t.id=?",
                     (ticket_id,)).fetchone()
    if not t:
        raise KeyError("ticket not found")
    due = ticket_due(t, cfg)
    replied = parse(t["first_reply_at"])
    holdout = t["grp"] == "holdout"
    decided = None
    if replied and replied <= now:
        decided = replied <= due
    elif due < now:
        decided = False
    out = []
    if decided is not None and _record(conn, "reply", f"T{ticket_id}", decided,
                                       {"customer": t["customer_id"], "region": None, "holdout": holdout}, now):
        out.append({"promise": "reply", "kept": decided})
        if not decided and not holdout:
            out[-1]["credit"] = _issue_credit(conn, t["customer_id"], t["order_id"], "reply",
                                              cfg["reply"]["credit"], now, cfg)
    conn.commit()
    return out


def sweep(conn, now):
    """Proactive detection: find broken promises nobody has complained about yet."""
    found = []
    for r in conn.execute("""SELECT id FROM orders WHERE (delivered_at IS NULL OR delivered_at>?) AND promised_at<?
                             AND id NOT IN (SELECT ref FROM evaluations WHERE promise='delivery')""",
                          (iso(now), iso(now))).fetchall():
        found += [dict(order=r["id"], **x) for x in evaluate_order(conn, r["id"], now)]
    cfg = get_config(conn)
    cutoff = iso(add_days(now, -cfg["reply"]["hours"] / 24.0))
    for r in conn.execute("""SELECT id FROM tickets WHERE first_reply_at IS NULL AND created_at<?
                             AND ('T'||id) NOT IN (SELECT ref FROM evaluations WHERE promise='reply')""",
                          (cutoff,)).fetchall():
        found += [dict(ticket=r["id"], **x) for x in evaluate_ticket(conn, r["id"], now)]
    return found


# --------------------------------------------------------------- tiers

def tier_of(conn, customer_id, now):
    """Earned SERVICE tier (faster, more personal service). Everyone keeps the same promises."""
    n = conn.execute("SELECT COUNT(*) n FROM orders WHERE customer_id=? AND placed_at>=?",
                     (customer_id, iso(add_days(now, -365)))).fetchone()["n"]
    return "Priority" if n >= 6 else "Regular" if n >= 3 else "Member"


# ----------------------------------------------------------------- ETA

def quote_eta(conn, region, now, window_days=180):
    cfg = get_config(conn)
    rows = conn.execute("SELECT placed_at, delivered_at, quoted_total FROM orders WHERE region=? AND delivered_at IS NOT NULL AND placed_at>=?",
                        (region, iso(add_days(now, -window_days)))).fetchall()
    d = sorted(days(parse(r["placed_at"]), parse(r["delivered_at"])) for r in rows)
    n = len(d)
    if n == 0:
        return {"region": region, "n": 0, "enough_data": False}
    def pct(p):
        return d[min(n - 1, int(math.ceil(p * n)) - 1)]
    avg_total = sum(r["quoted_total"] for r in rows) / n
    avg_credit = min(avg_total * cfg["delivery"]["credit_pct"] / 100.0, cfg["delivery"]["max_credit"])
    table = []
    for k in range(1, 15):
        rate = sum(1 for x in d if x <= k) / n
        table.append({"promise_days": k, "kept_rate": round(rate, 3),
                      "expected_credit_per_order": round((1 - rate) * avg_credit, 2)})
    rec = next((t["promise_days"] for t in table if t["kept_rate"] >= cfg["target_kept_rate"]), 14)
    return {"region": region, "n": n, "enough_data": n >= 20, "p50": round(pct(.5), 2), "p80": round(pct(.8), 2),
            "p95": round(pct(.95), 2), "recommended_days": rec, "target": cfg["target_kept_rate"], "table": table}


def regions(conn):
    return [r["region"] for r in conn.execute("SELECT DISTINCT region FROM orders ORDER BY region")]


# ------------------------------------------------------------ scorecard

def scorecard(conn, now, window=90):
    """Built from ledger rows only. Holdout customers are excluded (they are not covered by Kept)."""
    since = iso(add_days(now, -window))
    tally = {}
    credits = 0.0
    capped = 0
    for r in conn.execute("SELECT kind, payload FROM ledger WHERE ts>=? AND kind IN ('promise_kept','promise_broken','credit_issued','credit_capped')", (since,)):
        p = json.loads(r["payload"])
        if r["kind"] == "credit_issued":
            credits += p["amount"]; continue
        if r["kind"] == "credit_capped":
            capped += 1; continue
        if p.get("holdout"):
            continue
        t = tally.setdefault(p["promise"], {"kept": 0, "broken": 0})
        t["kept" if r["kind"] == "promise_kept" else "broken"] += 1
    rows = []
    tk = tb = 0
    for k, v in sorted(tally.items()):
        tot = v["kept"] + v["broken"]
        rows.append({"promise": k, "kept": v["kept"], "broken": v["broken"], "total": tot,
                     "kept_rate": round(v["kept"] / tot, 4) if tot else None})
        tk += v["kept"]; tb += v["broken"]
    v = ledger.verify(conn)
    return {"window_days": window, "from": since, "to": iso(now), "promises": rows,
            "overall_kept_rate": round(tk / (tk + tb), 4) if (tk + tb) else None,
            "judged": tk + tb, "credits_paid": round(credits, 2), "credit_events_capped": capped,
            "ledger": {"ok": v["ok"], "entries": v["count"], "head": v["head"]}}


# ------------------------------------------------------------- retention

def _wilson_diff(k1, n1, k2, n2):
    if not n1 or not n2:
        return None
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return {"diff": round(d, 4), "low": round(d - 1.96 * se, 4), "high": round(d + 1.96 * se, 4)}


def retention(conn, now):
    """Cohort: customers whose first order was 180-365 days ago.
    Retained = ordered in the last 90 days. Discount-free = at least one undiscounted order in that window."""
    lo, hi = iso(add_days(now, -365)), iso(add_days(now, -180))
    recent = iso(add_days(now, -90))
    firsts = conn.execute("""SELECT c.id, c.grp, MIN(o.placed_at) f FROM customers c JOIN orders o ON o.customer_id=c.id
                             GROUP BY c.id HAVING f>=? AND f<?""", (lo, hi)).fetchall()
    stats = {"kept": dict(n=0, retained=0, discount_free=0, coupon_bought=0),
             "holdout": dict(n=0, retained=0, discount_free=0, coupon_bought=0)}
    for c in firsts:
        s = stats[c["grp"]]
        s["n"] += 1
        rows = conn.execute("SELECT discount FROM orders WHERE customer_id=? AND placed_at>=?", (c["id"], recent)).fetchall()
        if rows:
            s["retained"] += 1
            if any((r["discount"] or 0) <= 0 for r in rows):
                s["discount_free"] += 1
            else:
                s["coupon_bought"] += 1
    for g, s in stats.items():
        s["retention"] = round(s["retained"] / s["n"], 4) if s["n"] else None
        s["discount_free_retention"] = round(s["discount_free"] / s["n"], 4) if s["n"] else None
    k, h = stats["kept"], stats["holdout"]
    return {"cohort_window": [lo, hi], "retained_window_days": 90, "groups": stats,
            "difference_discount_free": _wilson_diff(k["discount_free"], k["n"], h["discount_free"], h["n"]),
            "small_sample": min(k["n"], h["n"]) < 100}


def operations_metrics(conn, now):
    cfg = get_config(conn)
    since = iso(add_days(now, -90))
    # repeat contact: customers with 2+ tickets in 90d among customers with any ticket
    tk = conn.execute("SELECT customer_id, COUNT(*) n FROM tickets WHERE created_at>=? GROUP BY customer_id", (since,)).fetchall()
    repeat = sum(1 for r in tk if r["n"] >= 2)
    # decision rule: credits paid vs gross margin on orders placed by credited customers within 90d after the credit
    credited = conn.execute("SELECT customer_id, MIN(ts) t, SUM(amount) a FROM credits WHERE ts>=? GROUP BY customer_id", (since,)).fetchall()
    cost = sum(r["a"] for r in credited)
    follow = 0.0
    for r in credited:
        follow += conn.execute("SELECT COALESCE(SUM(final_total),0) s FROM orders WHERE customer_id=? AND placed_at>?",
                               (r["customer_id"], r["t"])).fetchone()["s"]
    margin = follow * cfg["margin_pct"] / 100.0
    reply = conn.execute("""SELECT AVG((julianday(first_reply_at)-julianday(created_at))*24) h FROM tickets
                            WHERE first_reply_at IS NOT NULL AND created_at>=?""", (since,)).fetchone()["h"]
    return {"customers_with_tickets_90d": len(tk), "repeat_contact_rate": round(repeat / len(tk), 4) if tk else None,
            "avg_first_reply_hours": round(reply, 2) if reply is not None else None,
            "credited_customers": len(credited), "credits_cost_90d": round(cost, 2),
            "gross_margin_on_followup_orders": round(margin, 2),
            "decision_rule_passes": margin > cost,
            "decision_rule_note": "Gross, not causal: it counts every later order by credited customers. "
                                  "Only the holdout comparison can show what the credits actually protected."}
