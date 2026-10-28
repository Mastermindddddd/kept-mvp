"""Application logic used by the HTTP layer and the seed script."""
import csv
import io
import json

from . import ai, engine, ledger, model
from .db import add_days, days, iso, parse

TIER_RANK = {"Priority": 0, "Regular": 1, "Member": 2}


def replay(conn, order_ids, ticket_ids, now):
    """Judge historical orders/tickets in time order, so the ledger reads like real life."""
    ev = []
    for oid in order_ids:
        o = conn.execute("SELECT promised_at, delivered_at FROM orders WHERE id=?", (oid,)).fetchone()
        p, d = parse(o["promised_at"]), parse(o["delivered_at"])
        if p <= now and (d is None or d > p):
            ev.append((p, "o", oid))
        if d and d <= now:
            ev.append((d, "o", oid))
    cfg = engine.get_config(conn)
    for tid in ticket_ids:
        t = conn.execute("SELECT created_at, first_reply_at FROM tickets WHERE id=?", (tid,)).fetchone()
        due = engine.ticket_due(t, cfg)
        r = parse(t["first_reply_at"])
        if due <= now and (r is None or r > due):
            ev.append((due, "t", tid))
        if r and r <= now:
            ev.append((r, "t", tid))
    ev.sort(key=lambda e: e[0])
    for ts, kind, ident in ev:
        (engine.evaluate_order if kind == "o" else engine.evaluate_ticket)(conn, ident, ts)


# ------------------------------------------------------------- orders

def create_order(conn, now, d):
    cid = d.get("customer_id")
    if not conn.execute("SELECT 1 FROM customers WHERE id=?", (cid,)).fetchone():
        raise ValueError("unknown customer")
    region = d.get("region") or conn.execute("SELECT region FROM customers WHERE id=?", (cid,)).fetchone()["region"]
    quoted = float(d.get("quoted_total"))
    if quoted <= 0:
        raise ValueError("quoted_total must be positive")
    pdays = d.get("promised_days")
    if pdays in (None, ""):
        pdays = engine.quote_eta(conn, region, now).get("recommended_days", 3)
    pdays = float(pdays)
    n = conn.execute("SELECT COUNT(*) n FROM orders").fetchone()["n"] + 1
    oid = f"O{100000 + n}"
    while conn.execute("SELECT 1 FROM orders WHERE id=?", (oid,)).fetchone():
        n += 1; oid = f"O{100000 + n}"
    conn.execute("INSERT INTO orders(id,customer_id,region,placed_at,promised_days,promised_at,quoted_total,discount) VALUES(?,?,?,?,?,?,?,?)",
                 (oid, cid, region, iso(now), pdays, iso(add_days(now, pdays)), quoted, float(d.get("discount") or 0)))
    conn.commit()
    return {"order_id": oid, "promised_days": pdays, "promised_at": iso(add_days(now, pdays))}


def deliver_order(conn, now, oid, final_total=None, delivered_at=None):
    o = conn.execute("SELECT * FROM orders WHERE id=?", (oid,)).fetchone()
    if not o:
        raise KeyError("order not found")
    if o["delivered_at"]:
        raise ValueError("order already delivered")
    when = parse(delivered_at) if delivered_at else now
    final = float(final_total) if final_total not in (None, "") else o["quoted_total"]
    conn.execute("UPDATE orders SET delivered_at=?, final_total=? WHERE id=?", (iso(when), final, oid))
    conn.commit()
    return engine.evaluate_order(conn, oid, max(now, when))


# ------------------------------------------------------------ tickets

def _order_facts(conn, order_id):
    if not order_id:
        return ""
    o = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
    if not o:
        return ""
    return f"order {o['id']} promised in {o['promised_days']:g} days, " + ("delivered" if o["delivered_at"] else "not yet delivered")


def create_ticket(conn, now, d):
    cid = d.get("customer_id")
    if not conn.execute("SELECT 1 FROM customers WHERE id=?", (cid,)).fetchone():
        raise ValueError("unknown customer")
    text = (d.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")
    cur = conn.execute("INSERT INTO tickets(customer_id,order_id,created_at,text) VALUES(?,?,?,?)",
                       (cid, d.get("order_id") or None, iso(now), text[:4000]))
    tid = cur.lastrowid
    conn.commit()
    return triage_ticket(conn, now, tid)


def triage_ticket(conn, now, tid):
    t = conn.execute("SELECT * FROM tickets WHERE id=?", (tid,)).fetchone()
    if not t:
        raise KeyError("ticket not found")
    r = ai.triage(conn, now, t["text"], _order_facts(conn, t["order_id"]))
    conn.execute("UPDATE tickets SET category=?,urgency=?,sentiment=?,root_cause=?,suggested_reply=?,triage_source=? WHERE id=?",
                 (r["category"], r["urgency"], r["sentiment"], r["root_cause"], r["suggested_reply"], r["source"], tid))
    conn.commit()
    return dict(r, ticket_id=tid)


def reply_ticket(conn, now, tid):
    t = conn.execute("SELECT * FROM tickets WHERE id=?", (tid,)).fetchone()
    if not t:
        raise KeyError("ticket not found")
    if not t["first_reply_at"]:
        conn.execute("UPDATE tickets SET first_reply_at=? WHERE id=?", (iso(now), tid))
        conn.commit()
    return engine.evaluate_ticket(conn, tid, now)


def resolve_ticket(conn, now, tid):
    conn.execute("UPDATE tickets SET resolved_at=COALESCE(resolved_at,?) WHERE id=?", (iso(now), tid))
    conn.commit()
    return {"ok": True}


def list_tickets(conn, now, status="open", limit=40):
    cfg = engine.get_config(conn)
    q = "SELECT t.*, c.name, c.grp FROM tickets t JOIN customers c ON c.id=t.customer_id"
    q += " WHERE t.resolved_at IS NULL" if status == "open" else ""
    rows = conn.execute(q + " ORDER BY t.created_at DESC LIMIT 400").fetchall()
    out = []
    for t in rows:
        due = engine.ticket_due(t, cfg)
        tier = engine.tier_of(conn, t["customer_id"], now)
        out.append({"id": t["id"], "customer_id": t["customer_id"], "name": t["name"], "tier": tier,
                    "order_id": t["order_id"], "created_at": t["created_at"], "text": t["text"],
                    "category": t["category"], "urgency": t["urgency"], "sentiment": t["sentiment"],
                    "root_cause": t["root_cause"], "suggested_reply": t["suggested_reply"], "triage_source": t["triage_source"],
                    "due_at": iso(due), "replied": bool(t["first_reply_at"]),
                    "overdue": (not t["first_reply_at"]) and due < now,
                    "hours_left": round((due - now).total_seconds() / 3600, 1) if not t["first_reply_at"] else None})
    # Priority layer: unanswered first, then overdue, tier, urgency, oldest
    out.sort(key=lambda x: (x["replied"], not x["overdue"], TIER_RANK[x["tier"]], -(x["urgency"] or 0), x["created_at"]))
    return out[:limit]


# ---------------------------------------------------------- customers

def customer_detail(conn, now, cid):
    c = conn.execute("SELECT * FROM customers WHERE id=?", (cid,)).fetchone()
    if not c:
        raise KeyError("customer not found")
    orders = [dict(o) for o in conn.execute("SELECT * FROM orders WHERE customer_id=? ORDER BY placed_at DESC LIMIT 10", (cid,))]
    credits = [dict(x) for x in conn.execute("SELECT * FROM credits WHERE customer_id=? ORDER BY ts DESC LIMIT 10", (cid,))]
    tickets = [dict(x) for x in conn.execute("SELECT id,created_at,category,first_reply_at,resolved_at,text FROM tickets WHERE customer_id=? ORDER BY created_at DESC LIMIT 5", (cid,))]
    last = conn.execute("SELECT * FROM orders WHERE customer_id=? ORDER BY placed_at DESC LIMIT 1", (cid,)).fetchone()
    tier = engine.tier_of(conn, cid, now)
    n12 = conn.execute("SELECT COUNT(*) n FROM orders WHERE customer_id=? AND placed_at>=?", (cid, iso(add_days(now, -365)))).fetchone()["n"]
    nxt = {"Member": ("Regular", 3), "Regular": ("Priority", 6)}.get(tier)
    return {"customer": dict(c), "tier": tier, "orders_12m": n12,
            "next_tier": {"name": nxt[0], "orders_needed": max(0, nxt[1] - n12)} if nxt else None,
            "orders": orders, "credits": credits, "tickets": tickets,
            "credits_total_90d": round(sum(x["amount"] for x in credits if x["ts"] >= iso(add_days(now, -90))), 2),
            "reorder_from": dict(last) if last else None}


def list_customers(conn, q="", limit=30):
    like = f"%{q}%"
    return [dict(r) for r in conn.execute("SELECT id,name,region,grp FROM customers WHERE id LIKE ? OR name LIKE ? ORDER BY id LIMIT ?", (like, like, limit))]


def draft_for_customer(conn, now, cid):
    c = conn.execute("SELECT * FROM customers WHERE id=?", (cid,)).fetchone()
    if not c:
        raise KeyError("customer not found")
    cfg = engine.get_config(conn)
    cust, orders, tickets = model.load(conn)
    f = model.features(orders.get(cid, []), tickets.get(cid, []), now, cfg["reply"]["hours"]) or {}
    credit = conn.execute("SELECT COALESCE(SUM(amount),0) s FROM credits WHERE customer_id=? AND ts>=?", (cid, iso(add_days(now, -30)))).fetchone()["s"]
    facts = {"customer_ref": cid, "late": bool(f.get("late_180")), "price_surprise": bool(f.get("price_surprise_180")),
             "open_tickets": int(f.get("open_tickets", 0)), "credit": round(credit, 2) if credit else 0,
             "days_since_last_order": int(f.get("recency_days", 0)), "contact": "a named member of our team"}
    r = ai.draft_recovery(conn, now, facts)
    return dict(r, facts=facts, customer=c["name"])


# --------------------------------------------------------- CSV import

def _d(s):
    s = (s or "").strip()
    if not s:
        return None
    if len(s) == 10:
        s += "T09:00:00"
    return parse(s.replace(" ", "T"))


def import_orders(conn, now, text):
    rd = csv.DictReader(io.StringIO(text))
    need = {"order_id", "customer_id", "placed_at", "promised_at", "quoted_total"}
    if not rd.fieldnames or not need <= set(rd.fieldnames):
        raise ValueError("CSV needs columns: " + ", ".join(sorted(need)) + " (optional: customer_name, region, final_total, discount, delivered_at)")
    ok, bad, ids = 0, [], []
    for i, r in enumerate(rd, start=2):
        try:
            oid, cid = r["order_id"].strip(), r["customer_id"].strip()
            placed, promised = _d(r["placed_at"]), _d(r["promised_at"])
            if not (oid and cid and placed and promised):
                raise ValueError("missing value")
            if promised < placed:
                raise ValueError("promised_at before placed_at")
            quoted = float(r["quoted_total"])
            if conn.execute("SELECT 1 FROM orders WHERE id=?", (oid,)).fetchone():
                raise ValueError("duplicate order_id")
            region = (r.get("region") or "Unknown").strip() or "Unknown"
            if not conn.execute("SELECT 1 FROM customers WHERE id=?", (cid,)).fetchone():
                conn.execute("INSERT INTO customers(id,name,email,region,joined_at,grp) VALUES(?,?,?,?,?,'kept')",
                             (cid, (r.get("customer_name") or cid).strip(), "", region, iso(placed)))
            delivered = _d(r.get("delivered_at"))
            final = float(r["final_total"]) if (r.get("final_total") or "").strip() else (quoted if delivered else None)
            conn.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (oid, cid, region, iso(placed), round(days(placed, promised), 2), iso(promised), quoted, final,
                          float(r.get("discount") or 0), iso(delivered) if delivered else None))
            ids.append(oid); ok += 1
        except (ValueError, KeyError) as e:
            bad.append({"row": i, "error": str(e)})
    conn.commit()
    replay(conn, ids, [], now)
    return {"imported": ok, "skipped": bad[:20], "skipped_count": len(bad)}


def import_tickets(conn, now, text):
    rd = csv.DictReader(io.StringIO(text))
    need = {"customer_id", "created_at", "text"}
    if not rd.fieldnames or not need <= set(rd.fieldnames):
        raise ValueError("CSV needs columns: " + ", ".join(sorted(need)) + " (optional: order_id, first_reply_at, resolved_at)")
    ok, bad, ids = 0, [], []
    for i, r in enumerate(rd, start=2):
        try:
            cid = r["customer_id"].strip()
            if not conn.execute("SELECT 1 FROM customers WHERE id=?", (cid,)).fetchone():
                raise ValueError("unknown customer_id (import orders first)")
            created = _d(r["created_at"])
            if not created or not (r["text"] or "").strip():
                raise ValueError("missing created_at or text")
            fr, rs = _d(r.get("first_reply_at")), _d(r.get("resolved_at"))
            tr = ai.triage_fallback(r["text"])      # bulk history uses the rule-based triage; live tickets use Claude
            cur = conn.execute("INSERT INTO tickets(customer_id,order_id,created_at,text,category,urgency,sentiment,root_cause,suggested_reply,triage_source,first_reply_at,resolved_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                               (cid, (r.get("order_id") or "").strip() or None, iso(created), r["text"][:4000], tr["category"], tr["urgency"],
                                tr["sentiment"], tr["root_cause"], tr["suggested_reply"], "fallback",
                                iso(fr) if fr else None, iso(rs) if rs else None))
            ids.append(cur.lastrowid); ok += 1
        except (ValueError, KeyError) as e:
            bad.append({"row": i, "error": str(e)})
    conn.commit()
    replay(conn, [], ids, now)
    return {"imported": ok, "skipped": bad[:20], "skipped_count": len(bad)}


# ------------------------------------------------------------- summary

def region_kept(conn, now, window=90):
    since = iso(add_days(now, -window))
    agg = {}
    for r in conn.execute("SELECT kind, payload FROM ledger WHERE ts>=? AND kind IN ('promise_kept','promise_broken')", (since,)):
        p = json.loads(r["payload"])
        if p.get("promise") != "delivery" or p.get("holdout") or not p.get("region"):
            continue
        a = agg.setdefault(p["region"], [0, 0])
        a[0 if r["kind"] == "promise_kept" else 1] += 1
    return {k: {"kept": v[0], "broken": v[1], "kept_rate": round(v[0] / (v[0] + v[1]), 3)} for k, v in agg.items() if v[0] + v[1]}


def summary(conn, now):
    engine.sweep(conn, now)
    radar = model.score_all(conn, now, limit=5)
    counts = {t: conn.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"] for t in ("customers", "orders", "tickets", "credits")}
    return {"now": iso(now), "counts": counts, "scorecard": engine.scorecard(conn, now), "retention": engine.retention(conn, now),
            "operations": engine.operations_metrics(conn, now), "regions": region_kept(conn, now),
            "radar": {"scored": radar.get("scored", 0), "total_value_at_risk": radar.get("total_value_at_risk", 0),
                      "model": radar.get("model")},
            "ai": ai.status()}


def ask(conn, now, question):
    s = summary(conn, now)
    body = model.latest(conn)
    drivers = []
    if body:
        pairs = sorted(zip(model.NAMES, body["coef"]), key=lambda p: -p[1])
        drivers = [{"signal": model.LABEL[n], "weight": round(c, 2)} for n, c in pairs if c > 0.05][:4]
    cats = {}
    for r in conn.execute("SELECT category, COUNT(*) n FROM tickets WHERE created_at>=? GROUP BY category", (iso(add_days(now, -90)),)):
        cats[r["category"] or "untriaged"] = r["n"]
    reg = s["regions"]
    worst = sorted(reg.items(), key=lambda kv: kv[1]["kept_rate"])[:2]
    facts = {"promise_kept_rate_90d": s["scorecard"]["overall_kept_rate"], "promises": {p["promise"]: p["kept_rate"] for p in s["scorecard"]["promises"]},
             "delivery_kept_rate_by_region": {k: v["kept_rate"] for k, v in reg.items()}, "ticket_categories_90d": cats,
             "churn_signal_weights": drivers, "value_at_risk_rand": s["radar"]["total_value_at_risk"],
             "discount_free_retention": {g: v["discount_free_retention"] for g, v in s["retention"]["groups"].items()},
             "avg_first_reply_hours": s["operations"]["avg_first_reply_hours"], "repeat_contact_rate": s["operations"]["repeat_contact_rate"]}
    lines = []
    if drivers:
        lines.append("The signals the model weights most for lost customers are: " + "; ".join(d["signal"] for d in drivers) + ".")
    if worst:
        lines.append("Delivery promises are kept least in " + " and ".join(f"{k} ({v['kept_rate']:.0%})" for k, v in worst) + ".")
    if cats:
        top = sorted(cats.items(), key=lambda kv: -kv[1])[0]
        lines.append(f"The most common support topic in the last 90 days is {top[0].replace('_', ' ')} ({top[1]} tickets).")
    lines.append("This is a pattern in your own data, not proof of cause; the holdout comparison is what tests cause.")
    r = ai.ask(conn, now, question, facts, " ".join(lines))
    r["facts_used"] = facts
    return r
