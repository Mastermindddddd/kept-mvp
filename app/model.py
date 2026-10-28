"""Friction-risk model: logistic regression trained in pure Python on the business's own history.

Question it answers: "given what has happened to this customer so far, how likely are they to
place NO order in the next 90 days?" It is trained on a snapshot 90 days ago (features use only
data from before the snapshot, labels come from the 90 days after), and is evaluated on customers
it never saw, against a recency-only baseline. That comparison is the honest test of the claim
that friction signals flag churn earlier than recency alone.
"""
import json
import math
import random

from . import engine
from .db import add_days, days, iso, parse

FEATURES = [
    ("recency_days", "Days since last order"),
    ("orders_180", "Orders in last 180 days"),
    ("late_180", "Late deliveries (180d)"),
    ("price_surprise_180", "Price surprises (180d)"),
    ("tickets_60", "Support contacts (60d)"),
    ("open_tickets", "Unresolved tickets"),
    ("slow_reply_180", "Slow or missing replies (180d)"),
    ("discount_share_180", "Share of orders bought with a discount"),
    ("last_order_late", "Last order was late"),
]
NAMES = [f[0] for f in FEATURES]
LABEL = dict(FEATURES)


def load(conn):
    cust, orders, tickets = {}, {}, {}
    for c in conn.execute("SELECT * FROM customers"):
        cust[c["id"]] = dict(c)
    for o in conn.execute("SELECT * FROM orders ORDER BY placed_at"):
        orders.setdefault(o["customer_id"], []).append({
            "placed": parse(o["placed_at"]), "promised": parse(o["promised_at"]), "delivered": parse(o["delivered_at"]),
            "quoted": o["quoted_total"], "final": o["final_total"], "discount": o["discount"] or 0, "total": o["quoted_total"]})
    for t in conn.execute("SELECT * FROM tickets ORDER BY created_at"):
        tickets.setdefault(t["customer_id"], []).append({
            "created": parse(t["created_at"]), "reply": parse(t["first_reply_at"]), "resolved": parse(t["resolved_at"])})
    return cust, orders, tickets


def _late(o, T):
    if o["delivered"] and o["delivered"] <= T:
        return o["delivered"] > o["promised"]
    return o["promised"] < T


def features(orders, tickets, T, reply_hours):
    os_ = [o for o in orders if o["placed"] < T]
    if not os_:
        return None
    last = os_[-1]
    w180 = [o for o in os_ if days(o["placed"], T) <= 180]
    t_all = [t for t in tickets if t["created"] < T]
    def slow(t):
        due = add_days(t["created"], reply_hours / 24.0)
        return (t["reply"] > due) if (t["reply"] and t["reply"] <= T) else due < T
    return {
        "recency_days": days(last["placed"], T),
        "orders_180": float(len(w180)),
        "late_180": float(sum(1 for o in w180 if _late(o, T))),
        "price_surprise_180": float(sum(1 for o in w180 if o["delivered"] and o["delivered"] <= T and o["final"] is not None and o["final"] > o["quoted"] + 0.005)),
        "tickets_60": float(sum(1 for t in t_all if days(t["created"], T) <= 60)),
        "open_tickets": float(sum(1 for t in t_all if not (t["resolved"] and t["resolved"] <= T))),
        "slow_reply_180": float(sum(1 for t in t_all if days(t["created"], T) <= 180 and slow(t))),
        "discount_share_180": (sum(1 for o in w180 if o["discount"] > 0) / len(w180)) if w180 else 0.0,
        "last_order_late": 1.0 if _late(last, T) else 0.0,
    }


def _sigmoid(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def _auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def _lift(scores, labels, frac=0.2):
    k = max(1, int(len(scores) * frac))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    base = sum(labels) / len(labels)
    top = sum(labels[i] for i in order) / k
    return {"top_share": frac, "churn_rate_in_top": round(top, 4), "churn_rate_overall": round(base, 4),
            "lift": round(top / base, 3) if base else None}


def train(conn, now, seed=11):
    cfg = engine.get_config(conn)
    cust, orders, tickets = load(conn)
    T = add_days(now, -90)
    X, y, ids = [], [], []
    for cid, os_ in orders.items():
        f = features(os_, tickets.get(cid, []), T, cfg["reply"]["hours"])
        if f is None:
            continue
        X.append([f[n] for n in NAMES])
        y.append(0 if any(T < o["placed"] <= now for o in os_) else 1)   # 1 = no order in the next 90 days
        ids.append(cid)
    if len(X) < 60 or sum(y) < 10 or sum(y) > len(y) - 10:
        raise ValueError("not enough history to train (need ~60+ customers with both churned and retained examples)")
    rnd = random.Random(seed)

    def fit(sel):
        mean = [sum(X[i][j] for i in sel) / len(sel) for j in range(len(NAMES))]
        std = [max(1e-9, math.sqrt(sum((X[i][j] - mean[j]) ** 2 for i in sel) / len(sel))) for j in range(len(NAMES))]
        Z = {i: [(X[i][j] - mean[j]) / std[j] for j in range(len(NAMES))] for i in range(len(X))}
        w, b = [0.0] * len(NAMES), 0.0
        lr, l2, n = 0.3, 0.01, len(sel)
        for _ in range(500):
            gw, gb = [0.0] * len(w), 0.0
            for i in sel:
                e = _sigmoid(b + sum(w[j] * Z[i][j] for j in range(len(w)))) - y[i]
                for j in range(len(w)):
                    gw[j] += e * Z[i][j]
                gb += e
            w = [w[j] - lr * (gw[j] / n + l2 * w[j]) for j in range(len(w))]
            b -= lr * gb / n
        return w, b, mean, std, Z

    # 5-fold cross-validation: every customer is scored by a model that never saw them.
    idx = list(range(len(X)))
    rnd.shuffle(idx)
    folds = [idx[k::5] for k in range(5)]
    oof = {}
    for k in range(5):
        trn = [i for f in range(5) if f != k for i in folds[f]]
        w, b, mean, std, Z = fit(trn)
        for i in folds[k]:
            oof[i] = _sigmoid(b + sum(w[j] * Z[i][j] for j in range(len(w))))

    def evaluate(sel):
        yy = [y[i] for i in sel]
        if len(yy) < 30 or sum(yy) < 5 or sum(yy) > len(yy) - 5:
            return {"n": len(yy), "note": "too few examples to evaluate"}
        sc = [oof[i] for i in sel]
        rec = [X[i][0] for i in sel]          # baseline: just "how long since they last bought"
        return {"n": len(yy), "churn_rate": round(sum(yy) / len(yy), 4), "auc_model": round(_auc(sc, yy), 4),
                "auc_recency_only": round(_auc(rec, yy), 4), "lift_model": _lift(sc, yy), "lift_recency_only": _lift(rec, yy)}
    # The early-warning question: among customers who still LOOK healthy (bought in the last 60 days), who lapses?
    active = [i for i in idx if X[i][0] <= 60]
    w, b, mean, std, _ = fit(idx)               # final model uses everyone
    body = {
        "features": NAMES, "mean": mean, "std": std, "coef": w, "intercept": b,
        "trained_at": iso(now), "snapshot": iso(T), "n_train": len(idx), "n_test": len(idx), "method": "5-fold cross-validation",
        "churn_rate": round(sum(y) / len(y), 4),
        "test_all": evaluate(idx), "test_active": evaluate(active),
        "note": "Trained and tested on this database's own history (5-fold cross-validation). On the bundled synthetic data the "
                "relationship is built into the generator, so this demonstrates the pipeline, not real-world accuracy.",
    }
    conn.execute("INSERT INTO model(trained_at, body) VALUES(?,?)", (iso(now), json.dumps(body)))
    conn.commit()
    return body


def latest(conn):
    r = conn.execute("SELECT body FROM model ORDER BY id DESC LIMIT 1").fetchone()
    return json.loads(r["body"]) if r else None


REASON = {
    "recency_days": lambda v: f"{int(v)} days since last order",
    "orders_180": lambda v: f"only {int(v)} orders in 180 days" if v <= 2 else f"{int(v)} orders in 180 days",
    "late_180": lambda v: f"{int(v)} late deliver{'y' if v == 1 else 'ies'} in 180 days",
    "price_surprise_180": lambda v: f"{int(v)} price surprise{'s' if v != 1 else ''} at delivery",
    "tickets_60": lambda v: f"{int(v)} support contact{'s' if v != 1 else ''} in 60 days",
    "open_tickets": lambda v: f"{int(v)} unresolved ticket{'s' if v != 1 else ''}",
    "slow_reply_180": lambda v: f"{int(v)} slow or missing repl{'y' if v == 1 else 'ies'}",
    "discount_share_180": lambda v: f"{int(v * 100)}% of orders bought on discount",
    "last_order_late": lambda v: "most recent order was late",
}
ACTION = {
    "last_order_late": "Proactive apology with the credit already applied and a named contact.",
    "late_180": "Proactive apology with the credit already applied and a named contact.",
    "price_surprise_180": "Refund the difference now and explain what changed. Do not send a coupon.",
    "open_tickets": "A named person closes the open ticket today and tells the customer.",
    "slow_reply_180": "A named person calls or messages personally; offer a Priority-queue trial.",
    "tickets_60": "A named person checks the history and reaches out once.",
    "discount_share_180": "Do not send another coupon. Offer a service benefit (priority queue, free reorder).",
    "recency_days": "Reorder reminder built from their last order. No discount.",
    "orders_180": "Reorder reminder built from their last order. No discount.",
}


def score_all(conn, now, limit=50):
    body = latest(conn)
    if not body:
        return {"model": None, "customers": [], "scored": 0, "total_value_at_risk": 0}
    cfg = engine.get_config(conn)
    cust, orders, tickets = load(conn)
    out = []
    for cid, os_ in orders.items():
        last = os_[-1]["placed"]
        if days(last, now) > 240:                  # already lost; the radar is for customers still savable
            continue
        f = features(os_, tickets.get(cid, []), now, cfg["reply"]["hours"])
        if f is None:
            continue
        z = [(f[n] - body["mean"][j]) / body["std"][j] for j, n in enumerate(NAMES)]
        contrib = [(NAMES[j], body["coef"][j] * z[j]) for j in range(len(NAMES))]
        risk = _sigmoid(body["intercept"] + sum(c for _, c in contrib))
        top = sorted([c for c in contrib if c[1] > 0.05], key=lambda c: -c[1])[:3]
        spend = sum(o["final"] if o["final"] is not None else o["quoted"] for o in os_ if days(o["placed"], now) <= 365)
        reasons = [{"feature": n, "label": LABEL[n], "text": REASON[n](f[n]), "weight": round(c, 3)} for n, c in top]
        action_key = next((n for n, _ in top if n != "recency_days"), top[0][0] if top else None)
        out.append({"customer_id": cid, "name": cust[cid]["name"], "grp": cust[cid]["grp"], "risk": round(risk, 4),
                    "annual_spend": round(spend, 2), "value_at_risk": round(risk * spend, 2),
                    "tier": engine.tier_of(conn, cid, now), "days_since_order": int(days(last, now)),
                    "reasons": reasons, "action": ACTION.get(action_key, "Review the account.") if top else "No action needed."})
    out.sort(key=lambda r: -r["value_at_risk"])
    total_var = round(sum(r["value_at_risk"] for r in out), 2)
    return {"model": {k: body[k] for k in ("trained_at", "n_train", "n_test", "test_all", "test_active", "note", "churn_rate")},
            "customers": out[:limit], "scored": len(out), "total_value_at_risk": total_var}
