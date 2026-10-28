"""Synthetic demo data. NOT real customers.

A small simulated shop is run for 12 months. Customers buy, orders are delivered late or with surprise
fees, people contact support and wait. Friction raises each customer's chance of lapsing, and a credit
softens it for customers in the Kept group. THAT RELATIONSHIP IS BUILT INTO THE GENERATOR (see
FRICTION_BITE and CREDIT_RELIEF), so results on this data demonstrate the pipeline, not real-world effect.

Everything is then replayed through the real engine, so the ledger, credits and scorecard are real
outputs of the real rules on this simulated history.
"""
import math
import random

from . import engine, model, service
from .db import add_days, connect, iso, now as real_now

REGIONS = {  # mean delivery days, spread
    "Gauteng": (2.1, 0.8), "Western Cape": (2.7, 0.9), "KwaZulu-Natal": (3.0, 1.0),
    "Eastern Cape": (4.0, 1.2), "Limpopo": (4.6, 1.3)}
FIRST = ["Thandi", "Sipho", "Lerato", "Pieter", "Aisha", "Johan", "Naledi", "Kabelo", "Zanele", "Michael", "Fatima",
         "Bongani", "Anele", "Riaan", "Nomsa", "David", "Palesa", "Yusuf", "Ayanda", "Elmarie", "Tumelo", "Chloe"]
LAST = ["Nkosi", "Dlamini", "van der Merwe", "Naidoo", "Mokoena", "Botha", "Khumalo", "Pillay", "Sithole", "Smith",
        "Molefe", "Jacobs", "Ndlovu", "Govender", "Mahlangu", "Williams", "Zulu", "Petersen", "Mthembu", "Cele"]
FRICTION_BITE = 0.9      # how strongly friction cuts the chance of buying
CREDIT_RELIEF = 0.7      # how much a credit/proactive recovery softens friction (Kept group only)

TICKET_TEXT = {
    "late_delivery": ["My order was supposed to be here by now and tracking hasn't moved. Where is my order?",
                      "Still waiting for my delivery, it is late again. This is the second time.",
                      "The courier hasn't arrived and nobody told me anything about the delay."],
    "hidden_cost": ["I was charged more than the price at checkout. Where does this extra delivery fee come from?",
                    "The total was higher than what I agreed to. There was a surcharge I never saw.",
                    "Hidden fee added at the end, this is not what the product page said."],
    "support_delay": ["I sent a message two days ago and nobody replied. Still waiting for an answer.",
                      "No response to my earlier question. I am fed up with being ignored."],
    "product_issue": ["The item arrived damaged, please advise.", "Wrong item in the box, need it swapped.",
                      "The product is not working out of the box."],
    "billing": ["I think I was charged twice on my card for the same order.", "Where is my refund? I returned it last week."],
}


def build(path=None, n_customers=700, seed=7, holdout_frac=0.10, now=None):
    conn = connect(path)
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=MEMORY")
    if conn.execute("SELECT COUNT(*) n FROM customers").fetchone()["n"]:
        raise RuntimeError("database already has data; delete data/kept.db to reseed")
    now = now or real_now()
    rnd = random.Random(seed)
    start = add_days(now, -365)
    order_ids, ticket_ids = [], []
    onum = 0
    for i in range(1, n_customers + 1):
        cid = f"C{1000 + i}"
        name = f"{rnd.choice(FIRST)} {rnd.choice(LAST)}"
        region = rnd.choices(list(REGIONS), weights=[40, 22, 18, 12, 8])[0]
        joined = add_days(start, rnd.uniform(0, 270))
        grp = "holdout" if rnd.random() < holdout_frac else "kept"
        disc_prone = rnd.random() < 0.3
        base = rnd.uniform(1 / 70, 1 / 22)
        conn.execute("INSERT INTO customers VALUES(?,?,?,?,?,?)",
                     (cid, name, f"c{1000 + i}@example.invalid", region, iso(joined), grp))
        friction, lapsed = 0.0, False
        day = joined
        first = True
        while day < now and not lapsed:
            p = 1.0 if first else base * (1.5 if disc_prone else 1.0) * math.exp(-FRICTION_BITE * friction)
            if rnd.random() < min(0.95, p):
                onum += 1
                oid = f"O{100000 + onum}"
                total = round(math.exp(rnd.gauss(6.0, 0.5)), 2)          # about R400 typical
                disc = round(total * 0.10, 2) if rnd.random() < (0.8 if disc_prone else 0.12) else 0.0
                quoted = round(total - disc, 2)
                mean, sd = REGIONS[region]
                d = max(0.4, rnd.gauss(mean, sd))
                promised_days = 3.0
                placed = add_days(day, rnd.random())
                delivered = add_days(placed, d)
                surprise = rnd.random() < 0.07
                final = round(quoted * rnd.uniform(1.06, 1.15), 2) if surprise else quoted
                conn.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
                             (oid, cid, region, iso(placed), promised_days, iso(add_days(placed, promised_days)), quoted,
                              final if delivered <= now else None, disc, iso(delivered) if delivered <= now else None))
                order_ids.append(oid)
                late = d > promised_days
                kinds = []
                if late and rnd.random() < 0.5: kinds.append("late_delivery")
                if surprise and rnd.random() < 0.55: kinds.append("hidden_cost")
                if rnd.random() < 0.04: kinds.append(rnd.choice(["product_issue", "billing"]))
                if late: friction += 1.0
                if surprise: friction += 0.8
                for k in kinds:
                    created = add_days(placed, min(d, promised_days + 0.2) if k == "late_delivery" else d + 0.1)
                    if created > now:
                        continue
                    reply_h = math.exp(rnd.gauss(1.4, 0.9))                 # median about 4h, long tail
                    if k in ("billing", "hidden_cost") and rnd.random() < 0.3: reply_h *= 4
                    reply = add_days(created, reply_h / 24)
                    unresolved = rnd.random() < (0.30 if k in ("billing", "hidden_cost") else 0.10)
                    cur = conn.execute(
                        "INSERT INTO tickets(customer_id,order_id,created_at,text,category,urgency,sentiment,root_cause,triage_source,first_reply_at,resolved_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (cid, oid, iso(created), rnd.choice(TICKET_TEXT[k]), k, 2, "frustrated", None, "seed",
                         iso(reply) if reply <= now else None,
                         None if unresolved or reply > now else iso(add_days(reply, rnd.uniform(0.2, 2)))))
                    ticket_ids.append(cur.lastrowid)
                    if reply_h > 4: friction += 0.9
                    if unresolved: friction += 1.0
                if grp == "kept" and (late or surprise):
                    friction = max(0.0, friction - CREDIT_RELIEF * (1.0 if late else 0.8))   # credit / refund softens it
                first = False
            friction *= 0.97
            if rnd.random() < 0.0008 + 0.004 * friction:
                lapsed = True
            day = add_days(day, 1)
    conn.commit()
    service.replay(conn, order_ids, ticket_ids, now)
    model.train(conn, now)
    conn.commit()
    return conn


if __name__ == "__main__":
    import sys
    c = build(sys.argv[1] if len(sys.argv) > 1 else None)
    from .db import now as _n
    print({t: c.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"] for t in ("customers", "orders", "tickets", "ledger", "credits")})
