"""Writes samples/orders.csv and samples/tickets.csv with dates ending yesterday (so 'last 90 days' views work).
Invented people and orders; a small story: Gauteng keeps its promise, Limpopo and the Eastern Cape do not,
a few orders carry a surprise fee, two are overdue and undelivered, and some tickets wait too long."""
import csv, os, random
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
rnd = random.Random(3)
today = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
people = [("S1", "Sipho Dlamini", "Gauteng"), ("S2", "Lerato Mokoena", "Gauteng"), ("S3", "Aisha Naidoo", "Western Cape"),
          ("S4", "Pieter Botha", "Western Cape"), ("S5", "Naledi Khumalo", "Limpopo"), ("S6", "Kabelo Sithole", "Limpopo"),
          ("S7", "Zanele Zulu", "Eastern Cape"), ("S8", "Riaan Jacobs", "Gauteng"), ("S9", "Nomsa Cele", "KwaZulu-Natal"),
          ("S10", "Yusuf Pillay", "Gauteng")]
speed = {"Gauteng": (2.0, .6), "Western Cape": (3.0, 1.0), "Limpopo": (5.5, 1.0), "Eastern Cape": (5.0, 1.0), "KwaZulu-Natal": (3.2, .9)}
orders, tickets = [], []
n = 0
for cid, name, region in people:
    for k in range(rnd.randint(2, 4)):
        n += 1
        placed = today - timedelta(days=rnd.randint(12, 80))
        promised = placed + timedelta(days=3)
        total = round(rnd.uniform(250, 1400), 2)
        d = max(.5, rnd.gauss(*speed[region]))
        delivered = placed + timedelta(days=d)
        final = round(total * rnd.uniform(1.07, 1.14), 2) if rnd.random() < .15 else total
        row = dict(order_id=f"S-{1000 + n}", customer_id=cid, customer_name=name, region=region, placed_at=placed.date(),
                   promised_at=promised.date(), quoted_total=total, final_total=final, delivered_at=delivered.date() if delivered < today else "")
        orders.append(row)
        if d > 3 and rnd.random() < .6:
            c = delivered if delivered < today else promised + timedelta(days=1)
            tickets.append(dict(customer_id=cid, created_at=(promised + timedelta(days=1)).date(), order_id=row["order_id"],
                                text="Where is my order? It is late and tracking has not moved.",
                                first_reply_at=(promised + timedelta(days=1, hours=rnd.choice([2, 3, 20, 30]))).date() if rnd.random() < .7 else "",
                                resolved_at=""))
        if final != total:
            tickets.append(dict(customer_id=cid, created_at=delivered.date(), order_id=row["order_id"],
                                text="I was charged more than the price at checkout. What is this extra fee?",
                                first_reply_at="", resolved_at=""))
# two overdue, still undelivered
for cid, name, region in [("S5", "Naledi Khumalo", "Limpopo"), ("S7", "Zanele Zulu", "Eastern Cape")]:
    n += 1
    placed = today - timedelta(days=6)
    orders.append(dict(order_id=f"S-{1000 + n}", customer_id=cid, customer_name=name, region=region, placed_at=placed.date(),
                       promised_at=(placed + timedelta(days=3)).date(), quoted_total=780.0, final_total="", delivered_at=""))
for name, rows, fields in (("orders.csv", orders, list(orders[0])), ("tickets.csv", tickets, list(tickets[0]))):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"wrote {len(orders)} orders and {len(tickets)} tickets to {HERE}")
