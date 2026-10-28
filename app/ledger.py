"""Append-only, hash-chained promise ledger.

Every kept/broken promise, credit and rule change is one row. Each row's hash covers the
previous row's hash, so editing or deleting any past row breaks every hash after it and
verify() reports where. The scorecard is computed ONLY from this table.

Limit (stated honestly): whoever controls the database could rebuild the whole chain.
Publishing the head hash somewhere outside the business (a third party, a public page)
is what makes tampering detectable by customers. See README.
"""
import hashlib
import json

GENESIS = "GENESIS"


def _digest(prev, ts, kind, ref, payload_json):
    return hashlib.sha256(f"{prev}|{ts}|{kind}|{ref}|{payload_json}".encode()).hexdigest()


def append(conn, kind, ref, payload, ts):
    row = conn.execute("SELECT hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
    prev = row["hash"] if row else GENESIS
    pj = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    h = _digest(prev, ts, kind, ref or "", pj)
    cur = conn.execute(
        "INSERT INTO ledger(ts,kind,ref,payload,prev_hash,hash) VALUES(?,?,?,?,?,?)",
        (ts, kind, ref or "", pj, prev, h),
    )
    return cur.lastrowid


def verify(conn):
    prev = GENESIS
    count = 0
    for r in conn.execute("SELECT * FROM ledger ORDER BY seq"):
        if r["prev_hash"] != prev or _digest(prev, r["ts"], r["kind"], r["ref"], r["payload"]) != r["hash"]:
            return {"ok": False, "count": count, "broken_at": r["seq"], "head": prev}
        prev = r["hash"]
        count += 1
    return {"ok": True, "count": count, "broken_at": None, "head": prev}
