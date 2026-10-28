"""Database and time helpers. Standard library only."""
import os
import sqlite3
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get(
    "KEPT_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kept.db")
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers(
  id TEXT PRIMARY KEY, name TEXT, email TEXT, region TEXT, joined_at TEXT,
  grp TEXT NOT NULL DEFAULT 'kept');          -- 'kept' or 'holdout'
CREATE TABLE IF NOT EXISTS orders(
  id TEXT PRIMARY KEY, customer_id TEXT, region TEXT, placed_at TEXT,
  promised_days REAL, promised_at TEXT, quoted_total REAL, final_total REAL,
  discount REAL DEFAULT 0, delivered_at TEXT);
CREATE TABLE IF NOT EXISTS tickets(
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, order_id TEXT,
  created_at TEXT, text TEXT, category TEXT, urgency INTEGER, sentiment TEXT,
  root_cause TEXT, suggested_reply TEXT, triage_source TEXT,
  first_reply_at TEXT, resolved_at TEXT);
CREATE TABLE IF NOT EXISTS ledger(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL,
  ref TEXT, payload TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations(
  promise TEXT NOT NULL, ref TEXT NOT NULL, PRIMARY KEY(promise, ref));
CREATE TABLE IF NOT EXISTS credits(
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, order_id TEXT,
  promise TEXT, amount REAL, ts TEXT, ledger_seq INTEGER);
CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS ai_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, feature TEXT, source TEXT,
  prompt_redacted TEXT, output TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS model(id INTEGER PRIMARY KEY AUTOINCREMENT, trained_at TEXT, body TEXT);
CREATE INDEX IF NOT EXISTS ix_orders_c ON orders(customer_id);
CREATE INDEX IF NOT EXISTS ix_tickets_c ON tickets(customer_id);
CREATE INDEX IF NOT EXISTS ix_credits_c ON credits(customer_id);
"""

FMT = "%Y-%m-%dT%H:%M:%S"


def connect(path=None):
    path = path or DB_PATH
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def iso(dt):
    return dt.strftime(FMT)


def parse(s):
    return datetime.strptime(s[:19], FMT) if s else None


def days(a, b):
    """Days from a to b (float)."""
    return (b - a).total_seconds() / 86400.0


def add_days(dt, d):
    return dt + timedelta(days=d)
