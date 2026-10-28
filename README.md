# Kept

A working MVP for the **Credify System Thinking Challenge 2026**, challenge area 03, Customer Loyalty:
*How can businesses retain customers when competitors are only one click away?*

Kept makes a business's reliability **checkable and rewarded**, so price is not the only thing a customer can compare.

- **Promise** – a delivery window quoted from the business's own history, an all-in price promise, a reply-time promise. When one breaks, a capped credit is paid automatically.
- **Proof** – every kept or broken promise goes into an append-only, hash-chained ledger. The public scorecard is computed only from that ledger.
- **Priority** – long-term customers earn faster, more personal service (not bigger coupons), and a risk radar shows who is showing friction, why, and what a person should do.

Loyalty is measured **without relying on discounts**: the headline metric is discount-free retention, compared with a holdout group that does not get Kept.

## What is real in this MVP

| Part | What it does | How |
| --- | --- | --- |
| Promise engine | Judges delivery, price and reply promises; pays credits; enforces a per-customer cap; refunds surprise charges in full; holdout customers are measured but never paid | `app/engine.py` |
| Proactive detection | A background sweep finds overdue orders and unanswered tickets before the customer complains | `engine.sweep` |
| Ledger | SHA-256 hash chain; editing or deleting any past row is detected and reported | `app/ledger.py` |
| Honest ETA | Recommends the smallest delivery promise that history says will be kept 90% of the time, per region, with expected credit cost | `engine.quote_eta` |
| Friction model | Logistic regression trained in pure Python on the database's own history, 5-fold cross-validated against a "days since last order" baseline | `app/model.py` |
| AI (Claude) | Ticket triage and reply drafts, proactive recovery messages, and "ask your data" | `app/ai.py` |
| Retention measurement | Discount-free retention, Kept vs holdout, with a confidence interval and a small-sample warning; repeat-contact rate; a credit-cost vs margin check | `engine.retention` |
| Service desk | Queue ordered by overdue, then earned tier, then urgency | `service.list_tickets` |
| CSV import | Bring real orders and tickets; historic data is judged and credited | `service.import_*` |
| Web app + JSON API | Seven views and 20 routes | `app/static/index.html`, `app/server.py` |

### How the AI is kept safe
- Emails, phone numbers and ID/card numbers are redacted before text leaves the server; customers are sent as ids, never names.
- Customer text is passed as untrusted data inside delimiters.
- Output must be valid JSON with categories from a fixed list; links are rejected; any number in free text must appear in the facts the system supplied; recovery messages may not offer discounts.
- The AI **never moves money or writes the ledger**. Credits come only from the rules engine.
- If there is no API key, the call fails, or the output fails a check, a rule-based fallback answers and the screen says "rule-based fallback" instead of "written by Claude".

## Run it (no installs)

Needs Python 3.9+ only. No packages, no build step.

```
python3 -m app.server
```

Open http://127.0.0.1:8000. On first start it generates synthetic demo data (about 3 seconds).

To use Claude, set a key before starting:

```
export ANTHROPIC_API_KEY=sk-ant-...        # optional; model via KEPT_MODEL (default claude-sonnet-5-5)
python3 -m app.server
```

Other settings: `KEPT_HOST`, `KEPT_PORT`, `KEPT_DB` (database path), `KEPT_TOKEN` (if set, every API route except the public scorecard needs the header `X-Kept-Token`; the web app asks for it).

Tests (26, standard library): `python3 -W ignore -m unittest discover -s tests`

To use your own data instead of the demo: `KEPT_DB=data/mine.db KEPT_NO_SEED=1 python3 -m app.server`, then use **Data & AI** to import orders first, then tickets (formats are shown on that page).

## Suggested demo (5 minutes)
1. **Overview** – promise record, delivery kept by region (Gauteng 90%, Limpopo 14% under a flat 3-day promise).
2. **Promises** – pick Limpopo: the honest window is longer. Place an order, deliver it 6 days later, watch the credit appear.
3. **Service desk** – paste an angry ticket; see triage, root cause, drafted reply and its source label. Try a prompt-injection ticket.
4. **Risk radar** – reasons and next actions, plus the honest model comparison.
5. **Public scorecard** – verify the ledger chain.

## Known limitations (read before judging)
- **The demo data is synthetic.** 700 invented customers simulated over 12 months. The effect of friction on lapsing and of credits on recovery is **built into the generator** (`app/seed.py`), so the retention gap and the model results show the pipeline working, **not** that Kept raises retention. The holdout in the demo is small (about 43 customers) and the app says so.
- **On the synthetic data the friction model barely beats recency alone** (AUC 0.640 vs 0.635 among customers who still looked healthy). We report it as is. Whether friction signals flag churn earlier than recency is exactly what a pilot on real data must test.
- **The Claude path was tested against a local stand-in for the Anthropic API, not the live service** (no API key was available while building). The request format follows the public Messages API; check it with your key. Without a key the rule-based fallback is used throughout.
- **The ledger proves tampering only if its head hash is published outside the business.** Someone who controls the database could rebuild the whole chain. The scorecard shows the head hash for this purpose; third-party anchoring is not built.
- **No integrations** with Shopify, couriers or helpdesks, no user accounts or roles (only an optional shared token), single-process SQLite. Not production-hardened.
- **No customer-facing messaging**: drafts are shown to staff, nothing is sent.
- **No legal or POPIA review.** Personal data handling here (redaction, no customer-visible scores) is a design intention, not compliance.
- Ledger timestamps for imported history are the event times, so they are not strictly increasing; the chain order is the sequence number.
- The holdout design withholds Kept from a few customers to measure its effect. A real deployment should decide the ethics and size of that group, or use a staggered rollout instead.

## Documentation
Problem, stakeholders, root causes, interventions compared, evidence, feasibility, risks and sources are in the project documentation and pitch deck submitted with this repository.

## Tools and AI assistance
Built with Claude (Anthropic) in a chat session: research, system analysis, all code in this repository and the drafting of the documentation were produced with Claude and reviewed by the team. The runtime AI features call Claude when a key is configured. No third-party code or packages are used (Python standard library only); fonts load from Google Fonts.

**Team contribution:** [add your name and what you personally decided, researched, built and tested]
