# Samples

Everything here works on invented data and needs no installs.

| File | What it is |
| --- | --- |
| `make_samples.py` | Regenerates `orders.csv` and `tickets.csv` with dates ending yesterday, so the 90-day views work. |
| `orders.csv`, `tickets.csv` | 33 orders and 19 tickets for 10 invented customers. Gauteng keeps its promise; Limpopo and the Eastern Cape do not; some orders carry a surprise fee; two are overdue and undelivered. Import them on the **Data & AI** page. |
| `walkthrough.py` | A 60-second tour in the terminal: imports the CSVs, shows the ledger-based scorecard, honest delivery windows, live ticket triage (including a prompt-injection attempt), automatic credits and a tamper check. Uses a throwaway database. |
| `shop-widget.html` | A stand-in shop page showing the **scorecard badge** and a **checkout delivery promise** powered by the public `/api/scorecard` and `/api/eta` endpoints. This is how a shop plugin or theme snippet would surface Kept to shoppers. |

## Try it with each AI option

```
python3 samples/walkthrough.py                                  # no AI: rule-based fallback
ANTHROPIC_API_KEY=sk-ant-... python3 samples/walkthrough.py     # Claude
KEPT_LLM=ollama python3 samples/walkthrough.py                  # free local model (see main README)
```
Step 5 prints which model wrote each triage result, or "fallback" if a safety check rejected it.

## See the shop widget
Start the main app (`python3 -m app.server`), then open `samples/shop-widget.html` in a browser. Pick a region and watch the promise change with the shop's own history. If you want a populated record first, import the sample CSVs or use the built-in demo data.

The risk radar needs about 60 customers with history to train, so it does not appear on the small sample; use the built-in demo data (700 customers) for that view.
