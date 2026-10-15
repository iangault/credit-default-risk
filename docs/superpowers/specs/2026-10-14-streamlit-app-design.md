# Streamlit App: Design Spec

Date: 2026-10-14

## Purpose

A small web app that lets a visitor try the final model, linked from the
author's portfolio site (iangault.io, Machine Learning projects) and the
README. The GitHub repo is mostly text and files; the app gives a recruiter or
hiring manager something to click.

Success: a non-technical visitor understands the main idea within a minute
or two. The model ranks clients by risk, and the cost of each kind of
mistake decides where to draw the line. Being able to score any made-up
client matters less than that.

This is piece A of three:

- **A. The app** (this spec): code, tests, local run.
- **B. Deploy:** the author creates the app on Streamlit Community Cloud
  from this repo; README gets the link and a short GIF.
- **C. Website page:** `projects/credit-default-risk/index.qmd` in the site
  repo, built like the Product Search RAG page, including its wake-up
  notice.

B and C are out of scope here.

## Decisions made (2026-10-14)

| Question | Choice |
|---|---|
| Where it runs | Streamlit Community Cloud free tier, like the RAG app. The site's wake-up notice covers the slow first load. |
| How the model gets there | `artifacts/model.joblib` and `model.json` are committed (done in `96c7f5d`). |
| Repo visibility | Public (done). |
| How much a visitor can change | Pick a real test client, then adjust a few things (not a full form). |
| Edits | Coherent scenarios that rewrite related columns together, not single-column edits. Credit limit and demographics change alone, since nothing else depends on them. |
| Demographics | A collapsed "Fairness check" section. |
| Cost ratio | One sidebar slider shared by both views. |
| First screen | A short About tab, then the two views. |
| Code structure | Logic in `src/credit_risk/showcase.py` (tested); `app/app.py` only does layout. |

## Layout

### Sidebar (shared by all tabs)

- Slider: "A missed defaulter costs N times a false alarm", 1 to 20, step 1,
  default 5.
- Below it: "Flag clients with a default risk of X% or more", where
  X = 1 / (1 + N).
- A short note: the app uses this formula at every ratio. The saved model's
  cutoff at 5x came from a search on training data and is 0.170, so counts
  at 5x differ slightly from `docs/results.md`.

### Tab 1: About

Four or five plain sentences:

- The problem: predict which credit card clients will miss next month's
  payment.
- The data: 30,000 real, anonymized clients of a Taiwanese bank (2005),
  published by UCI.
- The key idea: a missed defaulter costs more than a false alarm, so the
  cutoff is not 0.5.
- The headline result on 6,000 clients the model never saw: 77% of
  defaulters caught, and cost per client 29% lower than with a 0.5 cutoff.

Links to the GitHub repo and `docs/results.md`.

### Tab 2: Try a client

**Pick a client.** Four examples plus a random button:

| Example | Rule |
|---|---|
| A defaulter the model caught | Defaulted, probability at or above the saved cutoff |
| A defaulter the model missed | Defaulted, probability below the cutoff |
| A false alarm | Paid, probability at or above the cutoff |
| A low-risk client, correctly cleared | Paid, probability below the cutoff |

In each group the example is the client whose probability is nearest the
group's median (the rule `pick_clients` in `explain.py` uses for two of
these groups). Examples are chosen once, with the saved model's cutoff
(0.170), so they stay the same person when the slider moves. The random
button picks any test client.

**Client summary.** Their details in plain words (for example "Credit limit:
50,000 NT$", "Most recent payment: 2 months late") and whether they actually
defaulted.

**Controls.**

- Payment behavior (dropdown), which rewrites the six months together:

  | Option | Statuses (PAY_1 to PAY_6) | Payments (PAY_AMT1 to PAY_AMT6) |
  |---|---|---|
  | As recorded | unchanged | unchanged |
  | Pays in full every month | all -1 (paid in full) | PAY_AMTi = BILL_AMT(i+1) for i = 1 to 5; PAY_AMT6 unchanged |
  | Pays the minimum every month | all 0 (paid the minimum) | PAY_AMTi = 6% of BILL_AMT(i+1) for i = 1 to 5; PAY_AMT6 unchanged |
  | Fell 2 months behind | PAY_1 and PAY_2 set to 2; others unchanged | PAY_AMT1 = 4% of BILL_AMT2, PAY_AMT2 = 4% of BILL_AMT3; others unchanged |

  A payment is never below 0: a negative bill (credit balance) gives a
  payment of 0. Bills stay as recorded.

  Where the numbers come from (training clients only): 80% of clients who
  paid in full paid at least the previous bill; the median payment of
  clients paying the minimum is 6% of the previous bill; for clients 2
  months late in both of the last two months, the median is about 4%.
  "2 and 2" is the most common history for clients 2 months late now.

- Credit limit (slider), from the training minimum (10,000) to the training
  99th percentile (500,000). A client whose limit is above that range shows
  their real limit as the slider's maximum.

- **Fairness check: change only demographics** (collapsed by default): sex,
  marital status, education (dropdowns, cleaned categories) and age (slider,
  training minimum to maximum).

- A "Reset to original" button.

- One-line note under the controls: "Scenarios rewrite all six months using
  patterns common in the data; bills stay as recorded."

**Result.**

- Probability of default, shown large.
- Decision at the sidebar's cutoff: "Flag for review" or "Don't flag".
- When any control differs from the original: "Your changes moved this
  client's risk from A% to B%."
- Bar chart (Altair) of the 8 model columns that moved this client's score
  most, in plain words, colored by direction ("raises risk" / "lowers
  risk"). Not SHAP's waterfall plot, which is in log-odds.

**Fairness note** (inside the collapsed section):

- The model was trained with these columns, so it can use them. `SEX_2`
  (female) ranks 10th of 28 by mean SHAP size and lowers scores slightly.
- In many places a lender could not legally base a decision on some of
  them.
- Changing one client is a spot check, not a fairness audit. The proper
  checks (error rates by group; the model with and without these columns)
  are listed in `docs/results.md` as future work.

### Tab 3: Cost ratio

- Counts on the 6,000 test clients at the current cutoff: defaulters caught
  (of 1,327), defaulters missed, false alarms (of 4,673), cost per client.
  Shown beside two comparisons: cutoff 0.5, and flag nobody. Cost per client
  uses the slider's ratio for all three.
- The ROC and precision-recall figure (`plot_tradeoff_curves`) with one
  marker at the current cutoff.
- One sentence: the probabilities do not change with the slider; only the
  line drawn through them does.

## Data flow

**Startup**, once per app process, cached with `st.cache_resource`:

1. Load `configs/base.yaml`.
2. `load_model(cfg)`: loads the committed model and refuses one whose
   `model.json` does not match the config.
3. Rebuild the split in memory: `verify_raw`, `load_raw`, `clean`, `split`
   with the config's seed. This gives the same 6,000 test clients as
   `make evaluate`. Nothing is written to disk.
4. Score all test clients once (`predict_proba`).
5. Measure the slider ranges from the training split.
6. Pick the four example clients.

**Each interaction:**

- Slider: cutoff = 1 / (1 + N). The cost tab calls `decision_metrics` (from
  `evaluate.py`) for the current cutoff, 0.5, and flag nobody.
- Try a client: the selected test row and the control values live in
  `st.session_state`. The edited row is a copy of the original with the
  scenario, limit and demographics applied. It is scored with
  `predict_proba`, so it goes through the full saved pipeline, including
  calibration. Contributions come from `shap_explanation` (from
  `explain.py`) on that one row.

The app only uses test clients. Training clients are used for slider ranges
and nothing else is shown from them.

## Code

### `src/credit_risk/showcase.py` (new, no Streamlit imports)

| Name | What it does |
|---|---|
| `Assets` | Dataclass: model, cfg, test features, test labels, test probabilities, slider ranges, example rows. |
| `load_assets(cfg)` | Startup steps 2 to 6. |
| `cutoff_for(ratio)` | 1 / (1 + ratio). |
| `example_clients(y, proba, threshold)` | Row numbers for the four examples (median rule). |
| `SCENARIOS` | The four payment behavior options. |
| `apply_scenario(row, name)` | Returns a new row with the scenario applied; the input is not changed. |
| `edit_client(row, scenario, limit, demographics)` | Applies the scenario, limit and demographic values; returns a new row. |
| `score_client(model, row)` | Probability of default for one row. |
| `top_contributions(model, row, n=8)` | DataFrame of plain label, SHAP value, direction; the n largest by size. |
| `PLAIN_LABELS` | Model column name to plain words, for all 28 model columns. |
| `describe_client(row)` | Plain-word summary table for one client. |
| `outcomes_at_ratio(y, proba, ratio)` | The three-column comparison for the cost tab. |

Reused: `load_model`, `verify_raw`, `load_raw`, `clean`, `split`,
`decision_metrics`, `plot_tradeoff_curves`, `shap_explanation`. The median
rule in `pick_clients` is either reused or generalised so both modules share
one implementation.

### `app/app.py` (new)

Layout only: sidebar, three tabs, widgets, charts. Calls `showcase.py`.
Shows `st.error` and stops if `load_assets` fails.

### `app/requirements.txt` (new)

Pinned packages for Streamlit Community Cloud, including scikit-learn
1.9.1 and LightGBM 4.7.0 (the versions the model was trained with) and the
package itself (`.` or the repo path, whichever Community Cloud accepts).
The implementation plan confirms against Streamlit's documentation that a
requirements file in the app's folder is used ahead of `environment.yml` at
the repo root.

### Other changes

- `environment.yml`: add `streamlit`.
- `Makefile`: `make app` runs `streamlit run app/app.py`; add it to
  `make help`.

## Error handling

- Startup failure (model missing or stale, raw CSV checksum mismatch): the
  existing error message in an `st.error` box, then `st.stop()`.
- Library versions: `app/requirements.txt` pins the training versions, and a
  test fails if the pins differ from the installed versions. Retraining with
  newer libraries means updating the pins.
- Inputs: every control is limited to the ranges above, so the model only
  sees values inside what it was trained on (apart from the scenario rules'
  computed payments, which are bounded by the bills).

## Testing

`tests/test_showcase.py`, on a small LightGBM model fitted in the test (as
`test_evaluate.py` does):

- `cutoff_for(5)` is 1/6.
- `example_clients` returns four rows, each in the group its name says.
- Each scenario sets the statuses and payments in its table row, never
  produces a negative payment, and leaves bills and the input row unchanged.
- `edit_client` with "As recorded" and the original values returns a row
  equal to the original.
- `score_client` matches `predict_proba`.
- `top_contributions` returns n rows with plain labels; `PLAIN_LABELS` covers
  every model column.
- `outcomes_at_ratio` matches `decision_metrics`.

`tests/test_app.py`, with Streamlit's `AppTest` on the committed model and
data:

- The app starts with no exceptions, and each tab renders.
- Moving the slider changes the cutoff text.
- Each example client shows a probability.
- Changing payment behavior changes the probability.
- Opening the fairness section and changing sex runs without errors.

`tests/test_requirements.py`: the scikit-learn and LightGBM pins in
`app/requirements.txt` match the installed versions.

Manual check before sign-off: `make app`, click through every tab and
control, and share screenshots with the author.

## Out of scope

- Deploying, README link and GIF (piece B).
- The website page (piece C).
- A full 23-column input form.
- Fairness metrics by group, and retraining without demographics.
- Any change to the model or its cutoff.

## Revision after the manual check (2026-10-14)

The author's click-through found the shared sidebar confusing and the client view too
long to see the effect of a change. These replace the matching parts above:

- **No sidebar.** The cost ratio control (a number box with − / + buttons, 1 to 20,
  default 5) is in a narrow left panel on the Cost ratio tab and affects only that
  tab. The panel also shows defaulters caught, false alarms and cost per client (with
  the difference from cutoff 0.5), and a CSS rule keeps it in view while the table and
  charts scroll (on screens 640px or wider).
- **Cost ratio markers:** the current cutoff in orange-red, and cutoff 0.5 in gray as a
  fixed reference (one marker at 1x, where the two are the same).
- **Try a client uses the saved model's cutoff (0.170)**, the one that grouped the
  examples, so each example's decision matches its name.
- **Colored prediction:** a red box ("Predicted to default: flag for review") or a green
  one ("Predicted to pay: don't flag") with the probability, and the change from the
  original in percentage points.
- **Examples as a confusion matrix:** a 2×2 grid of buttons (rows: defaulted / paid;
  columns: flagged / not flagged), the selected one highlighted, plus a random button.
- **Your changes** is a heading-sized line, red when the risk goes up and green when it
  goes down.
- **Shorter controls:** payment
  behavior is a set of radio buttons. Client details and payment history move into a
  collapsed section below both columns, so the controls and the result fit on one
  screen.
