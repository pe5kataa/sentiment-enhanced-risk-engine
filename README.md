# Sentiment-Enhanced Tail-Risk Forecasting

Code and data for the diploma thesis *"An Information System for Sentiment-Enhanced Tail-Risk Forecasting: Integrating FinBERT-Based Text Analytics"* (TU Dresden, Faculty of Business and Economics, Chair of Business Information Systems).

## The idea

Standard market-risk models look only at past prices. But a lot of what moves markets shows up in *text* first — news, policy statements, analyst commentary — before it shows up in returns. This project asks a simple question:

> **If you feed financial news sentiment into a risk model, do your tail-risk forecasts actually get better?**

To answer it, two models are built that are identical in every way *except* one gets sentiment features and the other doesn't. Both forecast the full probability distribution of tomorrow's S&P 500 return, and from that distribution both **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** are read off at the 95% and 99% levels. The two are then put through the same out-of-sample backtest and compared.

Two research questions:

1. Does FinBERT sentiment improve VaR/ES accuracy compared to a market-only model?
2. Does it help specifically during crisis periods (COVID-19 crash, 2022 selloff)?

## Background in one paragraph

**VaR** is the loss you don't expect to exceed on a given day at some confidence level — e.g. a 99% VaR of −2% means "we expect to lose more than 2% on roughly 1 day in 100." **ES** is the average loss *given* that you already blew through VaR, so it says something about how bad the bad days actually are. Both are properties of the *tail* of the return distribution, which means you can only compute them well if you model the whole distribution — not just a point forecast. That's what this project does.

## How it works

```
 ┌──────────────┐        ┌──────────────┐
 │ Market data  │        │ News data    │
 │ S&P 500 EOD  │        │ 20,657 news  │
 │ 2017–2025    │        │ articles     │
 └──────┬───────┘        └──────┬───────┘
        │                       │
   log returns,            FinBERT scoring
   30d rolling vol         P(pos)−P(neg) per article
        │                       │
        │              aggregate to daily index
        │              (mean, median, std, count,
        │               frac pos/neg/neu)
        └───────────┬───────────┘
                    │  merged on trading date, lagged by 1 day
                    ▼
        ┌───────────────────────────┐
        │  NGBoost + Student's t    │
        │  rolling 756-day window   │
        │  one-step-ahead forecast  │
        └───────────┬───────────────┘
                    ▼
        predicted distribution → VaR & ES @ 95% / 99%
                    ▼
        backtest vs. realized returns
```

**Step by step:**

1. **Market data.** Daily S&P 500 closes (2017–2025) from the EODHD API, converted to log returns. A 30-day rolling volatility feature is added, lagged so today's return never leaks into today's feature.

2. **News data.** 20,657 articles pulled from the EODHD News API — SPY plus the largest S&P 500 constituents (AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, JPM, JNJ), deduplicated by link/title.

3. **Sentiment extraction.** Each article is scored with **FinBERT** (`ProsusAI/finbert`), a BERT variant fine-tuned on financial text. FinBERT returns P(positive), P(neutral), P(negative); the continuous sentiment score is `P(pos) − P(neg)`, which captures both direction and intensity. Long articles are chunked to 512 tokens and scored in pieces. Timestamps are aligned to the New York market day, not raw UTC.

4. **Daily aggregation.** Article scores collapse into a daily sentiment index: mean, median, standard deviation, standard error, article count, and the fraction of positive/negative/neutral articles.

5. **Distributional forecasting.** **NGBoost** with a **Student's t** distribution predicts location, scale, and degrees of freedom — not a single number, but an entire probability distribution per day. Student's t is used instead of a normal because financial returns have fat tails: extreme moves happen far more often than a bell curve allows. Base learners are depth-2 trees, 200 estimators, lr 0.01.

6. **Rolling backtest.** For every trading day, the model retrains on the previous **756 days (~3 years)** and forecasts one day ahead. Features and target are standardized inside each window and transformed back afterwards. Nothing from the future ever touches the training set. Out-of-sample evaluation runs from **February 2020 onward** (~1,500 forecast days), covering COVID-19, the 2022 drawdown, and 2025.

7. **Risk measures.** VaR is the lower-tail quantile of the predicted distribution. ES is computed by drawing 3,000 Monte Carlo samples from that distribution and averaging the ones below VaR.

8. **Evaluation.** Kupiec unconditional coverage (is the violation rate right?), Christoffersen independence (do violations cluster?), Diebold–Mariano (is one model's forecast loss significantly lower?), plus empirical ES metrics — tail-mean gap, bias, MAE, RMSE — measured on violation days only.

## Repository layout

```
notebooks/
  news_data_processing.ipynb   # pull news from EODHD, merge & dedupe
  finbert_model.ipynb          # FinBERT scoring → daily sentiment index
  01_baseline_var_es.ipynb     # market-only NGBoost model + rolling backtest
  02_sentiment_var_es.ipynb    # sentiment-augmented model, same setup
  03_evaluation_tests.ipynb    # Kupiec, Christoffersen, DM, ES metrics
  04_tables_plots.ipynb        # thesis tables and figures

data/
  raw/                         # SP500.csv, DAX.csv
  processed/                   # returns, daily sentiment, model outputs
  processed/evaluation/        # backtest summaries and plot sources

artifacts/plots/               # figures used in the thesis
```

Run the notebooks in the order listed: news → FinBERT → baseline → sentiment → evaluation → plots.

## Results

**VaR violations** (a violation = realized return fell below the forecast VaR):

| Model | Level | Violations | Rate | Expected |
|---|---|---|---|---|
| Baseline | 95% | 107 | 7.06% | 5% |
| Sentiment | 95% | 97 | 6.58% | 5% |
| Baseline | 99% | 22 | 1.45% | 1% |
| Sentiment | 99% | 24 | 1.63% | 1% |

Both models under-forecast risk at 95% and are rejected by the Kupiec test, though sentiment is somewhat closer to target. At 99% the baseline passes Kupiec (p = 0.098) and sentiment does not (p = 0.026). Violations don't cluster at 95% for either model; at 99% the baseline shows mild clustering (p = 0.039) while sentiment does not (p = 0.060).

**Diebold–Mariano:** no significant difference at 95% (p = 0.31). At 99% the difference is significant (p = 0.032) but in the *baseline's* favour — and economically tiny (mean loss difference ~2e-05).

**Expected Shortfall** on violation days:

| Model | Level | Tail-mean gap | MAE | RMSE |
|---|---|---|---|---|
| Baseline | 95% | −0.0015 | 0.0089 | 0.0179 |
| Sentiment | 95% | **+0.0004** | **0.0086** | **0.0143** |
| Baseline | 99% | −0.0114 | **0.0161** | **0.0200** |
| Sentiment | 99% | **−0.0079** | 0.0148 | 0.0212 |

This is where sentiment does best: at 95% the sentiment model's average predicted ES is almost exactly on top of the realized tail mean, with lower error dispersion. At 99% both models overshoot the tail loss, sentiment less so.

## What the study concludes

**Sentiment is complementary, not decisive.** Adding FinBERT features doesn't materially improve VaR forecasting — the two models track each other closely across the entire sample, including through the March 2020 crash, where they behave almost identically. There is a modest, real improvement in ES calibration at the 95% level, and that's about it.

The likely reason: the S&P 500 is one of the most liquid markets in the world, so public news is priced in fast, and the market variables (lagged return, rolling volatility) already absorb most of what the sentiment index would tell you. Big crisis moves turn out to be driven by systemic and macroeconomic forces rather than by anything a news-tone index captures.

The contribution is therefore as much architectural as empirical: it shows how unstructured text can be operationalized end-to-end inside a probabilistic risk-forecasting pipeline — an NLP subsystem feeding a distributional forecasting engine feeding regulatory risk measures — and it reports honestly what that buys you.

## Running it

**Requirements:** Python 3, `pandas`, `numpy`, `scipy`, `scikit-learn`, `ngboost`, `torch`, `transformers`, `matplotlib`, `requests`, `python-dotenv`, `tqdm`.

```bash
pip install pandas numpy scipy scikit-learn ngboost torch transformers matplotlib requests python-dotenv tqdm
```

**API key:** the data-collection notebooks read an EODHD key from a `.env` file in the project root:

```
api_key=YOUR_EODHD_KEY
```

If you only want to reproduce the modelling and evaluation, you can skip the collection steps — the processed data in `data/processed/` is already there.

**Note on runtime:** the rolling backtest refits NGBoost once per trading day over ~1,500 days, so notebooks 01 and 02 take a long time to run. FinBERT scoring of 20k articles also benefits considerably from a GPU or Apple Silicon MPS.

## Caveats

- Sentiment is aggregated at the index level from a news set skewed toward mega-cap constituents; it is not a true S&P-500-wide sentiment measure.
- Both models systematically under-forecast risk at the 95% level — this is a property of the setup, not a bug in the comparison, and it applies equally to both, which is what makes the comparison fair.
- Baseline and sentiment backtests end on slightly different dates (baseline runs a bit longer). Tests that compare the two directly, such as Diebold–Mariano, are restricted to the common set of dates.
