# Can Technical Indicators Predict Crypto Price Direction?

An exploratory study of Bitcoin (BTC-USD) and Ethereum (ETH-USD) price data. I tested whether price-derived signals (moving averages, RSI, MACD, Bollinger Bands) carry enough information to predict the direction of the next period. I then started extending the work with social-media sentiment.

**Short answer so far: not in any way I could demonstrate.** Models trained on price-based features fit the history well, but none of the experiments showed predictive power that would hold on unseen data. The main lesson of this project is about *evaluation*, not prediction.

---

## Question

> Given only historical price and volume data, can a model predict whether the next month's (or next days') closing price will be higher or lower than today's?

## Experiments

| Notebook | Approach | Data | Status |
|---|---|---|---|
| `BTC/mlprediction on btc_usd_chart.ipynb` | Rule-based: 20/50-day SMA crossover, projected 60 days ahead | Bitstamp BTC/USD daily (CSV) | Complete |
| `BTC/indicatorbasedprediction.ipynb` | Random Forest classifier on ~80 technical indicators (`ta` library) | ETH-USD and BTC-USD, monthly, 2020 – Feb 2025 (yfinance) | Complete, evaluated in-sample only |
| `BTC/csvX.ipynb` | LSTM on 7-day sequences of price + daily tweet sentiment (DistilBERT) | BTC-USD daily + Kaggle Bitcoin tweets dataset | Pipeline written, not fully run |
| `BTC/XLIVE.ipynb` | Same LSTM, fed with live X (Twitter) posts | X API | Stopped: X API access unavailable |

### 1. SMA crossover projection
The notebook computes the 20- and 50-day simple moving averages and labels each day "up" when SMA20 > SMA50. To look forward, it extends both averages linearly using their latest daily slope and simulates prices around them.

**Finding:** the 60-day forecast is "Down" for every day. That is a property of the method, not a market signal. When both averages are extended along straight lines from a state where SMA20 < SMA50, the crossover can only come from the slopes, so the projection mostly repeats the current trend. Extrapolating indicators is not prediction.

### 2. Random Forest on technical indicators
- Daily data is resampled to monthly candles (open, high, low, close, volume).
- Features: every indicator from `ta.add_all_ta_features`, plus SMA/EMA(20), RSI(14), Bollinger Bands and MACD.
- Target: `1` if next month's close is higher than this month's, otherwise `0`.
- Model: `RandomForestClassifier`, trained on all months except the latest, which is then predicted.

**Finding:** training accuracy was **1.00** for both coins. That number is a warning sign, not a result:
- It is measured on the same data the model was trained on.
- Since 2020 there are only about 60 monthly samples, against about 80 features. A Random Forest can memorise a dataset that small.
- The feature importances of the classic indicators (MACD, RSI, SMA, EMA, Bollinger) were each about 1–2%. No single indicator stood out.

A meaningful claim needs out-of-sample testing (see *Next steps*).

### 3. Adding sentiment (LSTM + DistilBERT)
The hypothesis behind this part was that price alone lacks the information that moves crypto markets, namely news and crowd sentiment. The pipeline scores tweets with a transformer model, averages the scores per day, merges them with closing prices, and trains a 2-layer LSTM to predict the next day's close.

The pipeline is written but not validated. While reviewing it I found several issues that would have to be fixed before any result could be trusted:
- `distilbert-base-uncased` is a base language model without a sentiment head. A fine-tuned checkpoint such as `distilbert-base-uncased-finetuned-sst-2-english` is needed.
- The `MinMaxScaler` is fitted on the full dataset before the train/validation split, which leaks future information.
- The price window (30 days) is very small for an LSTM, and it has to match the date range of the tweet dataset.
- Live data collection depends on X API access, which I no longer had.

## Conclusions

1. **Price-derived indicators alone showed no demonstrable predictive edge.** This matches the efficient-market view: public price information is largely already reflected in the price.
2. **High accuracy on training data is meaningless for time series.** It is the most common mistake in amateur trading models, and my first version made it too.
3. **If a signal exists, it is more likely in external information** (news, sentiment, on-chain data) than in candles. That is why the project moved toward sentiment.

## Next steps

- Replace the in-sample score with **walk-forward validation** (train on months 1…t, predict t+1, roll forward).
- Compare against **naive baselines**: "always up" and "same direction as last month". A model is only useful if it beats them.
- Use daily or weekly data to get enough samples, and reduce the feature set to avoid overfitting.
- Fix the sentiment pipeline issues listed above and evaluate it the same way.

## Tech stack

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · Hugging Face Transformers · `ta` · yfinance · tweepy · matplotlib · Google Colab

## Running the notebooks

```bash
pip install pandas numpy yfinance ta scikit-learn matplotlib tensorflow transformers tweepy joblib
```

- `indicatorbasedprediction.ipynb` downloads its data from Yahoo Finance and runs as is.
- `mlprediction on btc_usd_chart.ipynb` and `csvX.ipynb` were written for Google Colab and read CSV files from Google Drive (a Bitstamp BTC/USD daily export and a Kaggle Bitcoin tweets dataset). Adjust the file paths to run them locally.
- `XLIVE.ipynb` needs X API credentials.

## Disclaimer

This is a learning project, not financial advice. No trading was done based on these models.

## License

MIT, see [LICENSE](LICENSE).
