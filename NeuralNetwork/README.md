## 1. Fetch data
Run
```bash
python NeuralNetwork/fetch_samples.py
```

## 2. Clean data
Removes off-hour bars, because they are spurious and low in volume, mostly noise. This results in fully packed good valid bars for all symbol.

| Session  | Hours (US/Eastern) | 5-min bars          |
|----------|--------------------|---------------------|
| Full day | 09:30 – 16:00      | 6 + 12 × 6 = 78/day |
| Half day | 09:30 – 13:00      | 6 + 12 × 3 = 42/day |

Run
```bash
python NeuralNetwork/clean_data.py
```
To check that data is clean, look at distribution using `python check_clean.py`.
Check that there are no trades outside regular trading hours, and that there are no big outliers in y axis.

## 3. Build features
For a specified symbol, for each row, compute training features and save to `NeuralNetwork/data_features`.
Labels are also assigned at this stage.
```bash
python build_features.py -t {symbol}
```

## 4. 
Creates single `train.npz`, `valid.npz`, and `holdout.npz`. Holdout for testing ticker that did NOT enter training. Generalizability test.
```bash
python splitDataset.py
```