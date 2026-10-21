## 1. Fetch data
Run
```bash
python NeuralNetwork/fetch_samples.py
```

## 2. Clean data
I need to clean the rows, because 25–60% of each ticker's bars are pre/after-market with about 50× less volume, and quiet periods have no bars at all, so the series aren't on a regular grid right after the fetch.

Run
```bash
python NeuralNetwork/clean_data.py
```