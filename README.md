# Guys
Be sure to not change gitignore file, you can't upload the data file to github anyways since its large but still cautioning.

## Environment setup

Use Python `3.13`:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## what the preprocssing notebook does (highlights people):
- low-frequency cell-type removal (`<5000`)
- unknown ethnicity removal
- QC filtering (`n_genes`, `n_umis`, `pct_counts_mito`)
- training-only HVG selection (top 2000 after log-normalization)
- application of training HVGs to val/test


some one taskette help me run prerpocessing script to run data, my python file in memory went upto 80Gigs before crashing lol god save us all

also implemented dataloader, its simplest version havent spent a lot of time on it, you can make changes if it doesnt work in anyway, I'm sure it can be improved.

lastly added skeleton for 3 methods its just trash passing the baton to you now, SAYONARA let me sleep for next 10 days :D