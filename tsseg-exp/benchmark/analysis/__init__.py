"""Analysis subpackage — extracted from benchmark/paper_figures_final.ipynb.

Layout:
  helpers.py        — shared imports, constants, palette, MLflow manager, helpers
  data.py           — fetches and builds all DataFrames, with disk caching
  performance.py    — strip + CD + spider figures (CPD/SD, non-guided + guided)
  runtime.py        — runtime vs accuracy figure
  scatter.py        — CPD vs SD scatter figure
  sms_breakdown.py  — SMS error-type stacked-bars figure
  cache/            — pickled DataFrames produced by data.load_data()
"""
