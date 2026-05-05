# Time2Feat

Segments a multivariate time series by extracting intra-signal features (via
tsfresh) and inter-signal features (pair-wise distance metrics) on sliding
windows, selecting the most informative features, and clustering the windows
into states.

The algorithm was originally designed for clustering collections of
multivariate time series. Here it is adapted for single-series segmentation:
the input series is split into (possibly overlapping) windows, each window is
treated as a separate sample, features are extracted and selected, and finally
a clustering step assigns a state to each window. The per-window labels are
then mapped back to per-timepoint labels via majority vote.

## Key properties

- Type: state detection
- Semi-supervised (requires `n_states`)
- Univariate and multivariate (leverages both intra-signal and inter-signal features)
- Requires `tsfresh` (`pip install tsseg[time2feat]`)

## Implementation

Adapted from the original repository by Angela Bonifati, Francesco Del Buono,
Francesco Guerra and Donato Tiano (University of Modena and Reggio Emilia /
University of Lyon).

- Origin: adapted from https://github.com/softlab-unimore/time2feat
- Licence: not specified in the original repository

## Citation

```bibtex
@article{DBLP:journals/pvldb/BonifatiB0T22,
  author       = {Angela Bonifati and
                  Francesco Del Buono and
                  Francesco Guerra and
                  Donato Tiano},
  title        = {Time2Feat: Learning Interpretable Representations for Multivariate
                  Time Series Clustering},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {16},
  number       = {2},
  pages        = {193--201},
  year         = {2022}
}
```
