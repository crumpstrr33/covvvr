This directory contains files used in the analysis done and plots shown in the paper.

- The variance reduction heatmaps are created in `plots.ipynb` by importing the `2cv_*.pkl` files from the `data` directory. These files are created by `calc_percdiff.py` in the `scripts` directory.
- The correlation plots are also created in `plots.ipynb`.
- The tables are generated in `latex.ipynb`. The means table is generated from the `means_*.pkl` files which are created by `multirun.py`. The 1, 2 and all CV comparision table is generated from the `compare_*.pkl` files which are created by `cv_compare.py`.
