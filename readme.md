

Function Approximation with DNN's
=====

For context, see `latex/main.pdf` (semester paper HS18 at ETH, under Prof. Dr. H. Bölcskei, supervised by Dmytro Perekrestenko).

See `code/todo.txt` for some things left to do. The code has some redundancy and is mostly contained within one file, could be easily refactored to a very flexible framework for similar experiments.

The code runs within a conda3 environment, see `code/spec-file.txt` and `code/environment.yml` for the specifications.


Running the code:
----

For a simple run, set the `params` at the beginning, pick a function to approximate (see ex. `x_dat, y_dat, test_x_dat, test_y_dat = approxwilg()`), and in `main()` pick `exploreIndividual()` or `exploreHeatmap()`. Be sure to pick a unique experiment id for each run. Activate the conda environment and set the CUDA devices:

```
source activate dnn
CUDA_VISIBLE_DEVICES="" python aprox.py
```

When profiling make sure you are not multiprocessing to avoid (double) pickling errors:
`python -m cProfile -s cumtime aprox.py`

Multiprocessing can work via CUDA however was designed for the CPU due to the nature of the experiments.

There is some other code for plotting and for examining the gabor sampling, see `code/`.



