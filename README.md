# sparsity-in-binary-neural-nets
Drexel AI sparsity paper: https://arxiv.org/pdf/2101.06518.pdf

Datasets for Binary Classification
https://jamesmccaffrey.wordpress.com/2018/03/14/datasets-for-binary-classification/


## Development
* Install conda to your computer (once during first installation)
* Create an environment from environment.yml by  
`conda env create -f environment.yml`
(only once during first installation)
* Activate the environment (every time you start developing)
`conda activate sparsity`
* [OPTIONAL] If you need to install a new package, you can do `pip install package`.
* Finally, do `conda deactivate` when you are done.

## Run
`python main.py`
User arguments can be passed such as:
`python main.py -verbose=1 -batch_size=12` 
For a full list of arguments see, pipeline/argument_parser.py file.

If you end up using our work, please cite as below:
```
@article{alparslan2021evaluating,
  title={Evaluating Online and Offline Accuracy Traversal Algorithms for k-Complete Neural Network Architectures},
  author={Alparslan, Yigit and Moyer, Ethan Jacob and Kim, Edward},
  journal={arXiv preprint arXiv:2101.06518},
  year={2021}
}
```
