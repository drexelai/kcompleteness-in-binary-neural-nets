# sparsity-in-binary-neural-nets
Drexel AI sparsity paper

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

