# eval-selection-9
9 module in ML-intro RSschool 2022


Instructions  
1. How to install  
 - Clone repository  
 - Install poetry (if u dont have it)  
    You can get it from https://python-poetry.org/
 - Go to repository folder and run
```python
poetry install
```
 - Project have following structure
```
.
|___ data
|   |____ processed # folder to store data after manipulations
|   |____ raw  # raw data folder ! never change it ! 
|___ models # store your models here
|___ notebooks # place when you can keep your jupyter notebooks
|___ reports # store reports and visualizations of data there
|___ src/forest_ml # source code
    |___ data_prep # contains scripts to process data
    |   |___ create_folders.py # creates necessary folders
    |   |___ train_val.py # splitting strateges
    |___ model # 
    |   |___ knn_models.py # Pipeline to create KNN models
    |   |___ lr_models.py # Pipeline to create Linear Regression models
    |   |___ train.py # Training algorithm
    |___ visualization # creating different reports
    |   |___ profiling.py # pandas profiling report
    |___ __init__.py # make package
    |___ cli.py # command line interface
```
You can download dataset from [here](https://www.kaggle.com/competitions/forest-cover-type-prediction)



## CLI commands:
* Creates necessary folders  
```python
init-folders
```
* Creates pandas profilling report
```python
make-report
```  
* Train knn model
```python
train-knn-model
```
* Train lr model
```python
train-lr-model
```
* You also can use autotune for Linear model, but it may take time.
```python
        grid = {
            'clf__solver':['liblinear','saga'],
            'clf__penalty':['l1', 'l2'],
            'clf__C':[0.1, 100, 10, 0.1, 0.01]
        }

```
Command
```
train-lr-auto
```

To see help messege about all available flags to each command use flag --help next to command.  
For example:
```python
poetry run cli train-knn-model --help
```
For expirement tracking using mlflow. To start type in console
```python
mlflow ui
```