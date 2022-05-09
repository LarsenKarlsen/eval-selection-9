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
* and KNN model
```python
        grid = {
            'clf__weights':['uniform', 'distance'],
            'clf__metric':['euclidean', 'manhattan', 'minkowski'],
            'clf__n_neighbors':range(1, 21, 2)
        }
```
Command
```
train-knn-auto
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
### PS INCLUDE MLRUNS dir so u can see my runs
## MLFLOW Screenshots
<img width="1440" alt="Снимок экрана 2022-05-09 в 19 27 41" src="https://user-images.githubusercontent.com/48959644/167455765-0775e8db-9518-46c9-82ba-76c6dfec15cb.png">
<img width="1440" alt="Снимок экрана 2022-05-09 в 19 27 26" src="https://user-images.githubusercontent.com/48959644/167455748-665aa6f6-816d-48df-8854-d5a32c65a73f.png"><img width="1440" alt="Снимок экрана 2022-05-09 в 19 27 54" src="https://user-images.githubusercontent.com/48959644/167455777-e918452f-381f-40fa-a227-e5d83910a0ca.png">
<img width="1440" alt="Снимок экрана 2022-05-09 в 19 28 12" src="https://user-images.githubusercontent.com/48959644/167455788-4473228a-52d3-43fe-9b67-a035ee78adff.png">
<img width="1440" alt="Снимок экрана 2022-05-09 в 19 28 22" src="https://user-images.githubusercontent.com/48959644/167455793-dd154ce4-c0e7-4d7d-83f0-85bd27596da9.png">
<img width="1440" alt="Снимок экрана 2022-05-09 в 19 28 37" src="https://user-images.githubusercontent.com/48959644/167455797-67f1fe19-4850-4820-89ac-8f70af97ca32.png">


