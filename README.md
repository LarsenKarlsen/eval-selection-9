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
 - Run command line interface
```python
poetry run cli
```


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

To see help messege about all available flags to each command use flag --help next to command.  
For example:
```python
poetry run cli train-knn-model --help
```