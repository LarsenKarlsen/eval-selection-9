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
 - Next run
```python
poetry run cli
```
This command runs command line interface

CLI commands:
--help
```python
init-folders
```
Creates necessary folders  
```python
make-report
```  
Creates pandas profilling report
```python
train-knn-model
```
Train knn model
```python
train-lr-model
```
Train lr model

To see help messege about all available flags to each command use flag --help next to command.  
For example:
```python
poetry run cli train-knn-model --help
```