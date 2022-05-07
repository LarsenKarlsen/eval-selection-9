from email.policy import default
import click
from forest_ml.model.train import train_lr
from forest_ml.model.train import train_knn
from forest_ml.data_prep import create_folders
from forest_ml.visualization.profiling import raw_train_profiling_report


@click.group()
def cli():
    pass


@cli.command()
def init_folders():
    click.echo(f'[+] Creating data dir')
    create_folders.create_data_folders()
    click.echo(f'[+] Creating models folder')
    create_folders.create_models_folder()
    click.echo(f'[+] Creating notebook folder')
    create_folders.create_notebook_folder()

@cli.command()
@click.option(
    '--path',
    default="data/raw/train.csv",
    type= str,
    show_default=True,
)
def make_report(path):
    click.echo(f'[+] Creating pandas profilling report from data at {path}')
    report_path = raw_train_profiling_report(dataset_path=path)
    click.echo(f'[+] Report created at {report_path}')
    

@cli.command()
@click.option(
    "-d",
    "--data-path",
    default="data/raw/train.csv",
    type= str,#click.Path(exists=True, dir_okay=False, path_type=click.Path),
    show_default=True,
)
@click.option(
    "--model-name-prefix",
    default="LR_model_last",
    type=str,
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="models",
    type=str, #click.Path(dir_okay=True, path_type=click.Path),
    show_default=True,
)
@click.option(
    "--use-scaller",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    '--use-pca',
    default=0,
    type=int,
    show_default = True
)
@click.option(
    "--max-iter",
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--penalty",
    default='l2',
    type=str,
    show_default=True,
)
@click.option(
    "--solver",
    default='lbfgs',
    type=str,
    show_default=True,
)
def train_lr_model(data_path, model_name_prefix, save_model_path, use_scaller, use_pca, max_iter, c, penalty, solver):
    train_lr(
        data_path=data_path,
        name_prefix=model_name_prefix,
        save_model_path = save_model_path,
        use_scaller = use_scaller,
        use_PCA = use_pca,
        max_iter = max_iter,
        C = c,
        penalty = penalty,
        solver = solver 
    )


@cli.command()
@click.option(
    "-d",
    "--data-path",
    default="data/raw/train.csv",
    type= str,#click.Path(exists=True, dir_okay=False, path_type=click.Path),
    show_default=True,
)
@click.option(
    "--model-name-prefix",
    default="KNN_last",
    type=str,
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="models",
    type=str, #click.Path(dir_okay=True, path_type=click.Path),
    show_default=True,
)
@click.option(
    '--use-scaller',
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    '--use-pca',
    default=False,
    type=bool,
    show_default=True
)
@click.option(
    '--n-neighbors',
    default = 5,
    type=int,
    show_default=True
)
@click.option(
    '--weights',
    default = 'uniform',
    type=str,
    show_default=True
)
@click.option(
    '--algorithm',
    default='auto',
    type=str,
    show_default=True
)
@click.option(
    '--leaf-size',
    default=30,
    type=int,
    show_default=True
)
@click.option(
    '--p',
    default=2,
    type=int,
    show_default=True
)
@click.option(
    '--metric',
    default='minkowski',
    type=str,
    show_default=True
)
def train_knn_model(data_path, model_name_prefix, save_model_path, use_scaller,
                    use_pca, n_neighbors, weights, algorithm, leaf_size, p, metric):
    train_knn (
        data_path=data_path,
        name_prefix=model_name_prefix,
        save_model_path=save_model_path,
        use_scaller = use_scaller,
        use_PCA=use_pca,
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric
    )

if __name__ == '__main__':
    cli()