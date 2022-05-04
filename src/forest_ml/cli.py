import click
from forest_ml.model.train import train_lr


@click.group()
def cli():
    pass

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
    default="LR_model",
    type=str,
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="models/",
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
    default='saga',
    type=str,
    show_default=True,
)
def train_lr_model(data_path, model_name_prefix, save_model_path, use_scaller, max_iter, c, penalty, solver):
    train_lr(
        data_path=data_path,
        name_prefix=model_name_prefix,
        save_model_path = save_model_path,
        use_scaller = use_scaller,
        max_iter = max_iter,
        C = c,
        penalty = penalty,
        solver = solver 
    )

if __name__ == '__main__':
    cli()