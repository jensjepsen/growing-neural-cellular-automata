import typer, pytorch_lightning as pl, torch
import lib.model as model

app = typer.Typer()

@app.command()
def train(image_num: int=0, epochs: int=200, output_path: str='frontend/public/models/{image_num}_pool={use_pool}_damage={use_damage}_epochs={epochs}.onnx', use_pool: bool=True, use_damage: bool=True, data_points_per_epoch: int=100):
    args = locals().copy()
    
    lm = model.Model(image_num=image_num, use_pool=use_pool, use_damage=use_damage, data_points_per_epoch=data_points_per_epoch)

    trainer = pl.Trainer(
        max_epochs=epochs,
        **(
            {
                'accelerator': 'gpu',
                'devices': 1,
            }
            if torch.cuda.is_available()
            else {}
        )
    )
    trainer.fit(lm)

    lm.export_onnx(output_path.format(**args))

@app.command()
def train_all(epochs: int=200):
    for use_pool in [True, False]:
        for use_damage in ([True, False] if use_pool else [False]):
            for i in range(10):
                train(
                    image_num=i,
                    epochs=epochs,
                    use_pool=use_pool,
                    use_damage=use_damage
                )

if __name__ == '__main__':
    app()