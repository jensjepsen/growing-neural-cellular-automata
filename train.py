import pytorch_lightning as pl
import model


model = model.Model()

#resume_from_checkpoint='/Users/jjqx/src/cellularautomata/lightning_logs/version_29/checkpoints/epoch=64-step=650.ckpt'
trainer = pl.Trainer()

trainer.fit(model)