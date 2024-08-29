from src.utils_supervised import Transforms, PatchDataModule, SupervisedClassifier
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import timm
import torch
import pandas as pd
import lightning as L


BATCH_SIZE = 128
NUM_WORKERS = 8
LEARNING_RATE = 3e-4
MAX_EPOCHS = 100

torch.set_float32_matmul_precision('medium')

def main():

    '''
    LOADING THE DATA
    LABELS
    0 -> normal (normal slides)
    1 -> abnormal patches from abnormal slides
    2 -> normal patch in abnormal slide
    3 -> patches in the border of the annotation
    '''

    print("Loading dataset")    
    df = pd.read_csv('csv/camelyon16_data.csv')

    # just loading normal and abnormal patches from normal and abnormals slides respectively
    df = df.query("label in [0,1]").sample(frac=1.)    
    dataset = {}
    # dataset['train'] = df.query("subset=='train'")[['img_path', 'label']].values.tolist()  
    dataset['val'] = df.query("subset=='val'")[['img_path', 'label']].values.tolist()
    dataset['test'] = df.query("subset=='test'")[['img_path', 'label']].values.tolist()

    # Need to balance the classes
    train_df = df.query("subset=='train'")
    positive_df = train_df.query("label==0")
    negative_df = train_df.query("label==1")
    positive_n = positive_df.shape[0]
    negative_n = negative_df.shape[0]
    if positive_n > negative_n:
        positive_df = positive_df.sample(n=negative_n)
    else:
        negative_df = negative_df.sample(n=positive_n)
    dataset['train'] = pd.concat([positive_df, negative_df], axis=0)[['img_path', 'label']].values.tolist()

    # dataset['train'] = dataset['train'][:1000]
    # dataset['val'] = dataset['val'][:1000]
    # dataset['test'] = dataset['test'][:1000]

    train_transforms = Transforms(resize_int=128).train_transforms
    test_transforms = Transforms(resize_int=128).test_transforms

    datamodule = PatchDataModule(
        dataset=dataset,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS)
    datamodule.setup()
    
    one_batch = next(iter(datamodule.train_dataloader()))
    print("Shape of one batch: ", one_batch[0].shape, one_batch[1].shape)
    print("Data loaded successfully")  

    # TRAINING
    backbone = timm.create_model('resnet18', pretrained=True)
    backbone.fc = torch.nn.Linear(512, 2)
    model = SupervisedClassifier( 
        backbone=backbone,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        num_classes=2,
    )

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    progress_bar_callback = RichProgressBar(leave=False)

    trainer = L.Trainer(
        num_sanity_val_steps=0,  
        max_epochs=MAX_EPOCHS,
        precision='16-mixed',
        accelerator="gpu",
        devices='auto',
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar_callback],           
    )

    trainer.fit(model=model, datamodule=datamodule)

    # SAVING BEST MODEL
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model_state_dict = torch.load(best_model_path)['state_dict']
    torch.save(best_model_state_dict, 'models/supervised_resnet18.pth')

    # TESTING
    trainer.test(ckpt_path="best", datamodule=datamodule)

    print("Training and testing completed successfully")


    return


if __name__ == "__main__":
    main()
