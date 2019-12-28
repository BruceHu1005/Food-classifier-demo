from fastai.vision import *

path = Path('data/food-101/images')

data = ImageDataBunch.from_folder(path, valid_pct=0.2,
                                  ds_tfms=get_transforms(), size=224, num_workers=0, bs=64).normalize(imagenet_stats)

data.classes, data.c, len(data.train_ds), len(data.valid_ds)

learn = cnn_learner(data, models.resnet34, metrics=error_rate, pretrained=True)

lr = 1e-2

learn.fit_one_cycle(1, lr)

model_name = "resnet34"

learn.unfreeze()

learn.fit_one_cycle(1, max_lr=slice(1e-8, 1e-4))

learn.save(f'{model_name}-stage-2')

learn.load(f'{model_name}-stage-2');

learn.load(f'{model_name}-stage-2');

final_model_name = f'{model_name}-final'

learn.save(final_model_name)

learn.load(final_model_name);

learn.data.classes

shutil.rmtree("../models/", ignore_errors=True)
final_model_name = 'model.pkl'
learn.save('final_model_name')
learn.export()

with open('models/classes.txt', 'w') as f:
    json.dump(learn.data.classes, f)
