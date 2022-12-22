import torch

from tqdm import tqdm
from annoy import AnnoyIndex

from src.configs import DEVICE


def get_embedding(model, img):
    """
    Return image embedding from the desird layer
    """
    # Select the desired layer from the model object
    layer = model._modules.get('avgpool')
    # In case of MultilabelClassifier
    if layer is None:
        layer = model.model_wo_fc._modules.get('8')

    # The 'avgpool' layer has an output size of 512
    embedding = torch.zeros(512)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))

    # Attach copy_data function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Transform image and run the model
    model(img.unsqueeze(0).to(DEVICE))
    # Detach our copy function from the layer
    h.remove()

    return embedding


def build_index(model, train_dataset, model_output_size, model_name, num_trees, data_dir):
    num_skipped = 0
    num_saved = 0
    t = AnnoyIndex(model_output_size, 'angular')
    # for idx, (img, class_id, superclass_id) in enumerate(tqdm(train_dataset)):
    for idx, data in enumerate(tqdm(train_dataset)):
        img, class_id, superclass_id = data
        if img is None:
            num_skipped += 1
            continue
        embedding = get_embedding(model, img)
        t.add_item(idx, embedding)
        num_saved += 1

    print(f'\nNumber of skipped images because of errors: {num_skipped}')
    print(f'Number of successfully saved images: {num_saved}')

    t.build(num_trees)
    t.save(f'{data_dir}/{model_name}_train_{len(train_dataset)}_index.ann')

    return t


def load_index(model_output_size, model_name, train_ds, data_dir):
    # Super fast, will just mmap the file
    index = AnnoyIndex(model_output_size, 'angular')
    index.load(f'{data_dir}/{model_name}_train_{len(train_ds)}_index.ann')
    return index
