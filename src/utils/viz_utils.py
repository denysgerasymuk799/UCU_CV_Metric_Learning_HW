import random
import numpy as np
import matplotlib.pyplot as plt

from src.utils.annoy_utils import get_embedding


def normalized_img_tensor_to_rgb(img):
    rgb_img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1)

    return rgb_img


def show_img(dataset, idx):
    img, class_id, superclass_id = dataset[idx]

    fig = plt.figure()
    rows, cols = 1, 1
    fig.add_subplot(rows, cols, 1)
    rgb_img = normalized_img_tensor_to_rgb(img)
    plt.imshow(rgb_img)

    plt.axis('off')
    plt.title('Image')
    print(f'Class id: {class_id}\nSuper Class id: {superclass_id}\n')


def show_retrieval(model, annoy_index, train_dataset, test_dataset, k_closest):
    length_test_dataset = len(test_dataset)

    plt.rcParams['figure.figsize'] = [15, 10]
    rows, cols = 5, 5
    f, axarr = plt.subplots(rows, cols)
    for row in range(rows):
        img_idx = random.randint(1, length_test_dataset - 1)
        img, true_class_id, true_superclass_id = test_dataset[img_idx]
        # neightbours = annoy_index.get_nns_by_vector(feature_v, 5)[1:]
        embedding = get_embedding(model, img)
        neighbours = annoy_index.get_nns_by_vector(embedding, k_closest)  # get top k closest

        axarr[row, 0].imshow(normalized_img_tensor_to_rgb(img))
        axarr[row, 0].axis('off')
        axarr[row, 0].set_title(f'True Superclass {true_superclass_id}\nTrue Class {true_class_id}')

        for col in range(1, cols):
            axarr[row, col].axis('off')
            if col - 1 >= len(neighbours):
                break

            img, pred_class_id, pred_superclass_id = train_dataset[neighbours[col - 1]]
            axarr[row, col].imshow(normalized_img_tensor_to_rgb(img))
            axarr[row, col].spines['bottom'].set_color('0.5')
            axarr[row, col].set_title(f'Predicted Superclass {pred_superclass_id}\nPredicted Class {pred_class_id}')

    plt.subplots_adjust(top=1.4, bottom=0.01)
    line = plt.Line2D((.27, .27), (0, 1.4), color="grey", linewidth=3, linestyle='--')
    f.add_artist(line)
