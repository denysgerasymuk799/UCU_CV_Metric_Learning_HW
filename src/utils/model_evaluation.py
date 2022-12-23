import torch
import numpy as np
from tqdm import tqdm
from ml_metrics.average_precision import mapk
from sklearn.metrics import accuracy_score

from src.utils.annoy_utils import get_embedding


def evaluate(model, model_name, models_metrics_df, train_dataset, val_dataset, annoy_index, metrics_k):
    class_preds_lst = []
    class_true_lst = []
    superclass_preds_lst = []
    superclass_true_lst = []

    class_actual_lst = []
    class_predicted_lst = []
    superclass_actual_lst = []
    superclass_predicted_lst = []
    with torch.no_grad():
        for idx, (img, true_class_id, true_superclass_id) in enumerate(tqdm(val_dataset)):
            if img is None:
                continue
            embedding = get_embedding(model, img)
            neighbours = annoy_index.get_nns_by_vector(embedding, metrics_k)  # get top k closest

            pred_class_ids = []
            pred_superclass_ids = []
            for i in range(len(neighbours)):
                _, pred_class_id, pred_super_class_id = train_dataset[neighbours[i]]
                pred_class_ids.append(pred_class_id)
                pred_superclass_ids.append(pred_super_class_id)

            # Save results to compute accuracy
            class_true_lst.append(true_class_id)
            class_preds_lst.append(pred_class_ids[0])
            superclass_true_lst.append(true_superclass_id)
            superclass_preds_lst.append(pred_superclass_ids[0])
            # Save results to compute mAP@k
            class_actual_lst.append([true_class_id])
            class_predicted_lst.append(pred_class_ids)
            superclass_actual_lst.append([true_superclass_id])
            superclass_predicted_lst.append(pred_superclass_ids)

    precision = 4
    # Compute Accuracy
    class_id_accuracy = accuracy_score(np.array(class_true_lst), np.array(class_preds_lst))
    superclass_id_accuracy = accuracy_score(np.array(superclass_true_lst), np.array(superclass_preds_lst))
    class_id_accuracy = np.round(class_id_accuracy, precision)
    superclass_id_accuracy = np.round(superclass_id_accuracy, precision)
    print('\n')
    print(f'[{model_name}] Accuracy for class_id: {class_id_accuracy}')
    print(f'[{model_name}] Accuracy for superclass_id: {superclass_id_accuracy}')

    # Compute mAP@k
    class_id_mapk = mapk(class_actual_lst, class_predicted_lst, metrics_k)
    superclass_id_mapk = mapk(superclass_actual_lst, superclass_predicted_lst, metrics_k)
    class_id_mapk = np.round(class_id_mapk, precision)
    superclass_id_mapk = np.round(superclass_id_mapk, precision)
    print(f'[{model_name}] mAP@{metrics_k} for class_id: {class_id_mapk}')
    print(f'[{model_name}] mAP@{metrics_k} for superclass_id: {superclass_id_mapk}')

    model_metrics_dct = {
        'Model_Name': model_name,
        'Accuracy_Class_ID': class_id_accuracy,
        'Accuracy_Superclass_ID': superclass_id_accuracy,
        f'mAP@{metrics_k}_Class_ID': class_id_mapk,
        f'mAP@{metrics_k}_Superclass_ID': superclass_id_mapk,
    }
    return models_metrics_df.append(model_metrics_dct, ignore_index=True)
