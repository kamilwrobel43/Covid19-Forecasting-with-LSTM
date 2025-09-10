from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_preds_labels(model, test_dataset, device):
    predictions = []
    labels = []

    for item in tqdm(test_dataset):
        sequence = item["sequence"].to(device)
        label = item["label"]

        output = model(sequence.unsqueeze(dim=0))
        predictions.append(output.item())
        labels.append(label.item())

    return predictions, labels

def get_descaler(features_df, scaler, country = 'Poland'):
    idx = features_df.columns.get_loc(country)
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[idx], scaler.scale_[idx]

    return descaler

def descale(descaler, values):
    values_2d = np.array(values)[:, np.newaxis]
    return descaler.inverse_transform(values_2d).flatten()

