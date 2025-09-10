import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from data.data_preprocessing import get_features_df, train_test_split, get_scaled_dfs, \
    create_sequences, create_datasets, create_loaders
from results.evaluation import get_preds_labels, get_descaler, descale
from models.model import CovidPredictor
from utils.seed import set_seed
import torch.nn as nn
from training import train_model
from results.visualization import plot_predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = 'time_series_covid19_confirmed_global.csv'
    country = 'Poland'

    long_eps = 20
    lat_eps = 10
    sequence_length = 25
    train_split = .9
    batch_size = 8
    num_epochs = 50
    lr = 1e-3

    n_hidden = 128
    n_layers = 2
    dropout = 0.3

    set_seed()




    features_df, dates = get_features_df(lat_eps, long_eps, country, filename)
    train_df, test_df = train_test_split(features_df, train_size=train_split)
    input_size = train_df.shape[1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df)
    train_df_scaled, test_df_scaled = get_scaled_dfs(scaler, train_df, test_df)
    train_seq = create_sequences(train_df_scaled, country, sequence_length)
    test_seq = create_sequences(test_df_scaled, country, sequence_length)
    train_dataset, test_dataset = create_datasets(train_seq, test_seq)
    train_loader, test_loader = create_loaders(train_dataset, test_dataset, batch_size)

    model = CovidPredictor(input_size, n_hidden, n_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= lr)

    model = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

    preds, labels = get_preds_labels(model, test_dataset, device)

    descaler = get_descaler(features_df, scaler)
    predictions_descaled = descale(descaler, preds)
    labels_descaled = descale(descaler, labels)

    mae_score = mean_absolute_error(labels_descaled, predictions_descaled)

    print(f"MAE: {mae_score}")

    plot_predictions(predictions_descaled, labels_descaled, dates[int(len(features_df)*train_split)+sequence_length:])



if __name__ == '__main__':
    main()
