import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn

df = pd.read_csv('dataset_MRK_prediction.csv')
df = df.drop(['Date' ,'Unnamed: 0'], axis=1)
target = "Close_lead1"
features = list(df.columns.difference(["Close", 'Close_lead1']))
size = int(len(df) * 0.67)

df_train = df.loc[:size].copy()
df_test = df.loc[size:].copy()
target_mean = df_train[target].mean()
target_stdev = df_train[target].std()

for c in df_train.columns:
    mean = df_train[c].mean()
    stdev = df_train[c].std()

    df_train[c] = (df_train[c] - mean) / stdev
    df_test[c] = (df_test[c] - mean) / stdev
    
from Factory import SequenceDataset
torch.manual_seed(101)

batch_size = 1
sequence_length = 3

train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)
test_dataset = SequenceDataset(
    df_test,
    target=target,
    features=features,
    sequence_length=sequence_length
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))

print("Features shape:", X.shape)
print("Target shape:", y.shape)

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss

def test_model(data_loader, model, loss_function):
    
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss

def predict(data_loader, model):
        """Just like `test_loop` function but keep track of the outputs instead of the loss
        function.
        """
        output = torch.tensor([])
        model.eval()
        with torch.no_grad():
            for X, _ in data_loader:
                y_star = model(X)
                output = torch.cat((output, y_star), 0)
        
        return output
from sklearn.metrics import mean_squared_error    
import numpy as np
def calculate_mse(actual, predicted):
    """
    Calculate Mean Squared Error (MSE) between actual and predicted values.
    Args:
        actual (array-like): Ground truth values.
        predicted (array-like): Predicted values.
    Returns:
        float: Mean Squared Error (MSE).
    """
    # return mean_squared_error(actual, predicted)
    actual, predicted = np.array(actual), np.array(predicted)
    # 避免除以零的情况
    non_zero_indices = actual != 0
    actual = actual[non_zero_indices]
    predicted = predicted[non_zero_indices]
    
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mape
    
    
def Mycost(X):
    from Factory import ShallowRegressionLSTM
    
    learning_rate = X[0]
    num_hidden_units = int(X[1])
    
    model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    classical_loss_train = []
    classical_loss_test = []
    print("Untrained test\n--------")
    test_loss = test_model(test_loader, model, loss_function)
    print()
    classical_loss_test.append(test_loss)

    for ix_epoch in range(5):
        print(f"Epoch {ix_epoch}\n---------")
        train_loss = train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_loss = test_model(test_loader, model, loss_function)
        print()
        classical_loss_train.append(train_loss)
        classical_loss_test.append(test_loss)


    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_test[ystar_col] = predict(test_loader, model).numpy()

    df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

    for c in df_out.columns:
        df_out[c] = df_out[c] * target_stdev + target_mean
    mse = calculate_mse(df_out[target], df_out[ystar_col])
    print(df_out)
    print(f"mape {mse}\n---------")
    return mse
    # from Factory import QShallowRegressionLSTM
    # import time
    # learning_rate = X[0]
    # num_hidden_units = int(X[1])

    # Qmodel = QShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units, n_qubits=4)
    # loss_function = nn.MSELoss()
    # optimizer = torch.optim.Adam(Qmodel.parameters(), lr=learning_rate)

    # quantum_loss_train = []
    # quantum_loss_test = []
    # print("Untrained test\n--------")
    # start = time.time()
    # test_loss = test_model(test_loader, Qmodel, loss_function)
    # end = time.time()
    # print("Execution time", end - start)
    # quantum_loss_test.append(test_loss)

    # for ix_epoch in range(20):
    #     print(f"Epoch {ix_epoch}\n---------")
    #     start = time.time()
    #     train_loss = train_model(train_loader, Qmodel, loss_function, optimizer=optimizer)
    #     test_loss = test_model(test_loader, Qmodel, loss_function)
    #     end = time.time()
    #     print("Execution time", end - start)
    #     quantum_loss_train.append(train_loss)
    #     quantum_loss_test.append(test_loss)
        
    # train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # ystar_col_Q = "Model forecast"
    # df_train[ystar_col_Q] = predict(train_eval_loader, Qmodel).numpy()
    # df_test[ystar_col_Q] = predict(test_loader, Qmodel).numpy()

    # df_out_Q = pd.concat((df_train, df_test))[[target, ystar_col_Q]]

    # for c in df_out_Q.columns:
    #     df_out_Q[c] = df_out_Q[c] * target_stdev + target_mean
        
    # mse = calculate_mse(df_out_Q[target], df_out_Q[ystar_col_Q])
    # print(df_out_Q)
    # return mse
        