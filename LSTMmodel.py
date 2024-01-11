import pandas as pd
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import data_cleaners as dc
from collections import defaultdict
from datetime import datetime, timedelta, date
import inspect

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 15
QUANTILES = [0.025, 0.100, 0.250, 0.500, 0.750, 0.900, 0.975]


class TSDataset(Dataset):
    """Use PyTorch Dataset to generate Time Series dataset.
    Dataset stores the samples and their corresponding labels.
    """

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx].copy(), self.targets[idx].copy()
        return x, y


def make_loader(X, Y, batch_size=32, shuffle=True):
    """Use PyTorch DataLoader that wraps an iterable around the Dataset to
    enable easy access to the samples.

    Returns:
    dataloader: PyTorch DataLoader that wraps an iterable around a TSDataset object
    """
    dataset = TSDataset(data=X, targets=Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class QuantileLoss(nn.Module):
    """Implement a custom loss function by subclassing nn.Module and overriding
    the forward method.
    """

    def __init__(self, q_idx):
        """Constructor

        Parameters:
        :param q_idx: the index of the quantile in the QUANTILES list
        """
        super(QuantileLoss, self).__init__()

        # For quantile at index
        self.q = QUANTILES[q_idx]

    def forward(self, outputs, targets):
        """The forward method takes as input the predicted output and the actual output and
        returns the value of the loss.

        Parameters:
        :param outputs: predicted output
        :param targets: actual output

        Returns:
        The value of the quantile loss.

        NOTE: In order to enable correct backpropagation, use PyTorch functions while calculating the
        value of the quantile loss.
        """

        q = self.q

        # Calulate error
        # print(targets.shape, outputs.shape) # Uncomment to check dimensions
        q_e = torch.subtract(targets, outputs)
        q_loss = torch.mean(torch.maximum(q * q_e, (q - 1) * q_e))

        return q_loss


class LSTM_Model(nn.Module):
    """SpatioTemporal model for forecasting COVID-19 incidence."""

    def __init__(self, no_temporal_features=1):
        """Constructor

        Parameters:
        :param no_temporal_features: number of expected features in input to LSTM
        """
        super(LSTM_Model, self).__init__()
        self.num_layers = 2
        self.hidden_size = 64
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define model architecure, ensure LSTM is defined first.
        # For the LSTM layer, batch_first should be true; bidirectional should be false.
        # use hidden size as 64 and num layers as 2.
        # When you look at the paper you will see that they have two LSTMs. You can
        # achieve a similar effect by having a single LSTM with 2 layers.
        self.lstm = nn.LSTM(
            input_size=no_temporal_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

        # Followed by the fully connected layer.
        # Here we will use 64 as out_features to match the authors' implementation.
        # Note, this is the implementation in their github code, NOT in the paper.
        self.fc = nn.Linear(in_features=64, out_features=64)

        # Initialize the weights for the linear layer. To figure out the input
        # and output dimensions remember that this is a fully connected layer.
        torch.nn.init.xavier_uniform_(self.fc.weight)

        # Create an output for each quantile using nn.ModuleDict.
        # [.025, .100, .250, .500, .750, .900, .975]
        self.linear = nn.ModuleDict(
            {
                "025": nn.Linear(in_features=64, out_features=1),
                "100": nn.Linear(in_features=64, out_features=1),
                "250": nn.Linear(in_features=64, out_features=1),
                "500": nn.Linear(in_features=64, out_features=1),
                "750": nn.Linear(in_features=64, out_features=1),
                "900": nn.Linear(in_features=64, out_features=1),
                "975": nn.Linear(in_features=64, out_features=1),
            }
        )

        # Initialize the weights for each output module using the xavier_uniform initialization
        for quantile, module in self.linear.items():
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        """forward function defines the computation performed at every call

        Parameters:
        :param x: input to the model

        Returns:
        tuple containing containing the outputs for each quantile layer
        """
        # you can simply create a list of outputs in the order of the quantiles above but before
        # returning this list cast it to a tuple.

        out, (hn, cn) = self.lstm(x)

        out_last_step = out[:, -1, :]

        out_fc = self.fc(out_last_step)

        out_list = []
        for quantile, module in self.linear.items():
            out_temp = module(out_fc)

            out_list.append(out_temp)

        out_list = tuple(out_list)
        return out_list


def train_model(train_loader):
    """Function that trains the LSTM_Model defined above.

    Parameters:
    train_loader: PyTorch DataLoader object for the training data

    Returns:
    Trained LSTM_Model object
    """
    torch.manual_seed(18)

    lstm_model = LSTM_Model()
    lstm_model = lstm_model.to(device)

    criteria = []
    for i in range(len(QUANTILES)):
        criteria.append(QuantileLoss(i))
    optimizer = torch.optim.Adam(lstm_model.parameters())

    for epoch in range(NUM_EPOCHS):
        lstm_model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = lstm_model.forward(inputs)
            loss = 0.0
            for i in range(len(QUANTILES)):
                l_i = criteria[i](outputs[i], targets)
                loss += l_i
            loss.backward()
            optimizer.step()
    return lstm_model


def eval_model(lstm_model, test_loader, y_delta_cases, test_info):
    """Function that evaluates the trained lstm_model.

    Parameters:
    :param lstm_model: trained LSTM_Model object
    :param test_loader: PyTorch DataLoader for test data
    :param y_delta_cases: number of actuak new COVID-19 cases
    :param test_info: county information

    Returns:
    A Pandas Dataframe containing the results from model evaluation.

    """
    lstm_model.eval()
    with torch.no_grad():
        out_arr_pred = None  # output array for the predicted labels & error
        out_arr_err = None
        out_arr_lbl = None  # smoothened data labels (for reference)

        for inputs, targets in test_loader:
            inputs = inputs.to(device)  # BATCH_SIZEx(TEMPORAL_LAG+1)x1
            y_pred = lstm_model.forward(inputs)  # len(QUANTILES)xBATCH_SIZE

            arr_pred = None
            arr_err = None
            for i in range(len(QUANTILES)):  # calculate the error for each quantile
                y_hat = y_pred[i].cpu().data.numpy()
                if i == 0:
                    arr_pred = y_hat
                    arr_err = targets - y_hat
                else:
                    arr_pred = np.hstack((arr_pred, y_hat))
                    arr_err = np.hstack((arr_err, (targets - y_hat)))

            if out_arr_pred is None:
                out_arr_pred = arr_pred.copy()
                out_arr_err = arr_err.copy()
                out_arr_lbl = targets.cpu().data.numpy()
            else:
                out_arr_pred = np.vstack((out_arr_pred, arr_pred.copy()))
                out_arr_err = np.vstack((out_arr_err, arr_err.copy()))
                out_arr_lbl = np.vstack((out_arr_lbl, targets.cpu().data.numpy()))

        out_arr_pred = np.hstack((test_info.reshape((-1, 1)), out_arr_pred))
        out_arr_err = np.hstack((test_info.reshape((-1, 1)), out_arr_err))

        col_names_pred = ["GEOID"]
        col_names_err = ["GEOID"]

        for quantile in QUANTILES:
            col_names_pred.append("q_" + str(int(quantile * 1000)) + "_pred")
            col_names_err.append("q_" + str(int(quantile * 1000)) + "_err")

        # create results dataframe
        results_pred_df = pd.DataFrame(data=out_arr_pred, columns=col_names_pred)
        results_pred_df = results_pred_df.astype({"GEOID": int})
        results_err_df = pd.DataFrame(data=out_arr_err, columns=col_names_err)
        results_err_df = results_err_df.astype({"GEOID": int})
        results_df = results_pred_df.merge(results_err_df, on="GEOID")
        results_df["y_label"] = out_arr_lbl
        results_df["y_delta_cases"] = y_delta_cases

        counties_df = pd.read_csv("data/counties.csv")
        cols = ["FIPS", "POPULATION"]
        counties_df = counties_df[cols]

        results_df = results_df.merge(
            counties_df, right_on="FIPS", left_on="GEOID", how="left"
        )
        for quantile in QUANTILES:
            quantile = str(int(quantile * 1000))

            results_df["q_" + quantile + "_pred_transform"] = (
                np.exp(results_df["q_" + quantile + "_pred"]) - 1
            )
            results_df["q_" + quantile + "_pred_cases"] = (
                results_df["q_" + quantile + "_pred_transform"]
                * results_df["POPULATION"]
                / 10000
            )

        results_df["y_lbl_transform"] = np.exp(results_df["y_label"]) - 1
        results_df["y_label_transformed"] = (
            results_df["y_lbl_transform"] * results_df["POPULATION"] / 10000
        )

        # error = absolute difference between the predicted delta cases at the 50th quantile and the true delta cases
        results_df["y_q500_err"] = abs(
            results_df["y_delta_cases"] - results_df["q_500_pred_cases"]
        )
        results_df.drop(cols, axis=1, inplace=True)

    return results_df


def eval_results(
    burnin_weeks,
    num_of_biweekly_intervals,
    state,
    county,
    temporal_lag,
    forecast_horizon,
):
    """Function to train and evaluate results using train_model and eval_model functions respectively.

    Parameters:
    :param burnin_weeks: first forecast date is after INITIAL_DATE + burnin_weeks
    :param num_of_biweekly_intervals: number of biweekly intervals to predict
    :param state: US state
    :param county: County in the state (can be None)
    :param temporal_lag: temporal lag to use when getting COVID data
    :param forecast_horizon: forecast horizon to use when getting COVID data

    Returns:
    err_results_df: Pandas dataframe containing error in predicting COVID cases
    final_results_state: Final results for the given state
    final_results_county: Final results for the given county in the state
    """

    initial_date = date(
        int(dc.INITIAL_DATE.split("-")[0]),
        int(dc.INITIAL_DATE.split("-")[1]),
        int(dc.INITIAL_DATE.split("-")[2]),
    )

    first_forecast_date = initial_date + timedelta(weeks=burnin_weeks)
    final_results_state = []
    final_results_county = []

    print(f"Training for {num_of_biweekly_intervals} intervals...")
    for i in range(num_of_biweekly_intervals):
        data = dc.CUBData()

        forecast_date = first_forecast_date + timedelta(weeks=i * 2)

        df = data.get_data(forecast_date.strftime("%Y-%m-%d"))

        wdf = df[df["STATE"] == state]  # filter the records for the state

        # Now split the data into train, test and prepare for modeling
        (
            X_train_ts,
            y_train,
            X_test_ts,
            y_test,
            y_delta_cases,
            test_info,
        ) = dc.train_test_splits(
            forecast_date.strftime("%Y-%m-%d"),
            wdf,
            temporal_lag=temporal_lag,
            forecast_horizon=forecast_horizon,
        )

        train_loader = make_loader(X_train_ts, y_train)
        test_loader = make_loader(X_test_ts, y_test, shuffle=False)

        if i % 5 == 0 or i == num_of_biweekly_intervals - 1:
            print(
                f"Training interval {i+1} for {forecast_date} with {X_train_ts.shape[0]} train samples (state)"
            )

        lstm_model = train_model(train_loader)
        results_df = eval_model(lstm_model, test_loader, y_delta_cases, test_info)
        final_results_state.append(results_df)

        if county is not None:
            wdf = df[
                (df["STATE"] == state) & (df["NAME"] == county)
            ]  # filter the records for the county in the state
            if i == 0:
                county_geoid = wdf["GEOID"].values[0]
            # Now split the data into train, test and prepare for modeling
            (
                X_train_ts,
                y_train,
                X_test_ts,
                y_test,
                y_delta_cases,
                test_info,
            ) = dc.train_test_splits(
                forecast_date.strftime("%Y-%m-%d"),
                wdf,
                temporal_lag=dc.TEMPORAL_LAG,
                forecast_horizon=dc.FORECAST_HORIZON,
            )

            train_loader = make_loader(X_train_ts, y_train)
            test_loader = make_loader(X_test_ts, y_test, shuffle=False)

            if i % 5 == 0 or i == num_of_biweekly_intervals - 1:
                print(
                    f"Training interval {i+1} for {forecast_date} with {X_train_ts.shape[0]} train samples (county)"
                )

            lstm_model = train_model(train_loader)
            results_df = eval_model(lstm_model, test_loader, y_delta_cases, test_info)
            final_results_county.append(results_df)

    # find the error between predicted NEW cases (50th quantile) and actual NEW cases
    err_state = []
    err_county = []
    forecast_dates = []

    for i in range(num_of_biweekly_intervals):
        df_state = final_results_state[i]
        forecast_dates.append(first_forecast_date + timedelta(weeks=i * 2))

        if county is not None:
            df_state_county = df_state[df_state["GEOID"] == county_geoid]
            err_state.append(df_state_county["y_q500_err"].values[0])

            df_county = final_results_county[i]
            err_county.append(df_county["y_q500_err"].values[0])
        else:
            err_state.append(
                df_state["y_q500_err"].mean()
            )  # take the mean error for all counties in the state

    if county is None:
        err_results_df = pd.DataFrame(
            {"error(State)": err_state, "forecast_dates": forecast_dates}
        )
    else:
        err_results_df = pd.DataFrame(
            {
                "error(State)": err_state,
                "error(County)": err_county,
                "forecast_dates": forecast_dates,
            }
        )
    return err_results_df, final_results_state, final_results_county
