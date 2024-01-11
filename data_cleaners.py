import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from sklearn.preprocessing import MinMaxScaler

TEMPORAL_LAG = 9
FORECAST_HORIZON = 1
INITIAL_DATE = "2020-04-05"  # Starting date as specified by the authors


## Loads in dataset from JHU
class JHUCoreData:
    ##dates must be before x-x-2022 (depends on google data)

    def __init__(self):
        ##The core data is from the covid repository
        ##these fileswere downloaded on March 24, 2022
        ##https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/

        # self.date = date
        self.us_cases = self.get_us_case_data()
        self.us_deaths = self.get_us_death_data()
        self.global_cases = self.get_global_case_data()
        self.global_deaths = self.get_global_death_data()
        self.global_recoveries = self.get_global_recovery_data()
        self.latest_date = None

    def get_us_case_data(self):
        # keep it safe instead of going to website
        return pd.read_csv("assets/time_series_covid19_confirmed_US.csv")

    def get_us_death_data(self):
        # keep it safe instead of going to website
        return pd.read_csv("assets/time_series_covid19_deaths_US.csv")

    def get_global_case_data(self):
        # keep it safe instead of going to website
        return pd.read_csv("assets/time_series_covid19_confirmed_global.csv")

    def get_global_death_data(self):
        # keep it safe instead of going to website
        return pd.read_csv("assets/time_series_covid19_deaths_global.csv")

    def get_global_recovery_data(self):
        # keep it safe instead of going to website
        return pd.read_csv("assets/time_series_covid19_recovered_global.csv")


class CUBData(JHUCoreData):
    def __init__(self):
        super().__init__()

    def clean_JHU_cases(self):
        jh_covid_df = self.us_cases
        cols_to_drp = ["UID", "iso2", "iso3", "code3", "Country_Region", "Lat", "Long_"]

        # current cols:
        #
        # 'FIPS',
        # 'Admin2',
        # 'Province_State',
        # 'Combined_Key'
        #

        # preprocessing JH COVID data
        jh_covid_df.dropna(axis=0, how="any", inplace=True)
        jh_covid_df["FIPS"] = jh_covid_df["FIPS"].astype("int64")
        jh_covid_df.drop(columns=cols_to_drp, inplace=True)

        # Important: check to see the column index is adherent to the imported df
        first_date = datetime.strptime(jh_covid_df.columns[4], "%m/%d/%y").date()
        last_date = datetime.strptime(jh_covid_df.columns[-1], "%m/%d/%y").date()

        current_date = last_date
        previous_date = last_date - timedelta(days=1)

        conversion_format = "%-m/%-d/%y"
        while current_date > first_date:
            # For unix, replace # with - in the time format
            current_col = current_date.strftime(conversion_format)
            previous_col = previous_date.strftime(conversion_format)
            jh_covid_df[previous_col] = np.where(
                jh_covid_df[previous_col] > jh_covid_df[current_col],
                jh_covid_df[current_col],
                jh_covid_df[previous_col],
            )
            current_date = current_date - timedelta(days=1)
            previous_date = previous_date - timedelta(days=1)

        # if smooth:
        smoothed = jh_covid_df.copy()
        smoothed.iloc[:, 4:] = (
            jh_covid_df.iloc[:, 4:].rolling(7, min_periods=1, axis=1).mean()
        )

        return jh_covid_df, smoothed  # daily covid cases across US (original, smoothed)

    def combine_data(
        self, forecast_date, contiguous_counties, covid_df, covid_df_non_smooth
    ):
        assert forecast_date.weekday() == 6  ## forecast date should be a Sunday

        T_end = forecast_date - timedelta(days=1)  # Saturday
        T_start = T_end - timedelta(weeks=1)  # Saturday

        dates = [T_end, T_start]
        dates_case_str = [item.strftime("%-m/%-d/%y") for item in dates]

        jh_df = covid_df[["FIPS", *dates_case_str]]

        jh_df_non_smooth = covid_df_non_smooth[["FIPS", *dates_case_str]]

        temp = contiguous_counties

        temp["current_date"] = forecast_date.strftime("%Y-%m-%d")

        # add covid-related columns
        temp = temp.merge(jh_df, on="FIPS", how="left")  # smoothened data

        # rate (and log rate) of infection for the last 1 week using smoothened data (NOTE: smoothened data)
        temp["DELTA_INC_RATE"] = (
            (temp[dates_case_str[0]] - temp[dates_case_str[1]])
            / temp["POPULATION"]
            * 10000
        )
        temp["LOG_DELTA_INC_RATE"] = np.log(temp["DELTA_INC_RATE"] + 1)

        # drop unnecessary columns
        temp.drop(columns="DELTA_INC_RATE", inplace=True)
        temp.drop(
            columns=dates_case_str, inplace=True
        )  # drop both columns that had the smoothened data

        output = temp.merge(
            jh_df_non_smooth, on="FIPS", how="left"
        )  # add the two columns that have the actual case count
        output["DELTA_CASE_JH"] = (
            output[dates_case_str[0]] - output[dates_case_str[1]]
        )  # actual case count for the previous week

        output.drop(
            columns=dates_case_str, inplace=True
        )  # drop the two columns that have the actual case count
        # output has all columns from counties, current-date, DELTA_INC_RATE, DELTA_CASE_JH
        return output

    def get_data(self, current_date):
        counties_df = pd.read_csv("assets/counties.csv")

        covid_df_non_smooth, covid_df = self.clean_JHU_cases()

        covid_df_contiguous = covid_df[
            covid_df["FIPS"].isin(counties_df["FIPS"])
        ].copy()

        covid_df_non_smooth_contiguous = covid_df_non_smooth[
            covid_df_non_smooth["FIPS"].isin(counties_df["FIPS"])
        ].copy()

        df_lagged_list = []

        forecast_date = date(
            int(current_date.split("-")[0]),
            int(current_date.split("-")[1]),
            int(current_date.split("-")[2]),
        )
        initial_date = date(
            int(INITIAL_DATE.split("-")[0]),
            int(INITIAL_DATE.split("-")[1]),
            int(INITIAL_DATE.split("-")[2]),
        )

        # NOTE: Add the next week in order to get y_test that corresponds to X_test_ts below
        forecast_date = forecast_date + timedelta(weeks=1)

        while forecast_date >= initial_date:
            df_week = self.combine_data(
                forecast_date,
                counties_df,
                covid_df_contiguous,
                covid_df_non_smooth_contiguous,
            )
            df_lagged_list.append(df_week)

            forecast_date -= timedelta(days=7)

        df_lagged = pd.concat(df_lagged_list, axis=0)

        cols_to_save = [
            "GEOID",
            "NAME",
            "STATE",
            "current_date",
            "LOG_DELTA_INC_RATE",
            "DELTA_CASE_JH",
        ]

        df_lagged = df_lagged[cols_to_save]
        df_lagged.sort_values(by=["GEOID", "current_date"], inplace=True)
        return df_lagged


class covidData:
    def __init__(
        self,
        current_date,
        init_df,
        temporal_lag=TEMPORAL_LAG,
        forecast_horizon=FORECAST_HORIZON,
    ):
        self.current_date = current_date
        self.init_df = init_df
        self.temporal_lag = temporal_lag
        self.forecast_horizon = forecast_horizon
        self.all_ts_data = None
        self.all_targets = None

    def load_dataframes(self, print_info=False):
        # time series data
        timeseries_df = self.init_df

        # preprocess ts data
        timeseries_df = timeseries_df.replace([np.inf, -np.inf], np.NaN)
        na_cols = timeseries_df.columns[timeseries_df.isna().any()].tolist()
        for col in na_cols:
            timeseries_df[col] = timeseries_df.groupby(["current_date", "GEOID"])[
                col
            ].transform(lambda x: x.fillna(x.mean()))

        # columns: 'GEOID', 'NAME', 'STATE', 'current_date', 'LOG_DELTA_INC_RATE', 'DELTA_CASE_JH'
        target_cols = [0, 3, 4, 5]
        self.all_targets = timeseries_df.iloc[
            :, target_cols
        ]  # 'GEOID', 'current_date', 'LOG_DELTA_INC_RATE', 'DELTA_CASE_JH'
        if print_info:
            print("All TS shape", timeseries_df.shape)

        cols = [0, 1, 2, 3, 4]
        timeseries_df = timeseries_df.iloc[
            :, cols
        ]  # 'GEOID', 'NAME', 'STATE', 'current_date', 'LOG_DELTA_INC_RATE'
        self.all_ts_data = timeseries_df

    def create_train_test_datasets(self, print_info=False):
        temporal_lag = self.temporal_lag
        current_date = self.current_date
        forecast_horizon = self.forecast_horizon

        # create train/test sets
        geoid_list = np.unique(self.all_ts_data.GEOID)
        dates_recorded = np.unique(self.all_ts_data.current_date)
        no_timestamps = len(dates_recorded)
        if print_info:
            print("No timestamps: ", no_timestamps)

        X_train_ts, X_train_se, y_train = [], [], []
        X_test_ts, X_test_se, y_test, y_delta_cases = [], [], [], []
        test_info = []

        for i, id in enumerate(geoid_list):
            county_ts = self.all_ts_data[self.all_ts_data["GEOID"] == id].reset_index(
                drop=True
            )
            county_ts["current_date"] = pd.to_datetime(
                county_ts["current_date"], format="%Y-%m-%d"
            )
            county_ts.sort_values(by="current_date", inplace=True, axis=0)
            county_ts.reset_index(inplace=True)  # NOTE: add a new column "index"
            current_index = county_ts.index[county_ts["current_date"] == current_date][
                0
            ]

            if print_info:
                if i == 0:
                    print("Current index: ", current_index)

            county_ts = county_ts.values[:, 5]  # LOG_DELTA_INC_RATE
            county_ts = county_ts.reshape((-1, 1))

            county_test_info = self.all_targets[self.all_targets["GEOID"] == id].values[
                :, 0
            ]  # GEOID

            if print_info:
                if i == 0:
                    print("County TS data shape:", county_ts.shape)
                    # print("County target shape:", county_targets.shape)
                    print("County Test Info data shape:", county_test_info.shape)

            y_index = 0
            while y_index + temporal_lag + forecast_horizon <= current_index:
                ts_instance = county_ts[
                    y_index : y_index + temporal_lag + 1, :
                ]  # 0,1,2,..,8,9
                X_train_ts.append(ts_instance)
                y_train.append(
                    county_ts[y_index + temporal_lag + forecast_horizon]
                )  # 10 (when forecast_horizon == 1)
                if print_info:
                    if i == 0:
                        print("Train X:", ts_instance)
                        print(
                            "Train Target: ",
                            county_ts[y_index + temporal_lag + forecast_horizon],
                        )

                y_index += 1

            X_test_ts.append(
                county_ts[(current_index - temporal_lag) : (current_index + 1), :]
            )

            # append y_test
            y_test.append(
                county_ts[-1, :]
            )  # last row has the LOG_DELTA_INC_RATE for the target week (smoothened data)
            y_delta_cases.append(
                self.all_targets[self.all_targets["GEOID"] == id].values[-1:, 3]
            )  # DELTA_CASE_JH

            test_info.append(county_test_info[current_index])
            if print_info:
                if i == 0:
                    print(
                        "Test X:",
                        county_ts[
                            (current_index - temporal_lag) : (current_index + 1), :
                        ],
                    )

        data_dict = {
            "X_train_ts": np.array(X_train_ts, dtype="float32"),
            "y_train": np.array(y_train, dtype="float32"),
            "X_test_ts": np.array(X_test_ts, dtype="float32"),
            "y_test": np.array(y_test, dtype="float32"),  # smoothened data
            "y_delta_cases": np.array(y_delta_cases, dtype="int"),  # actual data
            "test_info": np.array(test_info),
        }

        return data_dict


def train_test_splits(
    current_date,
    init_df,
    temporal_lag=TEMPORAL_LAG,
    forecast_horizon=FORECAST_HORIZON,
    print_info=False,
):
    # create data object from csv
    data = covidData(
        current_date=current_date,
        init_df=init_df,
        temporal_lag=temporal_lag,
        forecast_horizon=forecast_horizon,
    )

    # preprocess
    data.load_dataframes(print_info=False)

    # create train-test split
    data_dictionary = data.create_train_test_datasets(print_info=False)

    X_train_ts = data_dictionary["X_train_ts"]
    y_train = data_dictionary["y_train"]
    X_test_ts = data_dictionary["X_test_ts"]
    y_test = data_dictionary["y_test"]  # log transformed
    y_delta_cases = data_dictionary["y_delta_cases"]  # actual

    test_info = data_dictionary["test_info"]

    n_temporal_fts = X_train_ts[0].shape[-1]
    if print_info:
        print("Number of temporal features:", n_temporal_fts)

    n_train = X_train_ts.shape[0]
    n_test = X_test_ts.shape[0]
    if print_info:
        print(
            "Number of training instances:",
            n_train,
            ", Number of test instances:",
            n_test,
        )

    for i, _ in enumerate(X_train_ts):
        X_train_ts[i] = np.asarray(X_train_ts[i]).astype("float32")
    for i, _ in enumerate(y_train):
        y_train[i] = np.asarray(y_train[i]).astype("float32")
    for i, _ in enumerate(X_test_ts):
        X_test_ts[i] = np.asarray(X_test_ts[i]).astype("float32")

    return X_train_ts, y_train, X_test_ts, y_test, y_delta_cases, test_info
