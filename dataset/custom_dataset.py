from gluonts.dataset.artificial import ComplexSeasonalTimeSeries
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import numpy as np
import pdb
from numpy import distutils
import distutils
import pandas as pd
from datetime import datetime

from utils.tools import get_seasonality_frequency
from evaluate.evaluate_forecast import mase_as_monash
from dataset.val_strategy import get_validation_split


class Dataset():
    def __init__(self, train_ds, test_ds) -> None:
        self.train = train_ds
        self.test = test_ds

def artificial():
    artificial_dataset = ComplexSeasonalTimeSeries(
        num_series=10,
        prediction_length=21,
        freq_str="H",
        length_low=30,
        length_high=200,
        min_val=-10000,
        max_val=10000,
        is_integer=False,
        proportion_missing_values=0,
        is_noise=True,
        is_scale=True,
        percentage_unique_timestamps=1,
        is_out_of_bounds_date=True,
    )
    train_ds = ListDataset(
    artificial_dataset.train,
    freq=artificial_dataset.metadata.freq
    )

    test_ds = ListDataset(
        artificial_dataset.test,
        freq=artificial_dataset.metadata.freq
    )
    return train_ds, test_ds, artificial_dataset


class Monash_Dataset():
    def __init__(self, args) -> None:
        self.full_file_path_and_name = args.dataset_folder+args.dataset 
        self.replace_missing_vals_with = args.replace_missing_vals_with
        self.value_column_name = args.value_col_name
        self.args = args # get default if some args missing
        self.frequency = None
        self.forecast_horizon = None
        self.contain_missing_values = None
        self.contain_equal_length = None
        self.final_forecasts_array = None
        self.convert_tsf_to_dataframe()

    def convert_tsf_to_dataframe(self):
        col_names = []
        col_types = []
        all_data = {}
        line_count = 0
        found_data_tag = False
        found_data_section = False
        started_reading_data_section = False

        with open(self.full_file_path_and_name, 'r', encoding='cp1252') as file:
            for line in file:
                # Strip white space from start/end of line
                line = line.strip()

                if line:
                    if line.startswith("@"): # Read meta-data
                        if not line.startswith("@data"):
                            line_content = line.split(" ")
                            if line.startswith("@attribute"):
                                if (len(line_content) != 3):  # Attributes have both name and type
                                    raise Exception("Invalid meta-data specification.")

                                col_names.append(line_content[1])
                                col_types.append(line_content[2])
                            else:
                                if len(line_content) != 2:  # Other meta-data have only values
                                    raise Exception("Invalid meta-data specification.")

                                if line.startswith("@frequency"):
                                    self.frequency = line_content[1]
                                elif line.startswith("@horizon"):
                                    self.forecast_horizon = int(line_content[1])
                                elif line.startswith("@missing"):
                                    self.contain_missing_values = bool(distutils.util.strtobool(line_content[1]))
                                elif line.startswith("@equallength"):
                                    self.contain_equal_length = bool(distutils.util.strtobool(line_content[1]))

                        else:
                            if len(col_names) == 0:
                                raise Exception("Missing attribute section. Attribute section must come before data.")

                            found_data_tag = True
                    elif not line.startswith("#"):
                        if len(col_names) == 0:
                            raise Exception("Missing attribute section. Attribute section must come before data.")
                        elif not found_data_tag:
                            raise Exception("Missing @data tag.")
                        else:
                            if not started_reading_data_section:
                                started_reading_data_section = True
                                found_data_section = True
                                all_series = []

                                for col in col_names:
                                    all_data[col] = []

                            full_info = line.split(":")

                            if len(full_info) != (len(col_names) + 1):
                                raise Exception("Missing attributes/values in series.")

                            series = full_info[len(full_info) - 1]
                            series = series.split(",")

                            if(len(series) == 0):
                                raise Exception("A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol")

                            numeric_series = []
                            for val in series:
                                if val == "?":
                                    numeric_series.append(self.replace_missing_vals_with)
                                else:
                                    numeric_series.append(float(val))

                            if (numeric_series.count(self.replace_missing_vals_with) == len(numeric_series)):
                                raise Exception("All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series.")

                            all_series.append(pd.Series(numeric_series).array)

                            for i in range(len(col_names)):
                                att_val = None
                                if col_types[i] == "numeric":
                                    att_val = int(full_info[i])
                                elif col_types[i] == "string":
                                    att_val = str(full_info[i])
                                elif col_types[i] == "date":
                                    att_val = datetime.strptime(full_info[i], '%Y-%m-%d %H-%M-%S')
                                else:
                                    raise Exception("Invalid attribute type.") # Currently, the code supports only numeric, string and date types. Extend this as required.

                                if(att_val == None):
                                    raise Exception("Invalid attribute value.")
                                else:
                                    all_data[col_names[i]].append(att_val)

                    line_count = line_count + 1

            if line_count == 0:
                raise Exception("Empty file.")
            if len(col_names) == 0:
                raise Exception("Missing attribute section.")
            if not found_data_section:
                raise Exception("Missing series information under data section.")

            all_data[self.value_column_name] = all_series
            self.df = pd.DataFrame(all_data)

            ################################ defaulting if unavailable ##################################

            # set frequency to default if missing
            if self.frequency is not None:
                self.freq, self.seasonality = get_seasonality_frequency(self.frequency)
            else:
                self.freq = "1Y"
                self.seasonality = 1

            if isinstance(self.seasonality, list):
                self.seasonality = min(self.seasonality) # Use to calculate MASE

            # If the forecast horizon is not given within the .tsf file, 
            # then it should be provided as a function input
            if self.forecast_horizon is None:
                if self.args.external_forecast_horizon is None:
                    raise Exception("Please provide the required forecast horizon")
                else:
                    self.forecast_horizon = self.args.external_forecast_horizon
    
    def get_monash_mase(self):
        outsample_array = np.array(self.test_series_list)
        insample_array = np.array(self.train_series_list)
        forecast = np.array(self.final_forecasts_array)
        return mase_as_monash(forecast, insample_array, outsample_array, self.seasonality)

    def get_monash_df(self):
        return self.df

    def get_monash_dataset(self):
        self.train_series_list = []
        self.test_series_list = []
        self.val_series_list = []
        self.train_val_series_list = []

        self.train_series_full_list = []
        self.test_series_full_list = []
        self.val_series_full_list = []
        self.train_val_series_full_list = []

        for index, row in self.df.iterrows():
            if self.args.time_col_name in self.df.columns:
                train_start_time = row[self.args.time_col_name]
            else:
                train_start_time = datetime.strptime('1900-01-01 00-00-00', '%Y-%m-%d %H-%M-%S') # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False

            series_data = row[self.args.value_col_name]

            # Creating training, val and test series. Test series will be only used during evaluation
            train_series_data = series_data[:len(series_data) - self.forecast_horizon]
            test_series_data = series_data[(len(series_data) - self.forecast_horizon) : len(series_data)]

            train_series_data, val_series_data, val_train_data = get_validation_split(series_data, train_series_data, train_start_time, 
                                                                    self.freq, self.forecast_horizon, self.args)

            train_val_series_data = train_series_data[self.forecast_horizon:]
            # series list
            self.train_series_list.append(train_series_data)
            self.test_series_list.append(test_series_data)
            self.train_val_series_list.append(train_val_series_data)

            # pdb.set_trace()
            # series full list
            self.train_series_full_list.append({
                FieldName.TARGET: train_series_data,
                FieldName.START: pd.Timestamp(train_start_time, freq=self.freq)
            })

            self.test_series_full_list.append({
                FieldName.TARGET: series_data,
                FieldName.START: pd.Timestamp(train_start_time, freq=self.freq)
            })

            self.train_val_series_full_list.append(
                {
                FieldName.TARGET: train_val_series_data,
                FieldName.START: pd.Timestamp(train_start_time, freq=self.freq)
            }
            )

            # define the validation split
            if val_train_data is not None:
                self.val_series_list.append(val_series_data) # uses only the val part
                self.val_series_full_list.append({
                    FieldName.TARGET: val_train_data,
                    FieldName.START: pd.Timestamp(train_start_time, freq=self.freq)
                })
                
        train_ds = ListDataset(self.train_series_full_list, freq=self.freq)
        test_ds = ListDataset(self.test_series_full_list, freq=self.freq)
        if len(self.val_series_list) > 1: 
            val_ds = ListDataset(self.val_series_full_list, freq=self.freq)
            train_val_ds = ListDataset(self.train_series_full_list + self.val_series_full_list, freq=self.freq)
        else:
            val_ds = None

        data_args={'freq': self.freq,
                'context_length' : self.args.context_length,
                'prediction_length': self.forecast_horizon}


        return train_ds, test_ds, val_ds, train_val_ds, data_args