
from utils.tools import return_new_time_by_addition

def get_validation_split(series_data, train_series_data, train_start_time, freq, forecast_horizon, args):
    if "oos" in args.validation_strategy:
        # remove one more of the forecast horizon as validation
        train_series_data = series_data[:-2 * forecast_horizon]
        val_series_data = series_data[len(train_series_data):-forecast_horizon]
        
        # like the test the validation should have the train data as well
        val_train_data = series_data[:-forecast_horizon] 
        
        # get val start time by setting offset as forecast horizon
        # val_start_time = return_new_time_by_addition(train_start_time, freq, forecast_horizon)

    # if args.validation_strategy=='re_oos':

    #     # remove one more of the forecast horizon as validation
    #     train_series_data = series_data[:-2 * forecast_horizon]
    #     val_series_data = series_data[len(train_series_data):-forecast_horizon]
        
    #     # like the test the validation should have the train data as well
    #     val_train_data = series_data[:-forecast_horizon] 
        
    #     # need the train_validation data to retrain
    #     train_val_data = series_data[:-forecast_horizon]
    

    else:
        val_series_data = None
        val_train_data = None

    return train_series_data, val_series_data, val_train_data#, val_start_time