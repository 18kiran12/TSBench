import numpy as np
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import pytorch_lightning

def mase_as_monash(final_forecasts_array, insample_array, outsample_array, seasonality):#seasonality or frequency
    mase_per_series = []
    for i in range(final_forecasts_array.shape[0]):
        mase = np.mean(np.abs(final_forecasts_array[i] - outsample_array[i])) / np.mean(np.abs(insample_array[i][:-seasonality] - insample_array[i][seasonality:]))
        # set seasonality as 1 if the row has no mase (example: seasonality is larger than series length)
        if np.isnan(mase):
            mase = np.mean(np.abs(final_forecasts_array[i] - outsample_array[i])) / np.mean(np.abs(insample_array[i][:-1] - insample_array[i][1:]))
        if mase!=np.inf:
            mase_per_series.append(mase)
    return np.mean(mase_per_series),len(mase_per_series)


def get_agg_metrics(ds, dummy_predictor, monash, integer_conversion=False, pytorch =False, num_samples=100,):
    forecast_it, ts_it = make_evaluation_predictions(
                        dataset=ds,  # test dataset
                        predictor=dummy_predictor,  # predictor
                        num_samples=100,  # number of sample paths we want for evaluation
                    )
    if pytorch:
        forecasts = list(f.to_sample_forecast() for f in forecast_it)
    else:
        forecasts = list(forecast_it)
    tss = list(ts_it)

    final_forecasts = []
    for f in forecasts:
        final_forecasts.append(f.median)
    
    if integer_conversion:
        final_forecasts = np.round(final_forecasts)

    evaluator = Evaluator(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    monash.final_forecasts_array = final_forecasts
    agg_metrics["MASE_monash"], num_series_used_for_mase = monash.get_monash_mase()
    
    return agg_metrics, final_forecasts