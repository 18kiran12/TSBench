
import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas

def plot_train_test_split(train_ds, val_ds, test_ds, args):
    """
    plot the entire train test split
    """
    entry_train = next(iter(train_ds))
    train_series = to_pandas(entry_train)
    train_series.plot()
    plt.axvline(train_series.index[-1], color='r') # end of train dataset

    if val_ds is not None:
        entry_val = next(iter(val_ds))
        val_series = to_pandas(entry_val)
        val_series.plot()
        plt.axvline(val_series.index[-1], color='b') # end of val dataset

    entry_test = next(iter(test_ds))
    test_series = to_pandas(entry_test)
    test_series.plot()

    plt.grid(which="both")
    plt.legend(["validation end", "train end"], loc="upper left")
    plt.savefig(args.run_folder +"/plots/" + args.run_full_name + "~datasplit.png")

def plot_prob_forecasts(tss, forecasts, args):
    """
    plot the first forecast window
    """
    ts_entry = tss[1]
    print("ts_entry :", len(ts_entry))
    forecast_entry = forecasts[1]
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig(args.run_folder +"/plots/" + args.run_full_name + "~forecast.png")