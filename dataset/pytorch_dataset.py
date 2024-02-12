from gluonts.dataset.field_names import FieldName
from gluonts.transform import AddObservedValuesIndicator, InstanceSplitter, ExpectedNumInstanceSampler, TestSplitSampler
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify


def get_context_data_loader(train_ts, val_ts, test_ts, train_val_ts, freq, context_length, prediction_length, 
                    batch_size, num_batches_per_epoch):
    mask_unobserved = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )

    training_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=prediction_length,
        ),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )

    prediction_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=TestSplitSampler(),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )


    train_data_loader = TrainDataLoader(
        # We cache the dataset, to make training faster
        Cached(train_ts),
        batch_size=batch_size,
        stack_fn=batchify,
        transform=mask_unobserved + training_splitter,
        num_batches_per_epoch=num_batches_per_epoch,
    )

    train_val_data_loader = TrainDataLoader(
        # We cache the dataset, to make training faster
        Cached(train_val_ts),
        batch_size=batch_size,
        stack_fn=batchify,
        transform=mask_unobserved + training_splitter,
        num_batches_per_epoch=num_batches_per_epoch,
    )

    val_data_loader = ValidationDataLoader(
        # We cache the dataset, to make training faster
        Cached(val_ts),
        batch_size=batch_size,
        stack_fn=batchify,
        transform=mask_unobserved + training_splitter
    )

    test_data_loader = ValidationDataLoader(
        # We cache the dataset, to make training faster
        Cached(test_ts),
        batch_size=batch_size,
        stack_fn=batchify,
        transform=mask_unobserved + prediction_splitter
    )
    return train_data_loader,val_data_loader, test_data_loader, train_val_data_loader, mask_unobserved + prediction_splitter
