from ml.data import process_data
from ml.model import inference, compute_model_metrics

def edu_slice_metrics(model, encoder, lb, test_data, cat_features):
    """ Function for calculating performance metrics on edu-slices census dataset."""
    with open('../metrics/slice_output.txt', mode='w+') as so_file:
        so_file.write("Model metrics per EDU class slice\n\n")
        for cls in test_data["education"].unique():
            df_temp = test_data[test_data["education"] == cls]
            X_sliced, y_sliced, encoder, lb = process_data(
                df_temp, categorical_features=cat_features, label="salary", training=False,
                encoder=encoder, lb=lb
            )
            preds = inference(model, X_sliced)
            precision, recall, fbeta = compute_model_metrics(y_sliced, preds)

            so_file.write(f"Class: {cls}")
            so_file.write('\n')
            so_file.write(f"Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")
            so_file.write('\n\n')
