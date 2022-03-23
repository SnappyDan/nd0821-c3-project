# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Daniel Slepov created the model. It is logistic regression using the default hyperparameters in scikit-learn 1.0.2.
Actually it is more like a mock model that is used just as a basis for demonstration of different MLOps principles and deployment processes.

## Intended Use

This model could be used to predict the salery of a given person based off a handful of attributes. The users are Public and HR services specialists.

## Training Data

The raw data was obtained from the UCI Machine Learning Repository. More datails concering the dataset can be found here (https://archive.ics.uci.edu/ml/datasets/census+income). The target class is "salary"

The original dataset was quite messy, so the first step was to clean it by removing spaces from the data.
The data has 32562 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the categorical features and a label binarizer was used on the labels.

## Evaluation Data
The model was evaluated on 20% of the initial dataset

## Metrics

The model was evaluated using Precision, Recall and Fbeta scores.
Precision: 0.7285223367697594, Recall: 0.2698917886696372, Fbeta: 0.393869019972132

Metrics were also calculated per different EDU class slices. For some cases the results are not so bad:

Class: Bachelors
Precision: 0.8198757763975155, Recall: 0.29333333333333333, Fbeta: 0.43207855973813425

Class: Masters
Precision: 0.8857142857142857, Recall: 0.2995169082125604, Fbeta: 0.4476534296028881

Class: Assoc-acdm
Precision: 0.9285714285714286, Recall: 0.2765957446808511, Fbeta: 0.42622950819672134

But for several classes the model quality is significantly diminished:

Class: 7th-8th
Precision: 0.25, Recall: 0.16666666666666666, Fbeta: 0.2

Class: 11th
Precision: 0.7, Recall: 0.6363636363636364, Fbeta: 0.6666666666666666

Class: Assoc-voc
Precision: 0.5357142857142857, Recall: 0.23809523809523808, Fbeta: 0.3296703296703297

Details can be found in (model/slice_output.txt).

## Ethical Considerations

According to the metrics calculated on different slices based on the "education" feature - model shows much poor results on  This implies an unfairness in the underlying data/model which can lead to potentical ethical issues.

## Caveats and Recommendations
The model needs to be reconstructed to meet the threshold/minimum quality requirements for all key EDU-slices of data. 
It is possible that more data for cetain categories should be collected. It is also reasonble to check the quality of the model for other feauture-slices.
Therefore the current version of the model can be used only as a starting point.






