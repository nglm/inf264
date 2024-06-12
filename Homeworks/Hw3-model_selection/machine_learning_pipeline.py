import numpy as np
from sklearn.model_selection import train_test_split, KFold

def split_dataset(X, Y, test_ratio=0.2, val_ratio=0.2, seed=264, summary={}):
    """
    Split dataset into training, validation, test datasets
    """
    # Extract test set from entire dataset
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_ratio, shuffle=True, random_state=seed)
    # Extract validation set from train_val dataset
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_ratio, shuffle=True, random_state=seed)

    summary["X_train"] = X_train
    summary["X_val"] = X_val
    summary["X_test"] = X_test
    summary["Y_train"] = Y_train
    summary["Y_val"] = Y_val
    summary["Y_test"] = Y_test

    print("Training dataset size:   ", len(X_train))
    print("Validation dataset size: ", len(X_val))
    print("Test dataset size:       ", len(X_test))
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def evaluate(model, X, Y, metric):
    """
    Evaluate (already trained) model performance on the given dataset

    'metric' is a performance metric, a python function comparing 'Y' and 'Y_pred'
    """
    Y_pred = model.predict(X)
    perf = metric(Y, Y_pred)
    return perf

def model_selection(model_classes, hparams, val_perf, minimize, summary={}):
    """
    Return index of the best model given their validation performances

    'model_classes', 'hparams' and 'val_perf' all have the same
    number of elements

    Note: Some performance metrics have to be minimized (e.g. MSE) and
    some have to be maximized (e.g. accuracy, f1-score)
    """
    if minimize:
        i_best = np.argmin(val_perf)
    else:
        i_best = np.argmax(val_perf)

    summary["i_best"] = i_best
    summary["best_params"] = hparams[i_best]
    summary["best_model_class"] = model_classes[i_best]

    print("\nBest model selected, with")
    print("type of model:          ", model_classes[i_best])
    print("hyperparameters:        ", hparams[i_best])

    return i_best

def best_model_evaluation(
    model, X_train, Y_train, X_val, Y_val, X_test, Y_test, metric,
    summary={}
):
    """
    Evaluate the selected model

    Train on the entire training/validation dataset
    Evaluate performance on both training/validation dataset and test dataset
    """
    # Take the whole training/validation dataset
    X_train_val = np.concatenate((X_train, X_val))
    Y_train_val = np.concatenate((Y_train, Y_val))

    # Train on the entire training/validation dataset
    model.fit(X_train_val, Y_train_val)

    # Evaluate performance
    train_val_perf = evaluate(model, X_train_val, Y_train_val, metric)
    test_perf = evaluate(model, X_test, Y_test, metric)

    summary["train_val_performance"] = train_val_perf
    summary["test_performance"] = test_perf

    print("\nSelected model performance:")
    print("Training (incl val):   %.4f" %train_val_perf)
    print("Test:                  %.4f" %test_perf)
    return model, train_val_perf, test_perf

def basic_pipeline(
    X, Y, model_classes, hparams, performance,
    test_ratio=0.2, val_ratio=0.2, seed=264
):
    """
    Typical simple machine learning pipeline
    """

    # Summary containing info about the entire pipeline
    summary = {}

    # Split dataset into a train, validation and test set
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(
        X, Y, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed,
        summary=summary)

    train_perfs = []
    val_perfs = []
    models = []
    # Loop over sets of model and hyperparameters:
    for modelClass, hparam in zip(model_classes, hparams):
        print("\nUsing hyper-parameters:", hparam)

        # Instantiate model with current set of hyperparameter
        model = modelClass(**hparam)

        # Train model
        model.fit(X_train, Y_train)

        # Compute train and validation loss
        train_perf = evaluate(model, X_train, Y_train, performance["metric"])
        val_perf = evaluate(model, X_val, Y_val, performance["metric"])
        print("Training performance:    %.4f" %train_perf)
        print("Validation performance:  %.4f" %val_perf)

        # Store info about this model
        models.append(model)
        train_perfs.append(train_perf)
        val_perfs.append(val_perf)

    # Model selection
    i_best = model_selection(
        model_classes, hparams, val_perfs, performance["minimize"],
        summary=summary)

    # Instantiate a model with the selected parameters
    best_params = hparams[i_best]
    best_modelClass = model_classes[i_best]
    best_model = best_modelClass(**best_params)

    # Evaluate the selected model
    best_model, train_val_perf, test_perf = best_model_evaluation(
        best_model, X_train, Y_train, X_val, Y_val, X_test, Y_test,
        performance["metric"], summary=summary
    )

    summary['best_model'] = best_model

    return models, i_best, best_model, summary

def KFold_split(X, Y, k=5, test_ratio=0.2, seed=264, summary={}):
    """
    Split dataset into a test dataset and train/val kfolds
    """
    # Extract test set from entire dataset
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_ratio, shuffle=True, random_state=seed)

    # Create train/validation kfolds splitter
    KFold_splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    X_train_folds = []
    X_val_folds = []
    Y_train_folds = []
    Y_val_folds = []

    # Split train_val dataset into folds
    for (kth_fold_train_idxs, kth_fold_val_idxs) in KFold_splitter.split(X_train_val, Y_train_val):
        X_train_folds.append(X_train_val[kth_fold_train_idxs])
        X_val_folds.append(X_train_val[kth_fold_val_idxs])
        Y_train_folds.append(Y_train_val[kth_fold_train_idxs])
        Y_val_folds.append(Y_train_val[kth_fold_val_idxs])

    summary["X_train_folds"] = X_train_folds
    summary["X_val_folds"] = X_val_folds
    summary["X_test"] = X_test
    summary["Y_train_folds"] = Y_train_folds
    summary["Y_val_folds"] = Y_val_folds
    summary["Y_test"] = Y_test

    print("Training dataset size:   ", len(X_train_folds[0]))
    print("Validation dataset size: ", len(X_val_folds[0]))
    print("Test dataset size:       ", len(X_test))
    return X_train_folds, Y_train_folds, X_val_folds, Y_val_folds, X_test, Y_test

def pipeline_with_KFold(
    X, Y, model_classes, hparams, performance,
    k=5, test_ratio=0.2, seed=264
):

    # Summary containing info about the entire pipeline
    summary = {}

    # Split dataset into a train, validation and test set
    X_train_folds, Y_train_folds, X_val_folds, Y_val_folds, X_test, Y_test = KFold_split(
        X, Y, k, test_ratio=test_ratio, seed=seed, summary=summary
    )

    train_mean_perfs = []
    val_mean_perfs = []
    models = []

    # Loop over sets of model and hyperparameters:
    for modelClass, hparam in zip(model_classes, hparams):
        print("\nUsing hyper-parameters:", hparam)

        train_perfs = []
        val_perfs = []

        # Extra loop for the cross validation
        for X_train, X_val, Y_train, Y_val in zip(X_train_folds, X_val_folds, Y_train_folds, Y_val_folds):

            # Instantiate model with current set of hyperparameter
            model = modelClass(**hparam)

            # Train model
            model.fit(X_train, Y_train)

            # Compute train and validation loss
            train_perf = evaluate(model, X_train, Y_train, performance["metric"])
            val_perf = evaluate(model, X_val, Y_val, performance["metric"])

            train_perfs.append(train_perf)
            val_perfs.append(val_perf)

        # Compute mean performance for this set of hyperparameters
        train_mean_perf = np.mean(train_perfs)
        val_mean_perf = np.mean(val_perfs)
        print("Training mean performance:    %.4f" %train_mean_perf)
        print("Validation mean performance:  %.4f" %val_mean_perf)

        # Store info about this set of hyperparameters
        train_mean_perfs.append(train_mean_perf)
        val_mean_perfs.append(val_mean_perf)

        # The model trained with the last fold will represent all other
        # models trained with this set of hyperparameters but on other folds
        models.append(model)

    # Model selection
    i_best = model_selection(
        model_classes, hparams, val_mean_perfs, performance["minimize"],
        summary=summary)

    # Instantiate a model with the selected parameters
    best_params = hparams[i_best]
    best_modelClass = model_classes[i_best]
    best_model = best_modelClass(**best_params)

    # Evaluate the selected model
    best_model, train_val_perf, test_perf = best_model_evaluation(
        best_model,
        X_train_folds[0], Y_train_folds[0],
        X_val_folds[0], Y_val_folds[0],
        X_test, Y_test,
        performance["metric"],
        summary=summary
    )

    summary['best_model'] = best_model

    return models, i_best, best_model, summary
