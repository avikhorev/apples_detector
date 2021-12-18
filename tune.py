import cv2
import optuna
import traceback

import train
import hough_circles_trackbar as detector

PAR = detector.PAR

OPTUNA_STUDY_NAME = 'study_1'

def suggest(param_name, suggest_fn, *suggest_args, par=PAR):
    par[param_name] = suggest_fn(param_name, *suggest_args)

def predict_obj(trial):
    suggest( 'blur', trial.suggest_int, 1, 30, 2)

    suggest( 'hMin', trial.suggest_int, 0, 179)
    suggest( 'hMax', trial.suggest_int, 0, 179)
    suggest( 'sMin', trial.suggest_int, 0, 255)
    suggest( 'sMax', trial.suggest_int, 0, 255)
    suggest( 'vMin', trial.suggest_int, 0, 255)
    suggest( 'vMax', trial.suggest_int, 0, 255)

    suggest( 'binarize',      trial.suggest_categorical, [False,True])
    suggest( 'equalize_hist', trial.suggest_categorical, [False,True])

    suggest( 'dp', trial.suggest_float, 1.0, 9.0 )
    suggest( 'param1', trial.suggest_int, 1, 500)
    suggest( 'param2', trial.suggest_int, 1, 100)
    suggest( 'min_dist', trial.suggest_int, 1, 1000)
    suggest( 'radius', trial.suggest_int, 1, 100)

    # suggest( 'nfeatures', trial.suggest_int, 0, 999, par=PAR_SIFT)
    return train.train_all()

def optimize_preds():

    study_name = OPTUNA_STUDY_NAME
    storage_name = f"sqlite:///{study_name}.db"
    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.create_study(
        # direction='maximize',
        study_name=study_name
        ,storage=storage_name
        ,load_if_exists=True
        ,sampler=sampler
        )

    try:
        study.optimize(
            lambda trial: predict_obj(trial),
            catch=(RuntimeError,),
            gc_after_trial=False,
            show_progress_bar=True
        )
    except:
        print(traceback.print_exc())

    print(f"--------------------------------")
    print(f"BEST TRIAL   :{study.best_trial}")
    print(f"Sampler      :{study.sampler.__class__.__name__}")
    print(f"--------------------------------")
    print(f"BEST VALUE  :{study.best_value:.5f}")
    print(f"BEST PARAMS :{study.best_params}")

optimize_preds()
