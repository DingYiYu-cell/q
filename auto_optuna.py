import optuna
from train import run_training


def objective(trial):
    test_params = {
        'lr':trial.suggest_float('lr',0.00001,0.001, log=True),
        'batch_size':trial.suggest_int('batch', 10,100, log=True),
        'cos_maxt':trial.suggest_int('cos_maxt',1, 20, log=True),
    }
    score = run_training(test_params)
    
    return score
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print(f"Best Score: {study.best_value}")
    print(f"Best Params: {study.best_params}")
    