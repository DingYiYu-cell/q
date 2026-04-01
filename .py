import optuna
from train import run_training


def objective(trial):
    test_params = {
        'opt_class':trial.suggest_categorical('opt_class', ['AdamW','RMSprop','SGD','RAdam','NAdam']),
        'lr':trial.suggest_float('lr',0.00001,0.001, log=True),
        'batch_size':trial.suggest_int('batch', 10,100, log=True),
        'cos_maxt':trial.suggest_int('cos_maxt',1, 15, log=True),
        'alpha':trial.suggest_float('alpha', 0.5, 1.5, log=True),
        'beta':trial.suggest_float('beta', 0.5, 1.5, log=True),
        
    }
    score = run_training(test_params)
    
    return score
if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize","minimize"])
    study.optimize(objective, n_trials=100)
    print(f"Best Score: {study.best_value}")
    print(f"Best Params: {study.best_params}")
    