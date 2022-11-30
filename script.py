import pandas as pd
import mlflow
import os
import datetime
 


## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.
# mlflow.set_tracking_uri("http://training.itu.dk:5000/")
mlflow.set_tracking_uri("http://0.0.0.0:5000")
# mlflow.set_tracking_uri("http://127.0.0.1:5001")

print("STARTING\n")
# mlflow.set_tracking_uri("mlruns")
print("SETTING EXPERIMENT\n")
# Set the experiment name
# print("GET EXPERIMENT")
# EXPERIMENT = mlflow.get_experiment_by_name("osri_energy_forecast")
# print("EXPERIMENT: ", EXPERIMENT)

# Import some of the sklearn modules you are likely to use.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import warnings
warnings.filterwarnings("ignore")


def Interpolate(x):
    x['Direction']   = x['Direction'].interpolate(method='pad', limit=13)      # categorical
    x['Speed']       = x['Speed'].interpolate(option='polynomial', order=2)   # polynomial
    x['SpeedShifted_1']       = x['SpeedShifted_1'].interpolate(option='polynomial', order=2)   # polynomial
    x['SpeedShifted_2']       = x['SpeedShifted_2'].interpolate(option='polynomial', order=2)   # polynomial
    x['SpeedShifted_3']       = x['SpeedShifted_3'].interpolate(option='polynomial', order=2)   # polynomial
    return x
    

def SpeedShift(x):
    # Shift speed
    x['SpeedShifted_1'] = x.Speed.shift(-1)
    x['SpeedShifted_2'] = x.Speed.shift(-2)
    x['SpeedShifted_3'] = x.Speed.shift(-3)
    return x

# Start a run
# Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
RUN_NAME = f'Experiment_{datetime.datetime.now}'
# print("RUN NAME: ", RUN_NAME)
# print("EXPERIMENT ID: ",  EXPERIMENT.experiment_id)

client = mlflow.MlflowClient()
mlflow.set_experiment("osri_energy_forecast")
experiment =  mlflow.get_experiment_by_name("osri_energy_forecast")
run = client.create_run(experiment.experiment_id)
run = mlflow.start_run(run_id = run.info.run_id)

df = pd.read_json("dataset.json", orient="split")
# Handle missing data
df = df.resample('3h').first()[1:]
df = df[~df['Total'].isna()]
df['Direction'] = df['Direction'].astype(object)
df['Speed'] = df['Speed'].astype(float)

X = df[["Speed","Direction"]]
y = df["Total"]

### PREPARE PIPELINE

# Transform numerical columns
numeric_transformation = StandardScaler()
# Transform categorical columns
categorical_transformation = OneHotEncoder(handle_unknown="ignore")
# Combine transformations
transformation = ColumnTransformer(
    transformers=[
#         ('speedShift', SpeedShift(), ['Speed']),
#         ('interpolation', Interpolate(), ['Direction', 'Speed', "SpeedShifted_1", "SpeedShifted_2", "SpeedShifted_3"]),
        ("numerical", numeric_transformation, ["Speed", "SpeedShifted_1", "SpeedShifted_2", "SpeedShifted_3"]),
        ('polynomial_speed', PolynomialFeatures(degree=2, include_bias=False), ['Speed']),
        ('polynomial_speedShifted1', PolynomialFeatures(degree=2, include_bias=False), ['SpeedShifted_1']),
        ('polynomial_speedShifted2', PolynomialFeatures(degree=2, include_bias=False), ['SpeedShifted_2']),
        ('polynomial_speedShifted3', PolynomialFeatures(degree=2, include_bias=False), ['SpeedShifted_3']),
        ("categorical", categorical_transformation, ["Direction"])
    ]
)

max_depth = 50
objective = 'reg:squarederror'
eval_metric = 'rmse'

mlflow.log_param("max_depth", max_depth)
mlflow.log_param("objective", objective)
mlflow.log_param("eval_metric", eval_metric)

regression_xgb = Pipeline(
    steps=[("transformation", transformation),
        ('StandardScaler', StandardScaler()),
        ("regression", xgb.XGBRegressor(max_depth=max_depth,
                                        objective=objective,
                                        eval_metric = eval_metric,
                                        ))]
)

#TODO: Log your parameters. What parameters are important to log?
#HINT: You can get access to the transformers in your pipeline using `pipeline.steps`

# Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
number_of_splits = 5

metrics = [
        ("MAE", mean_absolute_error, []),
        ("MSE", mean_squared_error, []),
        ("R2", r2_score, [])
    ]

ITR = 0
for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
    X_train = SpeedShift(x=X.iloc[train])
    X_train = Interpolate(x=X_train)
    y_train = y.iloc[train]
    
    X_test = SpeedShift(x=X.iloc[test])
    X_test = Interpolate(x=X_test)
    y_test = y.iloc[test]
    
    regression_xgb.fit(X_train,y_train)
    predictions = regression_xgb.predict(X_test)
    truth = y_test
    
    print("CREATE PLOT")
    from matplotlib import pyplot as plt 
    plt.plot(truth.index, truth.values, label="Truth")
    plt.plot(truth.index, predictions, label="Predictions")
    plt.savefig('figpath.png')
    plt.close()
    print("LOG PLOT AS ARTIFACT")
    mlflow.log_artifact('figpath.png', "PredictionsGraph")

    print("CALCULATE METRICS")
    # Calculate and save the metrics for this fold
    for name, func, scores in metrics:
        score = func(truth, predictions)
        scores.append(score)
        
    print("LOG METRICS")
    # Log a summary of the metrics
    for name, _, scores in metrics:
            # NOTE: Here we just log the mean of the scores. 
            # Are there other summarizations that could be interesting?
            mean_score = sum(scores)/number_of_splits
            mlflow.log_metric(f"mean_{name}", mean_score)
    print("LOG MODEL")
    mlflow.sklearn.log_model(regression_xgb, f'XGBoost_PowerProduction_{ITR}')
    ITR += 1

# mlflow.pyfunc.save_model("model", python_model=regression_xgb, conda_env="conda.yaml")

mlflow.end_run()