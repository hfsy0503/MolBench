import json, pprint
with open(r'D:\GitLab\molbench\molbench\hyper_parameters\regression_models\random_forest_regressor.json') as f:
    cfg = json.load(f)
pprint.pp(cfg)