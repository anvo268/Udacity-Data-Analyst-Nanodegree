import pickle

from Functions import *

# Set up the data
data = load_data()
# dropping my obvious outliers
data = remove_outliers(data)
X, y = data[all_features], data.poi

# Tuning for the pipeline
clf = [('clf', LogisticRegression())]
model = imb_pipeline(base_pipeline + clf)
param_grid = base_param_grid.copy()

lr_grid = evaluate_model(model, param_grid, X, y, optimize='precision')

# Tune LR hyperparameters
clf = [('clf', LogisticRegression())]
model = imb_pipeline(base_pipeline + clf)

param_grid = {}
for k, v in lr_grid.best_params_.items():
    if v == 'None':
        v = None
    param_grid[k] = [v]

param_grid['clf__penalty'] = ['l1', 'l2']
param_grid['clf__C'] = [0.1, 1, 10, 100, 1000, 5000]

grid = evaluate_model(model, param_grid, X, y, optimize='precision')

# Grid contains the final model
pickle.dump(convert_to_dict(data), open('my_dataset.pkl', 'wb'))
pickle.dump(grid.best_estimator_, open('my_classifier.pkl', 'wb'))
pickle.dump( ['poi'] + all_features, open('my_feature_list.pkl', 'wb'))