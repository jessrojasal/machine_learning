from preprocess import (
    DataAugmentation,
    DataImputation,
    FeatureCombination,
    RemoveOutliers,
    FilterAndDrop,
)

from model import (
    SplitData,
    ModelDecisionTreeRegressor,
    ModelRandomForestRegressor
)

# Preprocess
# Data Augmentation: Add new features
augmenter = DataAugmentation(base_path="data")
data_augmented = augmenter.add_new_features()

# Handle missing values with imputation
imputer = DataImputation(data_augmented)
data_no_missing_values = imputer.impute_missing_values()

# Create new columns by performing feature combination
feature_combiner = FeatureCombination(data_no_missing_values)
combined_data = feature_combiner.combine_features()

# Remove outliers
data_cleaner = RemoveOutliers(combined_data)
cleaned_data = data_cleaner.remove_outliers()

# Filter observations and drop columns
filter_obj = FilterAndDrop(cleaned_data)
data = filter_obj.filter_drop()

# Model
# Split database
splitter = SplitData(data, target_column='price')
X_train, X_test, y_train, y_test = splitter.split()

# Train Decision Tree Regressor model
dt_regressor = ModelDecisionTreeRegressor(random_state=100, 
                                        min_samples_split=15, min_samples_leaf=10, max_leaf_nodes=150, max_depth=20)
dt_regressor.train(X_train, y_train)
dt_metrics = dt_regressor.evaluate(X_train, X_test, y_train, y_test)

# Train Random Forest Regressor model
rf_regressor = ModelRandomForestRegressor(
    random_state=100, n_estimators=150, min_samples_split=100, 
    min_samples_leaf=17, max_leaf_nodes=100, max_depth=100
)
rf_regressor.train(X_train, y_train)
rf_metrics = rf_regressor.evaluate(X_train, X_test, y_train, y_test)