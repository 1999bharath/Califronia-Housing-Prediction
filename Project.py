#!/usr/bin/env python
# coding: utf-8

# Fetching and Loading Data

# In[ ]:


import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Call the function to fetch the housing data
fetch_housing_data()


# Loading Data into Pandas DataFrame

# In[2]:


import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
 csv_path = os.path.join(housing_path, "housing.csv")
 return pd.read_csv(csv_path)


# Initial Data Exploration

# In[3]:


housing=load_housing_data()
housing.head()


# In[4]:


housing.info()


# In[5]:


housing.describe()


# In[6]:


housing["ocean_proximity"].value_counts()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt
housing.hist(bins=100, figsize=(20,15))
plt.show()


# Splitting Data into Training and Test Sets

# In[10]:


import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[12]:


train_set,test_set=split_train_test(housing,0.2)


# In[13]:


len(train_set)


# In[14]:


len(test_set)


# In[15]:


from zlib import crc32
def test_set_check(identifier, test_ratio):
 return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
 ids = data[id_column]
 in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
 return data.loc[~in_test_set], data.loc[in_test_set]


# In[25]:


housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[26]:


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[27]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# Ensuring No Sampling Bias

# In[28]:


housing["income_cat"] = pd.cut(housing["median_income"],
 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
 labels=[1, 2, 3, 4, 5])


# In[29]:


housing["income_cat"].hist()


# Stratified Sampling

# In[30]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
 strat_train_set = housing.loc[train_index]
 strat_test_set = housing.loc[test_index]


# In[31]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[32]:


def calculate_proportions(data):
    return data["income_cat"].value_counts(normalize=True).sort_index()

overall_proportions = calculate_proportions(housing)
train_proportions = calculate_proportions(train_set)
test_proportions = calculate_proportions(test_set)

# Calculate the stratified error
stratified_error_train = (train_proportions - overall_proportions).abs()
stratified_error_test = (test_proportions - overall_proportions).abs()

# Display the proportions and errors
proportions_df = pd.DataFrame({
    "Overall": overall_proportions,
    "Train": train_proportions,
    "Test": test_proportions,
    "Train Error": stratified_error_train,
    "Test Error": stratified_error_test
})

print(proportions_df)

# Plot the proportions for comparison
proportions_df[["Overall", "Train", "Test"]].plot(kind='bar', figsize=(10, 6))
plt.title("Income Category Proportions")
plt.show()

# Plot the stratified errors
proportions_df[["Train Error", "Test Error"]].plot(kind='bar', figsize=(10, 6))
plt.title("Stratified Errors")
plt.show()


# Data Visualization

# In[33]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[34]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2)


# In[ ]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# In[35]:


corr_matrix = housing.corr()


# In[36]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# Feature Engineering and Transformation

# Data Cleaning

# In[37]:


median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)


# In[52]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[53]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


# In[54]:


housing_num = housing.drop("ocean_proximity", axis=1)


# In[55]:


imputer.fit(housing_num)


# In[56]:


X = imputer.transform(housing_num)


# In[57]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# Dealing with categorical values

# In[58]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[59]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[60]:


housing_cat_1hot.toarray()


# In[61]:


cat_encoder.categories_


# Custom Transformer

# In[62]:



from sklearn.base import BaseEstimator, TransformerMixin

# Indices for the relevant columns
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Create an instance of the transformer
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

# Assuming `housing` is a DataFrame and we want to transform its values
housing_extra_attribs = attr_adder.transform(housing.values)


# Feature scaling
# 
# 

# In[63]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[64]:


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
 ("num", num_pipeline, num_attribs),
 ("cat", OneHotEncoder(), cat_attribs),
 ])
housing_prepared = full_pipeline.fit_transform(housing)


# Model Training and Evaluation

# In[65]:


from sklearn.linear_model import LinearRegression


# In[66]:


lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[67]:


# Select some data for predictions
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

# Make predictions
predictions = lin_reg.predict(some_data_prepared)

# Print predictions and actual labels
print("Predictions:", predictions)
print("Labels:", list(some_labels))


# In[86]:



from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# Cross-Validation

# In[75]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[70]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[71]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
 scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[74]:


def display_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean())
  print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[79]:



from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)


# In[82]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
 scoring="neg_mean_squared_error", cv=10)
forest_reg_rmse_scores = np.sqrt(-scores)


# In[83]:


def display_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean())
  print("Standard deviation:", scores.std())

display_scores(forest_reg_rmse_scores)


# Hyperparameter Tuning with Grid Search

# In[91]:


from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[92]:


grid_search.best_estimator_


# In[94]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[95]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# Evaluating the Final Model

# In[96]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[97]:


final_rmse

