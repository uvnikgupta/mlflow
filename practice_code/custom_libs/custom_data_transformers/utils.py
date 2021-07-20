import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Custom column transformation can be defined as a class or as a function. 
# - When defined as a class, it can be used directly as a step in the pipeline
# - When defined as a function it can be wrapped with FunctionTransformer which 
#   can then be used as a step in the pipeline.
class HouseColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = 1
        self.bedrooms_ix = 2 
        self.population_ix = 3 
        self.households_ix = 4
        
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X[:,0], X[:,5],
                         rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X[:,0], X[:,2], X[:,5], rooms_per_household, population_per_household]


def HouseColumnTransformerFunc(X, add_bedrooms_per_room = True):
    add_bedrooms_per_room
    rooms_ix = 1
    bedrooms_ix = 2 
    population_ix = 3 
    households_ix = 4
        
    rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
    population_per_household = X[:, population_ix] / X[:, households_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X[:,0], X[:,5],
                     rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X[:,0], X[:,2], X[:,5], rooms_per_household, population_per_household]