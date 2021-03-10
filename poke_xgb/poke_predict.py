#%% Import data and libraries
import pandas as pd
import numpy as np
import altair as alt

from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split

combat = pd.read_csv('C:/Users/ecsan/py_scripts/Independant Work/ml/poke_xgb/combats.csv')
pokemon = pd.read_csv('C:/Users/ecsan/py_scripts/Independant Work/ml/poke_xgb/pokemon.csv')
tests = pd.read_csv('C:/Users/ecsan/py_scripts/Independant Work/ml/poke_xgb/tests.csv')

#%% Create df for features based off of combat
join_features = combat.join(
    pokemon.set_index('#'),
    on='First_pokemon')

complete_join = join_features.join(
    pokemon.set_index('#'),
    on='Second_pokemon',
    lsuffix='_first',
    rsuffix='_second')

features = complete_join.drop(columns=['Name_first','Name_second','Winner'])
#%% Create series with wins and losses based off of First pokemon in combat
col_01=combat['Winner'].eq(combat['First_pokemon']).replace(True,int(1))
targets = col_01.rename('Win/Loss')


#%% Handle categorical dtypes
for c in features:
    if features[c].dtypes == 'object':
        features[c] = features[c].astype('category')    
        if features[c].dtypes == 'category':
            features[c] = features[c].cat.codes

#%% Select targets and features
X_pred = features
y_pred = targets 

#%% Split up data 
X_train, X_test,y_train, y_test = train_test_split(
    X_pred,y_pred,test_size=.3, random_state=42
)

#%% Build and test model
model = XGBClassifier() 
model = model.fit(X_train,y_train)
predictions = model.predict(X_test)

#%% Evaluate
f_contributions = pd.DataFrame(
    {'f_names': X_train.columns, 
    'f_values': model.feature_importances_}).sort_values('f_values', ascending = False)
report = metrics.classification_report(y_test,predictions)
print(report)
metrics.plot_roc_curve(model,X_test,y_test)

#%% Create markdown report
report = metrics.classification_report(y_test,predictions, output_dict=True)
df = pd.DataFrame(report)
print(df.to_markdown())

#%% Chart evaluations
chart = (alt.Chart(f_contributions.query('f_values > 0.03'),
    title='Pokemon with Greater Speed are More Likely to Win')
    .encode(
        alt.X('f_values',
        title='Feature Values'),

        alt.Y('f_names',
        title='Feature Names',
        sort='-x'),

        alt.Color('f_names')
    ).mark_bar()
)
chart
# %%
