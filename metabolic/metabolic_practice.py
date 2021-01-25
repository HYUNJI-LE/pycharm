import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression , Ridge , Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# %matplotlib inline

# File load
file_df = pd.read_csv('./DataAnalasys/metabolic_syndrome/metabolic_syndrome_data.csv')


################################################전처리#####################################################


print(file_df.info())
print(file_df.head())

#Age_Category 는 age컬럼과 중복이고 null 값이 많아 제거할 것
#_risk 는 다른 변수들과 관계성이 높기 때문에 예측에 영향을 줄 수 있을 것 같아 제거할 것
target_name = 'Metabolic_Syndrome'
no_need_features = ['Index','Age_Category','Waist_risk','BP_risk','TG_risk','HDL_risk','Glucose_risk']
category_features = ['Gender']

# feature값 정리
file_df.drop(no_need_features, axis=1, inplace=True)
y_target = file_df[target_name]
X_features = file_df.drop([target_name],axis=1,inplace=False)
print(file_df.info())
print(X_features.head())

#Age feature 범주화
def get_category(Age):
    cat = ''
    if Age < 30: cat = '20s'
    elif Age < 40: cat = '30s'
    elif Age < 50: cat = '40s'
    elif Age < 60: cat = '50s'
    else : cat = '60s'

    return cat

ages = ['20s', '30s', '40s', '50s', '60s']

#카테고리 feature 값 분포 확인
print( 'Gender 값 분포', file_df['Gender'].value_counts())
print(file_df.groupby(['Gender', 'Metabolic_Syndrome'])['Metabolic_Syndrome'].count())
sns.barplot(x='Gender', y='Metabolic_Syndrome', data=file_df)
plt.show()
sns.barplot(x='Gender', y='Metabolic_Syndrome', hue=file_df['Age'].apply(lambda x : get_category(x)), data=file_df)
plt.show()


#target값의 정규 분포 확인 및 로그 변환
plt.title('Metabolic Syndrome Histogram')
sns.distplot(y_target)
plt.show()
y_target_log = np.log1p(y_target)
sns.distplot(y_target_log)
plt.show()

#feature값의 이상치 데이터 확인
for feature in X_features.drop(category_features, axis=1, inplace=False):
    plt.scatter(x = X_features[feature], y = y_target)
    plt.ylabel(target_name, fontsize=15)
    plt.xlabel(feature, fontsize=15)
    plt.show()

print(y_target.head())
print(X_features.head())

#이상치 데이터 제거
outlier_name = ['HR','TG','HDL','Muscle','Water','TC']
cond1 =  X_features[outlier_name[0]] > 100
cond2 =  X_features[outlier_name[1]] > 700
cond3 =  X_features[outlier_name[2]] > 120
cond4 =  X_features[outlier_name[3]] < 10
cond5 =  X_features[outlier_name[4]] < 10
cond6 = X_features[outlier_name[5]] > 300
outlier_index = X_features[ cond1 | cond2 | cond3 | cond4 | cond5 | cond6 ].index

print('이상치 레코드 index:', outlier_index.values)
print('이상치 삭제 전 X_features_ohe:', X_features.shape)

X_features.drop(outlier_index, axis=0, inplace=True)
print('이상치 삭제 후 X_features_ohe:', X_features.shape)


#skew 정도가 높은 features(TG,FBS,HDL,TC)의 로그변환 전후 분포 확인
sns.distplot(file_df['TG'])
plt.show()
sns.distplot(file_df['FBS'])
plt.show()
sns.distplot(file_df['HDL'])
plt.show()
sns.distplot(file_df['TC'])
plt.show()

#feature의 데이터 분포도 확인(왜곡된 데이터 확인)
features_index = X_features.dtypes.index
skew_features = X_features[features_index].apply(lambda x : skew(x))
print(skew_features.sort_values(ascending=False))
skew_features_change = skew_features[skew_features > 0.9]
X_features[skew_features_change.index] = np.log1p(X_features[skew_features_change.index])

sns.distplot(file_df['TG'])
plt.show()
sns.distplot(file_df['FBS'])
plt.show()
sns.distplot(file_df['HDL'])
plt.show()
sns.distplot(file_df['TC'])
plt.show()

#범주형 feature의 onehotencoding
print('get_dummies 수행 전 데이터 shape:', X_features.shape)
X_features_ohe = pd.get_dummies(file_df, columns=category_features)
print('get_dummies 수행 후 데이터 shape:', X_features_ohe.shape)

print(X_features_ohe.head())

pd.concat(g)
X_features['Gender'] = X_features_ohe['Gender']
print(X_features['Gender'].head())

################################################train-test_split####################################################

X_train, X_test, y_train, y_test = train_test_split(X_features_ohe,y_target_log,test_size=0.2, random_state=156 )

#################################################최적 파라미터####################################################
def get_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=5)
    grid_model.fit(X_features_ohe, y_target_log)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_

# 회귀 모델 정의
lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=0)
lasso_reg = Lasso(alpha=0.001)
en_reg = ElasticNet(alpha=0.0001, l1_ratio=0.8)
dt_reg = DecisionTreeRegressor(max_depth=7)
rf_reg = RandomForestRegressor(random_state=0, n_estimators=30, max_depth=7, min_samples_leaf=1, min_samples_split=2, n_jobs=-1)
gbm_reg = GradientBoostingRegressor(n_estimators=800, learning_rate=0.1, subsample=0.9)
xgb_reg = XGBRegressor(n_estimators=190, eta=0.1, min_child_weight=3, max_depth=2)
lgbm_reg = LGBMRegressor(n_estimators=900, learning_rate=0.05, max_depth=2, min_child_samples=6, num_leaves=3)


ridge_params = { 'alpha':[0, 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01] }
lasso_params = { 'alpha':[0.001]  }
en_params = { 'alpha':[0, 0.0001, 0.0002, 0.0005, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.02]}
rf_params = {'n_estimators':[30], 'max_depth' : [7], 'min_samples_leaf' : [1], 'min_samples_split' : [2]}
gbm_params = {'n_estimators':[700, 800, 900], 'learning_rate': [0.1, 0.11, 0.12], 'subsample': [0.7, 0.8, 0.9, 1]}
xgb_params = {'n_estimators':[190], 'eta': [0.1], 'min_child_weight': [3], 'max_depth': [2], 'colsample_bytree': [1, 2]}
lgbm_params = {'n_estimators':[900], 'learning_rate': [0.05], 'max_depth': [2], 'min_child_samples': [6], 'num_leaves': [3], 'colsample_bytree': [1], 'feature_fraction': [1]}

best_rige = get_best_params(ridge_reg, ridge_params)
best_lasso = get_best_params(lasso_reg, lasso_params)
best_en = get_best_params(en_reg, en_params)
best_rf = get_best_params(rf_reg, rf_params)
best_gbm = get_best_params(gbm_reg, gbm_params)
best_xgb = get_best_params(xgb_reg, xgb_params)
best_lgbm = get_best_params(lgbm_reg, lgbm_params)

############################################예측/평가#########################################################



