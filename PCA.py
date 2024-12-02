import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv('NORMALIZED_teamdata.csv')
meta = df[['season', 'score_away', 'score_home']]
features = df.drop(columns=['score_away', 'score_home', 'season'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
pca = PCA(n_components=0.95)
pca.fit(scaled_features)
loading_matrix = pd.DataFrame(pca.components_, columns=features.columns)
feature_importance = loading_matrix.abs().sum(axis=0)

sorted_features = feature_importance.sort_values(ascending=False)
threshold = 0.8 * feature_importance.sum()
cumulative_importance = sorted_features.cumsum()
significant_features = cumulative_importance[cumulative_importance <= threshold].index

filtered_features = df[significant_features]
final_dataset = pd.concat([meta, filtered_features], axis=1)
train_set = final_dataset[final_dataset['season'].between(2002, 2021)]
test_set = final_dataset[final_dataset['season'].between(2022, 2023)]

train_set = train_set.drop(columns=['season'])
test_set = test_set.drop(columns=['season'])

train_set.to_csv('TRAIN_PCA.csv', index=False)
test_set.to_csv('TEST_PCA.csv', index=False)

scaler = StandardScaler()
X_standardized = scaler.fit_transform(filtered_features)
X_standardized_df = pd.DataFrame(X_standardized, columns=filtered_features.columns)
final_standardized = pd.concat([meta, X_standardized_df], axis=1)
train_standardized = final_standardized[final_standardized['season'].between(2002, 2021)]
test_standardized = final_standardized[final_standardized['season'].between(2022, 2023)]

train_standardized = train_standardized.drop(columns=['season'])
test_standardized = test_standardized.drop(columns=['season'])

train_standardized.to_csv('TRAIN_PCA_STANDARDIZED.csv', index=False)
test_standardized.to_csv('TEST_PCA_STANDARDIZED.csv', index=False)