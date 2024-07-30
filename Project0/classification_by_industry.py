import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

current_working_distribution = "source\\t1_distribution.csv"
dataframe_from_distribution = pd.read_csv(current_working_distribution)
dataframe_from_distribution['ReportDate'] = pd.to_datetime(dataframe_from_distribution['ReportDate'],
                                               format = '%d/%m/%Y %H:%M:%S')
repo_for_Industry = dict()
industry_names = sorted(set(dataframe_from_distribution['IndustryName']))
total_time_span = sorted(set(dataframe_from_distribution['ReportDate']))

for name in industry_names:
    repo_for_Industry[name] = []

for time in total_time_span:
    specific_time_slice = dataframe_from_distribution[dataframe_from_distribution['ReportDate'] == time]
    industry_names_at_time = set(specific_time_slice['IndustryName'])
    for industry_name in industry_names:
        num_of_share = 0 if industry_name not in industry_names_at_time else \
            specific_time_slice[specific_time_slice['IndustryName'] == industry_name][
                'MarketValue'].values[0]
        repo_for_Industry[industry_name].append(num_of_share)

industry_dataframe = pd.DataFrame(repo_for_Industry)
industry_dataframe.to_csv("analytics\\Market_Value_by_Industry_t1")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(industry_dataframe)

pca = PCA(n_components = len(industry_names))
pca.fit(scaled_data.T)

components = pca.components_
concise_eigenvectors = components[0:5]

weights_df = pd.DataFrame(columns = ["Location", "Name", "Weights"])
data_set = np.array(list(repo_for_Industry.values()))

for i in range(5):
    target = np.array(concise_eigenvectors[i])
    data_set_transposed = data_set.T
    num_assets = data_set.shape[0]

    weights = cp.Variable(num_assets,nonneg = True)

    objective = cp.Minimize(cp.norm(data_set_transposed @ weights - target, 2))
    constraints = [cp.sum(weights) == 1]

    problem = cp.Problem(objective, constraints)

    problem.solve(solver = cp.SCS)

    if problem.status in ["optimal", "optimal_inaccurate"]:
        optimal_weights = weights.value
        combined_result = np.dot(optimal_weights, data_set)
        weights_list = optimal_weights.tolist()

        combined = sorted(zip(industry_names, weights_list), key = lambda pair: -pair[1])
        count = 0
        for info in combined:
            if count < 5:
                weights_df = weights_df._append({
                "Location": i,
                    "Name": info[0],
                    "Weights": info[1]
                }, ignore_index = True)
                count+=1
            else:
                pass
    else:
        print("Optimization failed:", problem.status)


weights_df.to_csv("analytics\\IndustryWeights_t1.csv", index = False)
print("Weights saved to analytics\\Industry_t1")