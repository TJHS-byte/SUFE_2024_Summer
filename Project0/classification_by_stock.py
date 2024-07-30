import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cvxpy as cp


# Load the data
current_working_distribution = "source\\t1_distribution.csv"
current_working_stock = "source\\target 1 top 10 key stock.csv"

dataframe_from_distribution = pd.read_csv(current_working_distribution)
dataframe_from_stock = pd.read_csv(current_working_stock)
dataframe_from_stock.drop(columns = ["MarketValue", "RatioInNV"], inplace = True)

# Calculate revenue of ETF
whole_dataframe = pd.merge(dataframe_from_distribution, dataframe_from_stock, on = 'ReportDate',
                           how = 'inner')
whole_dataframe.columns = whole_dataframe.columns.str.replace('_x', '').str.replace('_y', '')
whole_dataframe.to_csv("analytics\\WholeDataFrame_t1", index = False)
whole_dataframe['ReportDate'] = pd.to_datetime(whole_dataframe['ReportDate'],
                                               format = '%d/%m/%Y %H:%M:%S')
whole_dataframe["TotalMarketValue"] = whole_dataframe["MarketValue"] / whole_dataframe["RatioInNV"]
whole_dataframe["Who Cares ?"] = "Nobody"
whole_dataframe['MarketValueDiff'] = whole_dataframe['TotalMarketValue'].diff()
whole_dataframe.fillna(whole_dataframe['MarketValueDiff'].interpolate(), inplace = True)

# Calculate SharesHolding Vectors
repo_for_SharesHolding = dict()
stock_inner_codes = sorted(set(whole_dataframe['StockInnerCode']))
total_time_span = sorted(set(whole_dataframe['ReportDate']))

for code in stock_inner_codes:
    repo_for_SharesHolding[code] = []

for time in total_time_span:
    specific_time_slice = whole_dataframe[whole_dataframe['ReportDate'] == time]
    stock_inner_codes_at_time = set(specific_time_slice['StockInnerCode'])
    for stock_inner_code in stock_inner_codes:
        num_of_share = 0 if stock_inner_code not in stock_inner_codes_at_time else \
            specific_time_slice[specific_time_slice['StockInnerCode'] == stock_inner_code][
                'SharesHolding'].values[0]
        repo_for_SharesHolding[stock_inner_code].append(num_of_share)

shares_holding_dataframe = pd.DataFrame(repo_for_SharesHolding)
shares_holding_dataframe.to_csv("analytics\\ShareHoldings_t1")

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(shares_holding_dataframe)

# Perform PCA
pca = PCA(n_components = len(total_time_span))
pca.fit(scaled_data.T)

# Extract first 5 eigenvalues and their corresponding eigenvectors
components = pca.components_
concise_eigenvectors = components[0:5]

weights_df = pd.DataFrame(columns = ["Location", "Code", "Weights"])
data_set = np.array(list(repo_for_SharesHolding.values()))

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

        combined = sorted(zip(stock_inner_codes, weights_list), key = lambda pair: -pair[1])
        count = 0
        for info in combined:
            if count < 5:
                weights_df = weights_df._append({
                "Location": i,
                    "Code": info[0],
                    "Weights": info[1]
                }, ignore_index = True)
                count+=1
            else:
                pass
    else:
        print("Optimization failed:", problem.status)


weights_df.to_csv("analytics\\StockWeights_t1", index = False)
print("Weights saved to analytics\\StockWeights_t1")