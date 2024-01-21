import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score 
from scipy.optimize import curve_fit
from scipy.stats import t


def load_clean_transpose_data(file_path):
    """
        Load, clean, and transpose data from a CSV file.

        Parameters:
        - file_path (str): Path to the CSV file containing the data.

        Returns:
        - df (pd.DataFrame): Original DataFrame with cleaned and imputed data.
        - transposed_df (pd.DataFrame): Transposed DataFrame.

        """
    # Read the data
    df = pd.read_csv(file_path , header = None , names = ["Year" , "Time Code" , "Country Name" , "Country Code" ,
                                                      "Forest Area" , "Agricultural Land" , "Arable Land"])

    # Convert numeric columns to numeric values
    numeric_columns = ["Forest Area" , "Agricultural Land" , "Arable Land"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric , errors = 'coerce')

    # Impute missing values with mean
    imputer = SimpleImputer(strategy = 'mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Transpose the data
    transposed_df = df.transpose()

    return df , transposed_df


# Time Series Analysis
def model_func(x , a , b , c):
    """
        Compute the output of a quadratic function.

        Parameters:
        - x (float or array-like): Input value or array of input values.
        - a (float): Coefficient of the quadratic term.
        - b (float): Coefficient of the linear term.
        - c (float): Constant term.

        Returns:
        - float or array-like: Output value(s) of the quadratic function.
    """
    return a * x ** 2 + b * x + c


def err_ranges(cov_matrix , x , model_func , params , confidence = 0.95):
    """
        Calculate confidence intervals for the predicted values of a model.

        Parameters:
        - cov_matrix (numpy.ndarray): Covariance matrix of the model parameters.
        - x (array-like): Input values for the model.
        - model_func (callable): Model function that takes 'x' and 'params' as arguments.
        - params (array-like): Model parameters.
        - confidence (float, optional): Confidence level for the intervals (default is 0.95).

        Returns:
        - lower_bound (numpy.ndarray): Lower bounds of the confidence intervals for each predicted value.
        - upper_bound (numpy.ndarray): Upper bounds of the confidence intervals for each predicted value.
    """
    alpha = 1 - confidence
    n = len(x)
    df = max(0 , n - len(params))
    t_val = np.abs(t.ppf(alpha / 2 , df))

    predicted_values = model_func(x , *params)
    residuals = x - predicted_values
    resid_std = np.std(residuals , ddof = len(params))
    margin_of_error = t_val * resid_std
    lower_bound = predicted_values - margin_of_error
    upper_bound = predicted_values + margin_of_error

    return lower_bound , upper_bound


# Clustering
data_path = '5ba7de27-a159-4615-aa73-66492e9e0614_Data.csv'
original_data , cleaned_data = load_clean_transpose_data(data_path)

# Select relevant columns for clustering
features = ["Forest Area" , "Agricultural Land" , "Arable Land"]
df_features = original_data[features]

# Normalize the data
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_features)

# Perform k-means clustering
k = 3
kmeans = KMeans(n_clusters = k , random_state = 42)
original_data["Cluster"] = kmeans.fit_predict(df_normalized)

# Calculate silhouette score
silhouette_avg = silhouette_score(df_normalized , original_data["Cluster"])
print(f"Silhouette Score: {silhouette_avg}")

# Explore characteristics of each cluster
cluster_characteristics = original_data.groupby("Cluster")[["Forest Area" ,
                                                            "Agricultural Land" , "Arable Land"]].mean().reset_index()
print("Cluster Characteristics:")
print(cluster_characteristics)

# Inverse transform the cluster centers to the original scale
original_scale_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Plotting the clusters and cluster centers
plt.figure(figsize = (10 , 6))
for cluster in range(k):
    cluster_data = original_data[original_data["Cluster"] == cluster]
    plt.scatter(cluster_data["Forest Area"] , cluster_data["Agricultural Land"] ,
                label = f'Cluster {cluster}' , alpha = 0.7)

plt.scatter(original_scale_centers[: , 0] , original_scale_centers[: , 1] ,
            marker = 'X' , s = 200 , c = 'red' , label = 'Cluster Centers (Original Scale)')
plt.title("K-Means Clustering of Agricultural and Forest Data" , fontsize = 20)
plt.xlabel("Forest Area" , fontsize = 20)
plt.ylabel("Agricultural Land" , fontsize = 20)
plt.legend()
plt.show()


countries = ['Pakistan' , 'Australia' , 'China']

for country in countries:
    # Corrected the column name to 'Year'
    country_df = original_data[(original_data['Country Name']
                                == country) & (pd.to_numeric(original_data['Year'] ,
                                                             errors='coerce') <= 2030)]

    # Adjusted column names for years and forest_area
    years = country_df['Year'].astype(float)
    forest_area = country_df['Forest Area'].astype(float)

    params , covariance = curve_fit(model_func , years , forest_area)
    future_years = np.arange(2000 , 2031 , 1)
    predicted_forest_area = model_func(future_years , *params)

    lower_bound , upper_bound = err_ranges(covariance , future_years
                                           , model_func , params , confidence = 0.95)

    plt.figure()
    plt.scatter(years , forest_area , label = 'Actual Data')
    plt.plot(future_years , predicted_forest_area , label = 'Best Fitting Function' , color = 'red')
    plt.fill_between(future_years , lower_bound , upper_bound , color = 'red' , alpha = 0.2 ,
                     label = 'Confidence Interval (95%)')
    plt.xlabel('Year' , fontsize = 20)
    plt.ylabel(f'Forest area (% of land area) - {country}' , fontsize = 20)
    plt.title(f'Forest Area Prediction with Confidence Interval - {country}' , fontsize = 20)
    plt.legend()
    plt.show()
