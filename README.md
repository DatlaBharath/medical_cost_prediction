# medical_cost_prediction
1. **Import Libraries and Modules**: This section imports the necessary libraries and modules for data analysis, preprocessing, and modeling, as explained earlier.
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae, r2_score as r
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split as splt
import seaborn as sns
```
2. **Read Data**:
   ```python
   data = pd.read_csv('medical_cost.csv')
   ```
   This line reads a CSV file named 'medical_cost.csv' and loads it into a Pandas DataFrame called `data`.

3. **Identify Categorical Columns**:
   ```python
   mask = data.dtypes == np.object
   categorical_cols = data.columns[mask]
   ```
   It identifies columns in the DataFrame `data` with object (categorical) data types.

4. **Count Unique Categories in Categorical Columns**:
   ```python
   num_ohc_cols = (data[categorical_cols]
                   .apply(lambda x: x.nunique())
                   .sort_values(ascending=False))
   ```
   This code counts the number of unique values in each categorical column and stores the results in `num_ohc_cols` as a Series, sorted in descending order.

5. **One-Hot Encoding**:
   ```python
   data_ohc = data.copy()
   le = LabelEncoder()
   ohc = OneHotEncoder()
   ```
   It creates a copy of the original data in `data_ohc` and initializes LabelEncoder and OneHotEncoder.

6. **Loop for Encoding**:
   ```python
   for col in num_ohc_cols.index:
       dat = le.fit_transform(data_ohc[col]).astype(np.int)
       data_ohc = data_ohc.drop(col, axis=1)
       new_dat = ohc.fit_transform(dat.reshape(-1,1))
       n_cols = new_dat.shape[1]
    col_names = ['_'.join([col, str(x)]) for x in range(n_cols)]
    print(col_names)
    new_df = pd.DataFrame(new_dat.toarray(), index=data_ohc.index, columns=col_names)
    data_ohc = pd.concat([data_ohc, new_df], axis=1)
   ```
   - This loop iterates through the categorical columns identified earlier.
   - It first integer encodes the categorical values using `LabelEncoder`.
   - The original column is dropped from the DataFrame.
   - The integer-encoded data is then one-hot encoded using `OneHotEncoder`.
   - New column names are created for the one-hot encoded data.
   - The new one-hot encoded data is appended to the original DataFrame.

7. **Split Data and Apply Polynomial Features**:
   ```python
   X = data_ohc.drop('charges', axis=1)
   y = data_ohc['charges']
   pf = PolynomialFeatures(degree=2, include_bias=False)
   X_pf = pf.fit_transform(X)
   X_train, X_test, y_train, y_test = splt(X_pf, y, test_size=0.3, random_state=42)
   ```
   - The data is split into features (`X`) and the target variable (`y`).
   - Polynomial features (degree 2) are generated for the features and stored in `X_pf`.
   - The data is split into training and testing sets using `train_test_split`.

8. **Standardization of Features**:
   ```python
   s = StandardScaler()
   X_train_s = s.fit_transform(X_train)
   ```
   - The features in the training set (`X_train`) are standardized using `StandardScaler`.

9. **Linear Regression Modeling**:
   ```python
   lr = LinearRegression()
   lr.fit(X_train_s, y_train)
   X_test_s = s.transform(X_test)
   y_pred = lr.predict(X_test_s)
   ```
   - A linear regression model is trained on the standardized training data.
   - The model's predictions are made on the standardized test data.

10. **Evaluation and Visualization**:
    ```python
    r(y_test, y_pred)
    ```
    The code calculates the R-squared score between the true target values (`y_test`) and the predicted values (`y_pred`).

11. **Set Seaborn Plot Style**:
    ```python
    sns.set_context('talk')
    sns.set_style('ticks')
    sns.set_palette('dark')
    ```
    This code sets the style and context for Seaborn plots.

12. **Scatter Plot**:
    ```python
    ax = plt.axes()
    ax.scatter(y_test, y_pred, alpha=.5)
    ```
    A scatter plot is created to visualize the relationship between the true target values (`y_test`) and the predicted values (`y_pred`).

13. **Lasso and Ridge Regression Modeling**:
    ```python
    las007 = Lasso(alpha=1, max_iter=1000000)
    las007.fit(X_train_s, y_train)
    y_pred = las007.predict(X_test_s)

    red = Ridge(alpha=1)
    red.fit(X_train_s, y_train)
    y_pred_r = red.predict(X_test_s)
    ```
    Lasso and Ridge regression models are trained on the standardized training data, and predictions are made for both.

14. **Evaluation of Lasso and Ridge Models**:
    ```python
    r(y_test, y_pred)
    r(y_test, y_pred_r)
    ```
    The R-squared scores are calculated for both the Lasso and Ridge regression models.
