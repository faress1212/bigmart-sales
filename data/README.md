# Data

This project uses the **Big Mart Sales** dataset (train split).

From Kaggle:
https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets

**Dataset summary:**
- 8,523 rows, 12 columns
- Collected in 2013 for 1,559 products across 10 stores in different cities
- Columns: `Item_Identifier`, `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`, `Outlet_Identifier`, `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`, `Item_Outlet_Sales`
- `Item_Outlet_Sales` is the target column: both the main subject of the EDA plots and the value predicted by the regression models in `src/models/train_models.py`
