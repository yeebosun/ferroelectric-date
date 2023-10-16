# ferroelectric-date
1. Generate onehot coded datasets and perform correlation analysis using 'onehot.py'.
2. Separate datasets X and Y and divide them into 'X_train.xlsx' 'Y_train.xlsx' 'X_validation.xlsx' 'Y_validation.xlsx' 'X_test.xlsx' 'Y_test.xlsx'
3. Run 'XGboost.py' 'SVR.py' 'GBR.py' 'RF.py' and compare the performance of each algorithm. 
4. Run 'SHAP.py' and analyze using the model in 3.

When using the model, the data is first processed with onehot.py. Rename the generated data to 'X_prediction.xlsx' and run the 'prediction.py'. The result will be output as 'Y_prediction.xlsx'. Input and output samples are shown in 'X_prediction_example.xlsx' 'Y_prediction_example.xlsx'.

