Time Series Simulator with capability to store results from parameter sweeps.

Make sure feature and target data is lined up time safe in each row (df.shift(1) on features).

Each parameter test has a short tag to identify the experiment, for example 'pca3lreg' is a pipe with pca that passes three dimensions as features to the regression.

A simple trading rule can be applied to predictions to generate mtm's.

Results_xr is an xarray object to store time-sereis based results like predictions, mtm's, and target benchmark data.  These results are added dimension 'tag' to organize results

Results_df stores summary statistics each parameter simulation such as mean, stdev, start/end dates, time of simulation, etc.  Tags are used for column headers.   

