This is a self-contained SPY prediction model that uses SPDR sector ETFs as features. As you can see in the image below, it's designed to provide standard textbook regression output and allow easy parameter adjustments to observe the results. In this version, the sector return lags are decayed with an exponentially weighted average (EWA) varying from 1 to 7 days, using data from the yfinance Python library.  See attached for the code (.ipynb or .py file) and pdf of output.  In google colab the notebook should run with "run all".  

Results_xr is an xarray object to store time-sereis based results like predictions, mtm's, and target benchmark data.  These results are added dimension 'tag' to organize results

  

