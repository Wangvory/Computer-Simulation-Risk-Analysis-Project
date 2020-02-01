# Computer_Simulation_Project_Submission
 Computer Simulation & Risk Analysis Project
# Project1 Index Construction - Stock Sentiment Index(SSX)
## Index Constituents Selection
Apply BeautifulSoup package to scrap latest 10 news about S&P 500 companies
Apply Sentiment Intensity Analyzer package to analyze the sentiment scores
	Positive Words“Gain”，“High Return”
	Negative Words  ”Worry” , ”Lose”
 Automatically select top 15(Project1)/30(Project2) companies with the most positive latest news
Index Strat  Price=100, Changed By Daily Weighted Return
## SSX Construction & Return
Set individual stock weight by market cap
Graph shows sentiment scores and market cap
Stocks contain Media & Entertainment, Manufacture and Defense industry
## ETF Optimization – SSX Index
calculate optimized weighted in ETF
Apply binary combination function to create different binary lists without duplications and omissions
After creating matrix of all possible combination, we set different lists as the bound in optimizer to find minimized Tracking Error.
How many stocks (n) in ETF 
Build a loop for n in range(5,11)
For each n, we will calculate a minimum Tracking Error
Draw a plot for different n with minimum TE
n = 10 is our result
## Train Test Split & Conclusion
Train 2007-2013, Rolling Test 2013-2019. Window = 30 days, Optimized weight change time period = 30 days
Realized TE always a little bit higher than Forecast TE, TEs (when n=5) are higher than TEs (when n=10), Forecasted TE obviously smaller than realized TE, when n=10
# Project 2 Portfolio Construction
## Index Selection
SPDR Gold Shares (GLD) -- Commodity
EUR/USD (EURUSD=X) -- Currency 
iShares Core U.S. Aggregate Bond ETF (AGG) -- U.S. government & corporate bond 
Invesco Emerging Markets Sovereign Debt ETF (PCY) -- Emerging country’s government bond
Stock Sentiment Index (SSX) -- Sentiment Index created by ourselves (contain 30 stocks)
All of them are irrelevant and with high Sharpe Ratios
## Time Split
Window = 90 days
Lamda = 0.94
Varmodel = ma
Leverage = True
### Period 1 – 2008/03/31 – 2009/03/31
During Crisis:
Return of risk parity was lower than return of equal weighted portfolio
No winners during crisis
### Period 2: 2009/04/01 – 2018/12/31
After Crisis:
Return of risk parity was higher than return of equal weighted portfolio
SSX - Highest cumulative return
Risk Parity - Highest Sharpe Ratio
## Conclusions:
We did not take period one into consideration in our risk parity portfolio
Risk parity has the highest Sharp Ratio during period 2 no matter with or without leverage
Only risk parity portfolio has different Sharp Ratio (higher) when the leverage is 200%
Investment insight: Risk parity portfolio has the lowest risk and highest sharp ratio
