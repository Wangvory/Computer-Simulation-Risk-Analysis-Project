# Computer_Simulation_Project_Submission
 Computer Simulation & Risk Analysis Project
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
