#manutd-xg-analysis

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

#Aim : To understand differences in Manchester united's finishing efficiency
#over time, and under different managers.

df = pd.read_csv(
"C:/Users/Oliver Plumpton/Desktop/Man Utd Analysis/Data/final_matches.csv"
)

utd = df[df["team"] == "Manchester United"]

utd = utd.rename(columns={"gf": "Goals"})
utd = utd.rename(columns={"xg": "Expected Goals"})

utd["Shot Efficiency"] = (utd["Goals"]/utd["Expected Goals"])

utd["Season"] = ""

utd.loc[utd["date"] <= "2021-05-23", "Season"] = "2020/21"

utd.loc[(utd["date"] > "2021-05-23") & (utd["date"] <= "2022-05-22"), "Season"] = "2021/22"

utd.loc[(utd["date"] > "2022-05-22") & (utd["date"] <= "2023-05-28"), "Season"] = "2022/23"

utd.loc[(utd["date"] > "2023-05-28") & (utd["date"] <= "2024-05-19"), "Season"] = "2023/24"

utd.loc[utd["date"] > "2024-05-19", "Season"] = "2024/25"

year = (
    utd.groupby("Season")
    .agg({
        "Goals": "sum",
        "Expected Goals": "sum",
        "Shot Efficiency": ["mean", "std"]
        })
    )


year[["Goals","Expected Goals"]].plot(
    kind="bar",
    figsize=(10,5)
    )

plt.title("Manchester United Goals vs Expected Goals by Season")
plt.xlabel("Season")
plt.ylabel("Goals")
plt.legend(title="")

plt.show()


utd["Manager"] = ""

utd.loc[utd["date"] <= "2021-11-21", "Manager"] = "Solskjaer"

utd.loc[(utd["date"] < "2021-12-02") & (utd["date"] >= "2021-11-21"), "Manager"] = "Carrick"

utd.loc[(utd["date"] < "2022-05-22") & (utd["date"] >= "2021-12-02"), "Manager"] = "Rangnick"

utd.loc[(utd["date"] < "2024-10-28") & (utd["date"] >= "2022-05-22"), "Manager"] = "Ten Hag"

utd.loc[(utd["date"] < "2024-11-10") & (utd["date"] >= "2024-10-28"), "Manager"] = "Van Nistelrooy"

utd.loc[utd["date"] >= "2024-11-10", "Manager"] = "Amorim"


manager_analysis = (
    utd.groupby("Manager")
    .agg({
        "Goals": "sum",
        "Expected Goals": "sum",
        "Shot Efficiency": ["mean", "std"]
        })
    )
print(manager_analysis)

season_analysis = (
    utd.groupby("Season")
    .agg({
        "Goals": ["mean", "std"],
        "Expected Goals": ["mean", "std"],
        "Shot Efficiency": ["mean", "std"]
        })
    )
print(season_analysis)

#note that the shot_efficiency mean is not the sum of goals/ sum of xg as this would
#the seasons shot efficiency, instead it shows the average efficiency per game.
#highlight this through amorim and solskjaer comparisons


for manager in utd["Manager"].unique():
    manager_data = utd[utd["Manager"] == manager]

    if len(manager_data) < 10:
        continue

    result = linregress(
        manager_data["Expected Goals"],
        manager_data["Goals"]
    )

    print(manager)
    print("Slope:", round(result.slope,2))
    print("R²:", round(result.rvalue**2,2))
    print()

sol = utd[utd["Manager"] == "Solskjaer"]

plt.figure(figsize=(7,5))

plt.scatter(sol["Expected Goals"], sol["Goals"])

plt.xlabel("Expected Goals")
plt.ylabel("Goals")
plt.title("Solskjaer: Goals vs xG")

plt.xlim(0, 5)
plt.ylim(0, 10)

plt.show()

X = sol[["Expected Goals"]]
y = sol["Goals"]

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

print("R²:", model.score(X_poly, y))

x_range = np.linspace(
    sol["Expected Goals"].min(),
    sol["Goals"].max(),
    100
)

y_pred = model.predict(
    poly.transform(x_range.reshape(-1,1))
)

plt.scatter(sol["Expected Goals"], sol["Goals"])
plt.plot(x_range, y_pred)

plt.xlabel("Expected Goals")
plt.ylabel("Goals")
plt.title("Solskjaer Goals vs xG (Quadratic Fit)")

plt.show()


print("This analysis suggests that, as demonstrated by the bar chart comparing "
      "goals with expected goals, the players were much more effecient accross "
      "the whole season in 2020/21 at capitalising on chances created, this then "
      "reduced going into 2021/22, despite keeping a positive goals vs xg "
      "difference, it was much less, similar to that of the 2023/24 season. We also see, "
      "despite differences in total expected goals, that goals scored remained "
      "consistent from 2021/22 to 2023/24, with 2024/25 having the lowest "
      "expected goals, as well as chances converted the hightened goals scored"
      "in the 2020/21 season coincides with the arrival of Bruno, and he had his "
      "statistical greatest season in terms of goals and assists. This may have resulted"
      "in the clubs hightened attacking output, however this cannot be confirmed by this "
      "analysis. In 2021, Ronaldo rejoined the club, and"
      "despite his 18 goals, the club saw its second lowest total goal scored from "
      "the past 5 years, suggesting that the performance of sole players is unlikely "
      "to offset team attacking insufficiencies, and was likely a result of mangerial "
      "instability at this time")

print()

print("the bar chart does a good job at capturing the season's picture as a whole, "
      "however struggles to demonstrate game to game differences. To do so, I "
      "added a new column to the data called Shot Efficiency, which calculates "
      "goals per game, divided by the expected goals per game. A value of 1 here "
      "would mean in this match, the players scored the exact amount of expected "
      "goals, and the higher this value, the more clinical they are relative to "
      "the quality of the chance. I then calculated the mean of this per season. "
      "It is worth noting that this is not the same as seasonal shot efficiency, "
      "given by total goals over total expected goals. Instead, this shows the "
      "average clincality per game. Because of this, it can be skewed by some games "
      "with a vast difference in goals and expected goals, so I also calculated the "
      "standard deviation, to understand more how likely this is to change per match. "
      "From this data we see now that the highest mean shot efficiency actually came "
      "from the 2023/24 season at 1.19 to 2.d.p, with the 2020/21 season coming in "
      "second, suggesting that perhaps the vast difference in seasonal goals vs "
      "expected goals came from a few matches with a large difference, however on the "
      "whole the finishing remained closer to that of other seasons than previously "
      "expected. However, the standard deviation was approximately 25% higher in the "
      "2023/24 season than the 2020/21, so the higher mean is likely attributed "
      "to single match outliers in shot efficiency, as well as noting that the "
      "difference in mean was only small at a 4.3% increase.")

print()

print("Perhaps the more interesting comparisons arise from difference in managers "
      "and how their tactics influence the type of chances created, as well as "
      "squad selection impacting shot efficiency. To look at this, I again compared "
      "the shot efficiency mean and standard deviation. Once disregarding the "
      "care-taker and interim managers, We can compare the remaining managers"
      "We see that Amorim had "
      "the greatest mean, with also the greatest standard deviation. suggesting "
      "that game to game was very unpredictable with respect to the shot efficiency "
      "of the players. This is more than likely explained by the fact that this was "
      "a transitional period for the club, not only with a new manager, but a "
      "manager that played 3 at the back, and so players had alot more than "
      "usual to learn for their new role in the side. This data also only captures "
      "the early stages of Amorim's tenure, and so is harder to compare to managers "
      "such as Ten Hag, where this timeline captures the whole of his life-cycle at "
      "the club. We see Solskjaer has the next highest mean, with a much lower "
      "standard deviation. This alligns with the earlier findings of the 2020/21 "
      "season, where united were much more clinical over the season. It should be "
      "noted that this only includes the last season of an almost 3 year reign "
      "and so he would have been familiar with the players strengths, useful to "
      "extract the highest performancefrom them. Combining these findings with"
      "the columns demonstrating the total goals and expected goals from each "
      "managerm, we can conclude that solskjaer managed the most clinical man united"
      "between the 2020/21 and 2024/25 season. This could highlight a difference "
      "in tactics between Solskjaer and other managers, perhaps suggesting he is more"
      "threating on transition, and can capitalise on fewer chances created from "
      "counter attacks. To confirm this, I investigated any possible regression "
      "trends between goals and expected goals, to see how much an increase in "
      "expected goals resulted in a likely increase in goals scored, i then also"
      "calculated the R^2 value of this linear regression model, to evaluate "
      "the variance in the model. I found that Solskjaer demonstrated a 2.25 larger "
      "gradeint when compared to the second largest of 0.77 obtained by "
      "Rangnick. This suggests that Solskjaer's team found disproportionately more "
      "success in scoring from an increase in expected goals. It also has a strong "
      "linear correlation, with an R^2 value of 0.61, suggesting that expected goals "
      "were a good predictor of goals scored, much more than other managers, whose "
      "combined highest R^2 value was Ten Hag at 0.32. Since the gradient of this model "
      "was high at 1.73, I decided to plot a scatter graph, which each individual match "
      "showing goals against expected goals. From inspection of this, "
      "although a clear positive correlation could be seen, it seemed to support "
      "an exponential model may be more suited to the data. I then replotted the data "
      "with an quadratic model predictor, and the data seemed to better match this model "
      "as well as returning an R^2 value of 0.65, a modest improvement on the 0.61 "
      "obtained using the linear model, suggesting some evidence of non-linearity, "
      "however, the linear model still captures much of the relationship . Finally "
      "it is worth noting that Amorim saw the lowest R^2 value of 0.03, suggesting that "
      "expected goals had next to no correlation to goals scored, suggesting that "
      "the high mean of shot efficiency saw earlier, was heavily influenced by "
      "individual outliers in games, and the conversion rate of players under Amorim "
      "was not reliable. It will however, also be heavily iompacted by the smaller "
      "sample size of Amorim, compared to other mangers and so this should be interpreted "
      "cautiously")

print()

print("Overall, the combination of seasonal goal totals, shot-efficiency statistics and "
      "regression analysis consistently identifies Solskjaer's side as the most clinical "
      "attacking team over the period analysed. While tactical differences are a plausible "
      "explanation, the influence of squad quality, player form and managerial stability "
      "should also be considered when interpreting these findings.")
