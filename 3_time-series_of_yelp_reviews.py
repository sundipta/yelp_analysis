import pandas as pd
import numpy as np
import json
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


FILEPATH = "yelp_datasets/"
PITT_MEX_REVIEW_FILENAME = "pittsburgh_mexican_yelp_reviews.csv"


def get_month_year_list(df_col):
    """get_month_year_list will take a column of dates and create a list of
    consecutive month years from the min date to the max date in the format year-month"""
    mon_years = []
    for year in range(df_col.min().year, df_col.max().year + 1):
        mon_year = ""
        if year == df_col.min().year:
            for month in range(df_col.min().month, 13):
                mon_year = str(year) + "-" + str(month).zfill(2)
                mon_years.append(mon_year)
        else:
            for month in range(1, 13):
                mon_year = str(year) + "-" + str(month).zfill(2)
                mon_years.append(mon_year)
    return mon_years


def wrangle_data(filepath, pitt_mex_review_filename):
    reviews = pd.read_csv(filepath + pitt_mex_review_filename)

    reviews["date"] = pd.to_datetime(reviews["date"])
    reviews["month_year"] = reviews["date"].dt.to_period("M").astype(str)

    reviews_time_grouped = reviews.groupby(["month_year"]).count().reset_index()

    mon_years = get_month_year_list(reviews["date"])
    # steps to create a dataframe with the number of reviews per month from the first yelp review posted for this subset of restaurants
    reviews_merged = pd.DataFrame({"month_year": mon_years}).merge(
        reviews_time_grouped, how="left"
    )
    reviews_per_monyear = reviews_merged.filter(items=["month_year", "review_id"])
    reviews_per_monyear.fillna(0, inplace=True)
    reviews_per_monyear.rename(
        columns={"month_year": "date", "review_id": "number_of_reviews"}, inplace=True
    )
    reviews_per_monyear["date_time"] = pd.to_datetime(
        reviews_per_monyear["date"], format="%Y-%m"
    )

    # subset the data to where there are enough reviews to start seeing any seasonal trends
    recent_num_reviews = reviews_per_monyear[
        reviews_per_monyear["date_time"] >= np.datetime64("2011-01-01")
    ]

    return recent_num_reviews, reviews_per_monyear


def plot_seasonal_decomp(recent_num_reviews, reviews_per_monyear):
    # do an additive seasonal decomposition to check for seasonality
    decomposition = seasonal_decompose(
        recent_num_reviews["number_of_reviews"], model="additive", freq=12
    )
    decomposition.plot()
    plt.savefig("seasonal_decomposition_plot.png")

    # plot the number of reviews per month along with the seasonal decomposition model
    minor_ticks = [
        np.datetime64("2005-01-01"),
        np.datetime64("2007-01-01"),
        np.datetime64("2009-01-01"),
        np.datetime64("2011-01-01"),
        np.datetime64("2013-01-01"),
        np.datetime64("2015-01-01"),
        np.datetime64("2017-01-01"),
        np.datetime64("2019-01-01"),
    ]

    fig, ax = plt.subplots()

    plt.scatter(
        reviews_per_monyear.date_time,
        reviews_per_monyear["number_of_reviews"],
        c="#073763",
        label="Raw Data",
    )
    plt.plot(
        recent_num_reviews["date_time"],
        decomposition.trend + decomposition.seasonal,
        c="#FF6700",
        linewidth=2,
        label="Model",
    )
    plt.legend(["Seasonal + Trend", "Raw Data"])
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which="both")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Yelp Reviews per Month")
    plt.savefig("num_yelp_reviews_per_month.png")


if __name__ == "__main__":
    # set pyplot figure parameters
    parameters = {
        "figure.figsize": [15, 10],
        "axes.labelsize": 25,
        "axes.titlesize": 35,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "legend.handlelength": 2,
    }
    plt.rcParams.update(parameters)

    recent_data, all_data = wrangle_data(FILEPATH, PITT_MEX_REVIEW_FILENAME)
    plot_seasonal_decomp(recent_data, all_data)
