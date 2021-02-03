import pandas as pd
import json

YELP_DATA_FILEPATH = "yelp_datasets/"
BUSINESS_DATASET_FILE = "yelp_academic_dataset_business.json"
REVIEW_DATASET_FILE = "yelp_academic_dataset_review.json"

RANKED_RESTAURANTS = "pittsburgh_mexican_restaurants.csv"
PITT_MEX_REVIEW_JSON_FILENAME = "pittsburgh_mexican_yelp_reviews.json"
PITT_MEX_REVIEW_CSV_FILENAME = "pittsburgh_mexican_yelp_reviews.csv"


def parse_datasets(
    yelp_data_filepath,
    business_dataset_file,
    review_dataset_file,
    ranked_restaurants,
    pitt_mex_review_json_filename,
    pitt_mex_review_csv_filename,
):
    # open business data, read line by line, and append to a list
    business_dict_list = []
    with open(yelp_data_filepath + business_dataset_file) as f:
        for line in f:
            business_dict = json.loads(line)
            business_dict_list.append(business_dict)

    # create a pandas dataframe of business, then subset to Pittsburgh
    business_df = pd.DataFrame(business_dict_list)
    pittsburgh_business_df = business_df[business_df["city"] == "Pittsburgh"]

    # select restaurants that have a cetegory containing the string 'Mexican'
    mexican_restaurants = pittsburgh_business_df[
        pittsburgh_business_df["categories"].str.contains("Mexican", na=False)
    ]
    # rank by stars and number of reviews and save to csv
    mexican_restaurants.sort_values(["stars", "review_count"], ascending=False).to_csv(
        yelp_data_filepath + ranked_restaurants, index=False
    )

    # subset the full dataset of reviews to just those for Pittsburgh Mexican restaurants and write to a new file
    business_ids = set(mexican_restaurants["business_id"].values)

    # open the review dataset and save reviews for pertinent restaurants in another file
    with open(yelp_data_filepath + review_dataset_file, "r") as f:
        with open(yelp_data_filepath + pitt_mex_review_json_filename, "w") as g:
            for line in f:
                json_dict = json.loads(line)
                if json_dict["business_id"] in business_ids:
                    g.write(line)

    # open the json file with just the Pittsburgh Mexican restaurant reviews
    review_dict_list = []
    with open(yelp_data_filepath + pitt_mex_review_json_filename) as f:
        for line in f:
            review_dict = json.loads(line)
            review_dict_list.append(review_dict)
    reviews = pd.DataFrame(review_dict_list)

    reviews.to_csv(yelp_data_filepath + pitt_mex_review_csv_filename, index=False)


if __name__ == "__main__":
    parse_datasets(
        YELP_DATA_FILEPATH,
        BUSINESS_DATASET_FILE,
        REVIEW_DATASET_FILE,
        RANKED_RESTAURANTS,
        PITT_MEX_REVIEW_JSON_FILENAME,
        PITT_MEX_REVIEW_CSV_FILENAME,
    )
