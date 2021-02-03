import pandas as pd
import geopandas as gpd
import plotnine as p

FILEPATH = "yelp_datasets"
RESTAURANT_DATA_FILE = "pittsburgh_mexican_restaurants.csv"
PITTSBURGH_SHAPEFILE = "Neighborhoods_.geojson"


def plot_restaurants_per_neighborhood(
    filepath, restaurant_data_file, pittsburgh_shapefile
):
    mexican_restaurants = pd.read_csv(filepath + restaurant_data_file)

    gdf = gpd.GeoDataFrame(
        mexican_restaurants,
        geometry=gpd.points_from_xy(
            mexican_restaurants.longitude, mexican_restaurants.latitude
        ),
    )

    restaurant_locations = gdf.filter(items=["geometry"])

    # import Pittsburgh neighborhood shapefile
    neighborhood_polygons = gpd.read_file(pittsburgh_shapefile).filter(
        items=["hood", "hood_no", "geometry"]
    )

    # spatial join to figure out which neighborhood each restaurant is in
    restaurants_in_polys = gpd.sjoin(
        restaurant_locations, neighborhood_polygons, how="inner", op="intersects"
    )

    restaurants_counted = restaurants_in_polys.groupby("hood_no").count().reset_index()
    restaurants_in_hoods = restaurants_counted.filter(items=["hood_no", "hood"])
    restaurants_in_hoods.rename(columns={"hood": "num_restaurants"}, inplace=True)

    restaurants_per_shape = gpd.GeoDataFrame(
        pd.merge(neighborhood_polygons, restaurants_in_hoods, how="left")
    )

    restaurant_map = (
        p.ggplot(restaurants_per_shape)
        + p.geom_map(p.aes(fill="num_restaurants"))
        + p.scale_colour_gradient(low="white", high="black")
        + p.theme(
            panel_background=p.element_rect(fill="white"),
            axis_text_x=p.element_blank(),
            axis_text_y=p.element_blank(),
            axis_ticks_major_x=p.element_blank(),
            axis_ticks_major_y=p.element_blank(),
        )
    ) + p.scale_fill_gradient(low="#efefef", high="#073763", name="# Restaurants")

    restaurant_map.save("restaurant_map.png")


if __name__ == "__main__":
    plot_restaurants_per_neighborhood(
        FILEPATH, RESTAURANT_DATA_FILE, PITTSBURGH_SHAPEFILE
    )
