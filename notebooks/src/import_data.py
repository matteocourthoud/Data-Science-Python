import numpy as np
import pandas as pd

def import_data():
    """Import data
    
    Parameters
    ----------
    none

    Returns
    -------
    df_listings: DataFrame
    df_prices: DataFrame
    df: DataFrame
    """
    
    # Import listings data
    url_listings = "http://data.insideairbnb.com/italy/emilia-romagna/bologna/2021-12-17/data/listings.csv.gz"
    df_listings = pd.read_csv(url_listings)

    # Import pricing data
    url_prices = "http://data.insideairbnb.com/italy/emilia-romagna/bologna/2021-12-17/data/calendar.csv.gz"
    df_prices = pd.read_csv(url_prices, compression="gzip")

    # Cleaning prices data
    df_prices['price'] = df_prices['price'].str.replace('[$|,]', '', regex=True).astype(float)
    df_prices['date'] = pd.to_datetime(df_prices['date'], format='%Y-%m-%d')
    df_prices['available'] = df_prices['available']=='f'

    # Cleaning listings data
    df_listings['price'] = df_listings['price'].str.replace('[$|,]', '', regex=True).astype(float)
    df_listings = df_listings[['id', 'name', 'neighbourhood_cleansed', 'latitude', 'longitude', 'price',
                               'host_id', 'host_name', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
                               'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities',
                               'number_of_reviews', 'reviews_per_month', 'first_review', 'last_review', 'review_scores_rating']]

    df_listings = df_listings.rename(columns={'id': 'listing_id',
                                              'name': 'listing_title',
                                              'price': 'mean_price',
                                              'neighbourhood_cleansed': 'neighborhood'})

    # Merge
    df = pd.merge(df_listings, df_prices, on='listing_id', how='inner')
    
    # Return
    return df_listings, df_prices, df