# Load libraries ---------------------------------------------

from datetime import datetime, timedelta
from dateutil.easter import easter
from data_preprocessing.dataset_specification import DatasetSpecification

import pandas as pd
import numpy as np
# ------------------------------------------------------------


class DataPreprocessingToolkit(object):

    def __init__(self):
        dataset_specification = DatasetSpecification()

        self.sum_columns = dataset_specification.get_sum_columns()
        self.mean_columns = dataset_specification.get_mean_columns()
        self.mode_columns = dataset_specification.get_mode_columns()
        self.first_columns = dataset_specification.get_first_columns()

        self.nights_buckets = dataset_specification.get_nights_buckets()
        self.npeople_buckets = dataset_specification.get_npeople_buckets()
        self.room_segment_buckets = dataset_specification.get_room_segment_buckets()

        self.arrival_terms = dataset_specification.get_arrival_terms()

        self.item_features_columns = dataset_specification.get_items_df_feature_columns()

    # #########################
    # Entire datasets functions
    # #########################

    def fix_date_to(self, df):
        df.loc[:, "date_to"] = df["date_to"].apply(lambda x: x + timedelta(days=1))
        return df

    def add_length_of_stay(self, df):
        df.loc[:, "length_of_stay"] = (df["date_to"] - df["date_from"]).dt.days
        return df

    def add_book_to_arrival(self, df):
        df.loc[:, "book_to_arrival"] = (df["date_from"] - df["booking_date"]).dt.days
        return df

    def add_nrooms(self, df):
        df.loc[:, "n_rooms"] = 1
        return df

    def add_weekend_stay(self, df):
        s = df["date_from"].dt.dayofweek
        e = df["date_to"].dt.dayofweek
        dt = (df["date_to"] - df["date_from"]).dt.days
        df.loc[:, "weekend_stay"] = (((s >= 4) & (s != 6)) | (e >= 5) | ((e < s) & (s != 6)) | (dt >= 6))
        df.loc[:, "weekend_stay"] = df["weekend_stay"].replace({True: 'True', False: 'False'})
        return df

    def add_night_price(self, df):
        s = df["length_of_stay"]
        df.loc[:, "night_price"] = round(((df["accomodation_price"] / df["n_rooms"]) / s),2)
        return df

    def clip_book_to_arrival(self, df):
        df.loc[:, "book_to_arrival"] = np.maximum(df["book_to_arrival"], 0)
        return df

    def sum_npeople(self, df):
        df.loc[:, "n_people"] = np.maximum(df["n_people"] + df["n_children_1"] + df["n_children_2"] + df["n_children_3"], 1)
        return df

    def filter_out_company_clients(self, df):
        df = df.loc[df["is_company"] == 0]
        return df

    def filter_out_long_stays(self, df):
        df = df.loc[df["length_of_stay"] <= 21]
        return df

    def leave_one_from_group_reservations(self, df):
        unique_group_rows = []

        df.loc[:, "group_id"] = df["group_id"].fillna(-1)

        group_ids = []
        for idx, row in df.iterrows():
            if row["group_id"] != -1:
                if row["group_id"] not in group_ids:
                    unique_group_rows.append(row)
                    group_ids.append(row["group_id"])
            else:
                unique_group_rows.append(row)

        cleaned_dataset = pd.DataFrame(unique_group_rows, columns=df.columns)

        return df

    def aggregate_group_reservations(self, df):
        non_group_reservations = df.loc[df["group_id"] == "",
                                        self.sum_columns + self.mean_columns + self.mode_columns + self.first_columns]
        group_reservations = df.loc[df["group_id"] != ""]

        agg_datasets = [group_reservations.loc[:, ["group_id"] + self.sum_columns].groupby("group_id").sum(),
                        group_reservations.loc[:, ["group_id"] + self.mean_columns].groupby("group_id").mean(),
                        group_reservations.loc[:, ["group_id"] + self.mode_columns].groupby("group_id").agg(lambda x: x.value_counts().index[0]),
                        group_reservations.loc[:, ["group_id"] + self.first_columns].groupby("group_id").first()]

        group_reservations = agg_datasets[0]
        for i in range(1, len(agg_datasets)):
            group_reservations = group_reservations.merge(agg_datasets[i], on="group_id")

        group_reservations = group_reservations.reset_index(drop=True)

        df = pd.concat([non_group_reservations, group_reservations])

        return df

    def leave_only_ota(self, df):
        df = df.loc[df.loc[:, "Source"].apply(lambda x: "booking" in x.lower() or "expedia" in x.lower())]
        return df

    def map_date_to_term_datasets(self, df):
        df.loc[:, "date_from"] = df["date_from"].astype(str).apply(lambda x: x[:10])
        df.loc[:, 'term'] = df['date_from'].apply(lambda x: self.map_date_to_term(x))
        return df

    def map_length_of_stay_to_nights_buckets(self, df):
        df.loc[:, 'length_of_stay_bucket'] = df['length_of_stay'].apply(lambda x: self.map_value_to_bucket(x, self.nights_buckets))
        return df

    def map_night_price_to_room_segment_buckets(self, df):
        df.loc[:, 'room_segment'] = df['night_price'].apply(lambda x: self.map_value_to_bucket(x, self.room_segment_buckets))
        return df

    # def map_night_price_to_room_segment_buckets(self, df):
    #    night_prices = df.loc[df['accomodation_price'] > 1].groupby('room_group_id')['night_price'].mean().reset_index()
    #    night_prices.columns = ['room_group_id', 'room_night_price']
    #    df = pd.merge(df, night_prices, on=['room_group_id'], how='left')
    #    df.loc[df['room_night_price'].isnull(), 'room_night_price'] = 0.0
    #    df.loc[:, 'room_segment'] = df['room_night_price'].apply(
    #        lambda x: self.map_value_to_bucket(x, self.room_segment_buckets))
    #    df = df.drop(columns=['room_night_price'])
    #    return df

    def map_npeople_to_npeople_buckets(self, df):
        df.loc[:, 'n_people_bucket'] = df['n_people'].apply(lambda x: self.map_value_to_bucket(x, self.npeople_buckets))
        return df

    def map_item_to_item_id(self, df):
        df.loc[:, 'item'] = df[self.item_features_columns].astype(str).agg(' '.join, axis=1)

        ids = df['item'].unique().tolist()
        mapping = {ids[i]: i for i in range(len(ids))}

        df['item_id'] = df['item'].apply(lambda x: mapping[x])

        return df

    def add_interaction_id(self, df):
        df.loc[:, 'interaction_id'] = range(df.shape[0])
        return df

    # ################
    # Column functions
    # ################

    def bundle_period(self, diff):
        diff = float(diff)
        if int(diff) < 0:
            return "<0"
        elif int(diff) <= 7:
            return diff
        elif 7 < int(diff) <= 14:
            return "<14"
        elif 14 < int(diff) <= 30:
            return "<30"
        elif 30 < int(diff) <= 60:
            return "<60"
        elif 60 < int(diff) <= 180:
            return "<180"
        elif int(diff) > 180:
            return ">180"

    def bundle_price(self, price):
        mod = 300.0
        return int((price + mod / 2) / mod) * mod

    def map_date_to_season(self, date):
        day = int(date[8:10])
        month = int(date[5:7])
        if (month == 12 and day >= 21) or (month == 1) or (month == 2) or (month == 3 and day <= 19):
            return "Winter"
        if (month == 3 and day >= 20) or (month == 4) or (month == 5) or (month == 6 and day <= 20):
            return "Spring"
        if (month == 6 and day >= 21) or (month == 7) or (month == 8) or (month == 9 and day <= 22):
            return "Summer"
        if (month == 9 and day >= 23) or (month == 10) or (month == 11) or (month == 12 and day <= 20):
            return "Autumn"

    def map_value_to_bucket(self, value, buckets):
        if value == "":
            return str(buckets[0]).replace(", ", "-")
        for bucket in buckets:
            if bucket[0] <= value <= bucket[1]:
                return str(bucket).replace(", ", "-")

    def map_date_to_term(self, date):

        m = int(date[5:7])
        d = int(date[8:10])
        term = None

        for arrival_term in self.arrival_terms:
            if arrival_term == "Easter":
                year = int(date[:4])
                easter_date = easter(year)
                easter_start = easter_date + timedelta(days=-4)
                easter_end = easter_date + timedelta(days=1)
                esm = easter_start.month
                esd = easter_start.day
                eem = easter_end.month
                eed = easter_end.day
                if ((m > esm) or (m == esm and d >= esd)) and ((m < eem) or (m == eem and d <= eed)):
                    term = arrival_term
                    break

            elif arrival_term == "NewYear":
                sm = self.arrival_terms[arrival_term][0]["start"]["m"]
                sd = self.arrival_terms[arrival_term][0]["start"]["d"]
                em = self.arrival_terms[arrival_term][0]["end"]["m"]
                ed = self.arrival_terms[arrival_term][0]["end"]["d"]
                if ((m > sm) or (m == sm and d >= sd)) or ((m < em) or (m == em and d <= ed)):
                    term = arrival_term
                    break

            else:
                is_match = False

                for i in range(len(self.arrival_terms[arrival_term])):
                    sm = self.arrival_terms[arrival_term][i]["start"]["m"]
                    sd = self.arrival_terms[arrival_term][i]["start"]["d"]
                    em = self.arrival_terms[arrival_term][i]["end"]["m"]
                    ed = self.arrival_terms[arrival_term][i]["end"]["d"]
                    if ((m > sm) or (m == sm and d >= sd)) and ((m < em) or (m == em and d <= ed)):
                        term = arrival_term
                        is_match = True
                        break

                if is_match:
                    break

        return term

    def map_dates_to_terms(self, dates):

        terms = []
        for date in dates:
            term = self.map_date_to_term(date)
            terms.append(term)

        return terms

    def filter_out_historical_dates(self, date_list):
        """
        Filters out past dates from a list of dates.
        """
        future_dates = []

        for date in date_list:
            if date >= datetime.now():
                future_dates.append(date.strftime("%Y-%m-%d"))

        return future_dates
