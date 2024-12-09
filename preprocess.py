import pandas as pd
import numpy as np

class DataAugmentation:
    """
    Load, process, and augment database for real estate analysis.

    Attributes:
        base_path (str): Path to the folder containing the datasets used to create new features.
    """

    def __init__(self, base_path="data"):
        """
        Initialize DataAugmentation using path to the data directory.

        :param base_path: Path to the folder containing the datasets needed to create new features.
        """
        self.base_path = base_path

    def load_csv(self, filename: str, sep=",", **kwargs):
        """
        Load a CSV file.

        :param filename: Name of CSV file.
        :param sep: Separator used in CSV file.
        :param kwargs: Additional arguments for `pd.read_csv`.
        :return: DataFrame loaded from CSV file, or None if the file is not found.
        """
        path = f"{self.base_path}/{filename}"
        try:
            return pd.read_csv(path, sep=sep, **kwargs)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None

    def load_excel(self, filename, **kwargs):
        """
        Load an Excel file.

        :param filename: Name of Excel file.
        :param kwargs: Additional arguments for `pd.read_excel`.
        :return: DataFrame loaded from Excel file, or None if the file is not found.
        """
        path = f"{self.base_path}/{filename}"
        try:
            return pd.read_excel(path, **kwargs)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None

    def load_data(self):
        """
        Load all datasets required for feature creation.
        """
        self.postal_codes_belgium = self.load_csv("postal-codes-belgium.csv", sep=";")
        self.codes_nis = self.load_csv("codes-ins-nis-postaux-belgique.csv", sep=";")
        self.criminal_figures = self.load_csv(
            "criminal_figures_statistics_table.csv", sep=";", skiprows=0, header=[1]
        )
        self.fiscal_statistics = self.load_excel(
            "fisc2022_C_NL.xls", sheet_name="Totaal", skiprows=5, header=[1]
        )
        self.pop_density = self.load_excel(
            "Pop_density_en.xlsx", sheet_name="2024", skiprows=0, header=[1]
        )
        self.sales_statistics = self.load_excel(
            "FR_immo_statbel_trimestre_par_commune.xlsx",
            sheet_name="Par commune",
            skiprows=1,
            header=[1],
        )
        self.immoweb_database = self.load_csv("immoweb_database.csv", sep=",")

    def process_data(self):
        """
        Process the datasets and create the final real estate dataset with relevant features.
        """
        postal_codes = self.postal_codes_belgium.assign(
            postal_code=self.postal_codes_belgium["Postal Code"],
            municipality=self.postal_codes_belgium[
                "Municipality name (French)"
            ].combine_first(self.postal_codes_belgium["Municipality name (Dutch)"]),
        )[["postal_code", "municipality"]]

        nis = self.codes_nis[["NIS-code Municipaity", "Postal code"]].rename(
            columns={"NIS-code Municipaity": "nis", "Postal code": "postal_code"}
        )

        prosperity_index = self.fiscal_statistics[["Unnamed: 1", "Unnamed: 18"]].rename(
            columns={"Unnamed: 1": "nis", "Unnamed: 18": "prosperity_index"}
        )

        population = self.pop_density[
            ["Refnis", "Population", "Population / km²"]
        ].rename(
            columns={
                "Refnis": "nis",
                "Population": "total_population",
                "Population / km²": "Population_km",
            }
        )
        criminal_statistics = self.criminal_figures[[" ", "2023"]].rename(
            columns={" ": "municipality", "2023": "crimes"}
        )
        sales_statistics = self.sales_statistics[
            ["refnis", "année", "période", "prix médian(€)", "prix médian(€).3"]
        ].rename(
            columns={
                "refnis": "nis",
                "année": "year",
                "période": "period",
                "prix médian(€)": "median_price_houses",
                "prix médian(€).3": "median_price_apart",
            }
        )

        # Create 'property_median_price' columns in sales_statistics
        sales_statistics = sales_statistics[sales_statistics["year"].isin([2023, 2024])]

        sales_statistics = (
            sales_statistics.groupby("nis")
            .agg({"median_price_houses": "median", "median_price_apart": "median"})
            .reset_index()
        )

        sales_statistics["median_price_per_municipality"] = sales_statistics[
            ["median_price_houses", "median_price_apart"]
        ].median(axis=1)

        # Converto postal_code to numerical before merging
        self.immoweb_database["postal_code"] = pd.to_numeric(
            self.immoweb_database["postal_code"], errors="coerce"
        ).astype("Int64")
        postal_codes["postal_code"] = pd.to_numeric(
            postal_codes["postal_code"], errors="coerce"
        ).astype("Int64")

        # Merge databases
        postal_codes = postal_codes.merge(nis, on="postal_code", how="inner")
        postal_codes = postal_codes.merge(
            sales_statistics, on=["nis"], how="inner", copy=True
        )
        postal_codes = postal_codes.merge(prosperity_index, on="nis", how="inner")
        postal_codes = postal_codes.merge(population, on="nis", how="inner")
        postal_codes = postal_codes.merge(
            criminal_statistics, on="municipality", how="inner"
        )
        data = self.immoweb_database.merge(postal_codes, on="postal_code", how="inner")

        # Create column 'crime_rate'
        data["crimes"] = pd.to_numeric(data["crimes"], errors="coerce")
        data["total_population"] = pd.to_numeric(
            data["total_population"], errors="coerce"
        )
        data["crime_rate"] = (data["crimes"] / data["total_population"]) * 1000

        # Converto all columns and string to lowe case
        data.columns = data.columns.str.lower()
        data = data.map(lambda x: x.lower() if isinstance(x, str) else x)

        # Drop columns
        columns_to_drop = [
            "nis",
            "total_population",
            "crimes",
            "median_price_houses",
            "median_price_apart",
        ]
        data = data.drop(columns=columns_to_drop)
        data.drop_duplicates(inplace=True)

        self.data = data

    def add_new_features(self):
        """
        Execute all functions to add new features and print database information.
        :return: The database with new features.
        :print: size, columns and preview of database.
        """
        self.load_data()
        self.process_data()
        print(f"Dataframe size (rows,columns): {self.data.shape}")
        print(f"Dataframe columns: {self.data.columns}")
        print(f"Dataframe preview: \n {self.data.head()}")
        return self.data


class DataImputation:
    """
    Handle missing values with imputation for real estate dataset.

    Attributes:
        Dataset: The dataset containing real estate information whit missing values.
    """

    def __init__(self, data):
        """
        Initialize DataImputation with a DataFrame.

        :param data: DataFrame to perform imputation for missing values.
        """
        self.data = data

    def impute_boolean_columns(self):
        """
        Convert 'open_fire' and 'type_of_property' to boolean columns.
        Fill missing values in boolean columns with 0.
        """
        boolean_columns = [
            "furnished",
            "open_fire",
            "terrace",
            "garden",
            "swimming_pool",
            "disabled_access",
            "lift",
            "type_of_property",
        ]

        self.data["open_fire"] = self.data["open_fire"].fillna(0)
        self.data["open_fire"] = self.data["open_fire"].apply(
            lambda x: 1 if x != 0 else 0
        )
        self.data["type_of_property"] = self.data["type_of_property"].map(
            {"house": 1, "apartment": 0}
        )
        self.data[boolean_columns] = self.data[boolean_columns].fillna(0)

    def impute_garden_terrace_area(self):
        """
        Fill missing values in garden_area and terrace_area columns with 0.
        """
        self.data["garden_area"] = self.data["garden_area"].fillna(0)
        self.data["terrace_area"] = self.data["terrace_area"].fillna(0)

    def impute_building_and_number_facades(self):
        """
        Impute missing values in the 'state_of_the_building' and 'number_of_facades' columns using the mode of each
        group (type_of_property, municipality) and the global mode for remaining missing values.
        """
        col_to_impute = ["state_of_the_building", "number_of_facades"]

        for col in col_to_impute:
            mode_values = self.data.groupby(["type_of_property", "municipality"])[
                col
            ].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

            def impute_with_group_mode(row):
                if pd.isna(row[col]):
                    return mode_values.get(
                        (row["type_of_property"], row["municipality"])
                    )
                return row[col]

            self.data[col] = self.data.apply(impute_with_group_mode, axis=1)
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

    def impute_living_area(self):
        """
        Impute missing values in the 'living_area' column using the median of each group
        (type_of_property, municipality) and the global median for remaining missing values.
        """
        median_values = self.data.groupby(["type_of_property", "municipality"])[
            "living_area"
        ].median()

        def impute_with_group_median(row):
            if pd.isna(row["living_area"]):
                return median_values.loc[(row["type_of_property"], row["municipality"])]
            return row["living_area"]

        self.data["living_area"] = self.data.apply(impute_with_group_median, axis=1)
        self.data["living_area"] = self.data["living_area"].fillna(
            self.data["living_area"].median()
        )

    def drop_median_per_municipality(self):
        """
        Drop observations with missing values in "median_price_per_municipality" column.
        """
        self.data = self.data.dropna(subset=["median_price_per_municipality"])

    def drop_columns(self):
        """
        Drop unnecessary columns.
        """
        self.data = self.data.drop(
            [
                "surface_area_plot_of_land",
                "surface_of_the_land",
                "subtype_of_property",
                "garden_area",
                "terrace_area",
                "municipality",
            ],
            axis=1,
        )

    def impute_missing_values(self):
        """
        Execute all the imputations for missing values in the database and print summary.
        :return: The database without missing values.
        :print: size of database and count of missing values per column.
        """
        self.impute_boolean_columns()
        self.impute_garden_terrace_area()
        self.impute_building_and_number_facades()
        self.impute_living_area()
        self.drop_median_per_municipality()
        self.drop_columns()

        summary = []
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_percentage = np.round(self.data[col].isnull().mean() * 100, 3)
            unique = self.data[col].nunique()
            summary.append([missing_count, f"{missing_percentage}%", unique])
        summary_df = pd.DataFrame(
            summary,
            index=self.data.columns,
            columns=["Missing Count", "Missing Percentage", "Unique Values"],
        )
        print(f"Database summary after imputing missing values: \n {summary_df}")
        print(f"Dataframe size (rows,columns): {self.data.shape}")

        return self.data


class FeatureCombination:
    """
    Create new columns by permorming feature combination.

    Attributes:
        Dataset: The dataset containing real estate information withoutmissing values.
    """

    def __init__(self, data):
        """
        Initialize FeatureCombination with a DataFrame.

        :param data: DataFrame to perform feature combination on.
        """
        self.data = data

    def combine_exterior_space(self):
        """
        Create 'exterior_space' column by combining 'garden' and 'terrace'.
        """
        self.data["garden"] = self.data["garden"].astype(int)
        self.data["terrace"] = self.data["terrace"].astype(int)
        self.data["exterior_space"] = (
            self.data["garden"] | self.data["terrace"]
        ).astype(int)
        self.data.drop(["garden", "terrace"], axis=1, inplace=True)

    def combine_accessibility(self):
        """
        Create 'accessible' column by combining 'disabled_access' and 'lift'.
        """
        self.data["disabled_access"] = self.data["disabled_access"].astype(int)
        self.data["lift"] = self.data["lift"].astype(int)
        self.data["accessible"] = (
            self.data["disabled_access"] | self.data["lift"]
        ).astype(int)
        self.data.drop(["disabled_access", "lift"], axis=1, inplace=True)

    def combine_state_categories(self):
        """
        Create broader categories for 'state_of_the_building' by combining how values behave with respect to price.
        Combine 'fully_equipped_kitchen' and 'state_of_the_building' as 'extra_investment', the higher values the less work required.
        """
        state_mapping = {
            "just_renovated": 3,
            "as_new": 4,
            "good": 3,
            "to_renovate": 2,
            "to_be_done_up": 2,
            "to_restore": 1,
        }
        self.data["state_of_the_building"] = self.data["state_of_the_building"].map(
            state_mapping
        )

        self.data["extra_investment"] = (
            self.data["state_of_the_building"] + self.data["fully_equipped_kitchen"]
        )
        self.data.drop(
            ["fully_equipped_kitchen", "state_of_the_building"], axis=1, inplace=True
        )

    def combine_features(self):
        """
        Apply all feature combination methods.

        :return: The database with features combined.
        :print: Dataframe columns after feature combination.
        """
        self.combine_exterior_space()
        self.combine_accessibility()
        self.combine_state_categories()
        print(f"Dataframe columns after feature combination: {self.data.columns}")
        return self.data


class RemoveOutliers:
    """
    A class to detect and remove outliers from specified columns using the IQR (Interquartile Range) method.

    Attributes:
        Dataset: The dataset containing real estate information with outliers.
    """

    def __init__(self, data):
        """
        Initializes RemoveOutliers class with a DataFrame.

        :param data: Dataset to perform imputation for missing values.
        """
        self.data = data

    def detect_outliers_iqr(self, column):
        """
        Detects outliers using the IQR (Interquartile Range) method.

        :param column: The numerical column from which to detect outliers.
        :return: A version of the original column with outliers removed.
        """
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        IQR = q3 - q1
        lwr_bound = q1 - (1.5 * IQR)
        upr_bound = q3 + (1 * IQR)
        column_cleaned = column[(column >= lwr_bound) & (column <= upr_bound)]

        return column_cleaned

    def remove_outliers(self):
        """
        Removes outliers from the specified columns.

        :return: The DataFrame with outliers removed.
        :print: Summary of dataset without outliers.
        """
        # Remove outliers from 'price', 'living_area' and number_of_facades'
        columns_to_clean = ["price", "living_area", "number_of_facades"]
        for column in columns_to_clean:
            self.data.loc[:, column] = self.detect_outliers_iqr(self.data[column])
            self.data = self.data[self.data[column].notna()]

        # Remove outliers from "number_of_rooms"
        # Replace 0 number_of_rooms in houses by mode by group
        mode_values = self.data.groupby(["type_of_property", "postal_code"])[
            "number_of_rooms"
        ].agg(lambda x: x.mode()[0])

        def replace_with_mode(row):
            if row["number_of_rooms"] == 0 and row["type_of_property"] == 1:
                return mode_values.loc[(row["type_of_property"], row["postal_code"])]
            return row["number_of_rooms"]

        self.data["number_of_rooms"] = self.data.apply(replace_with_mode, axis=1)

        # Remove outliers from 'number_of_rooms': filter records greater than or equal to 7
        self.data = self.data[self.data["number_of_rooms"] < 7]

        print(
            f"Dataframe size(rows,columns) after removing outliers: {self.data.shape}"
        )

        return self.data


class FilterAndDrop:
    """
    Filtering observations and dropping unnecessary columns.

    Attributes:
        data: The real estate dataset to be processed.
    """

    def __init__(self, data):
        """
        Initializes the FilterAndDrop with a dataset.
        """
        self.data = data

    def filter_observations(self):
        """
        Filters the dataset by postal code. Only postal codes that appear
        more than 30 times in the dataset will be retained.

        :return: A filtered DataFrame with valid postal codes.
        """
        postal_code_counts = self.data["postal_code"].value_counts()
        valid_postal_codes = postal_code_counts[postal_code_counts > 30].index
        self.data = self.data[self.data["postal_code"].isin(valid_postal_codes)]

        return self.data

    def drop_columns(self):
        """
        Drops unnecessary columns from the dataset.

        :return: The DataFrame after dropping the specified columns.
        """
        columns_to_drop = [
            "open_fire",
            "swimming_pool",
            "type_of_property",
            "furnished",
            "postal_code",
            "population_km",
            "crime_rate",
            "number_of_rooms",
            "accessible",
            "number_of_facades",
            "exterior_space"
        ]
        self.data = self.data.drop(columns=columns_to_drop)
        return self.data

    def filter_drop(self):
        """
        Apply the filter and drop function.

        :return: A filtered dataset with specified columns dropped.
        :print: information about the dataset after processing.
        """
        self.filter_observations()
        self.drop_columns()
        self.data = self.data.drop_duplicates()

        categorical_features = self.data.select_dtypes(include=["object"]).columns
        numerical_features = self.data.select_dtypes(exclude=["object"]).columns

        print(
            f"Dataframe size (rows, columns) after filter observation and drop columns: {self.data.shape}"
        )
        print(f"Dataframe columns: {self.data.columns}")
        print("Numerical features: " + str(len(numerical_features)))
        print("Categorical features: " + str(len(categorical_features)))

        return self.data
