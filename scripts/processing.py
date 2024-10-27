import os
from pathlib import Path
import pandas as pd
from typing import Type, Union
from variables import categorical_cols, ordinal_cols
from sklearn.model_selection import train_test_split


# PRIMARY CLASS USED TO LOAD CSV FILES AND PERFORM MINIMAL DATA PROCESSING (E.G. REPLACE NAN VALUES WITH SENTINEL VALUE)


class primary:
    def __init__(
            self,
            data_dir: Path,
            years: list,
            filename_format: str,
            Print: bool,
            sentinel: Union[int, float, str, None] = None,
    ) -> None:

        self.data_dir = data_dir
        self.years = years
        self.filename_format = filename_format
        self.sentinel = sentinel
        self.Print = Print

        restrict_years = []
        for year in self.years:
            filename = self.filename_format.replace("yyyy", str(year))
            filepath = self.data_dir.joinpath(filename)
            if filepath.exists():
                restrict_years.append(year)
        self.years = restrict_years

        # New attributes
        self.dataset = extract_dataset(self.filename_format)
        self.df = load_data(
            self.data_dir,
            self.years,
            self.filename_format,
            self.sentinel,
        )

        if self.dataset == "ncdb":
            print(
                "\n\nNote that, for the 'ncdb' dataset, no NaN values arise from the source csv files:\n"
            )
            print("  'N', 'NN', 'NNNN' mean 'data element is not applicable';")
            print("  'Q', 'QQ' mean 'other';")
            print("  'U', 'UU' mean 'unknown'.")

        if Print:
            synopsis(self.df)

        if not self.df.empty:
            print("\nSource dataframe:".upper(), "self.df")


# FUNCTIONS USED BY PRIMARY CLASS


def extract_dataset(filename_format):
    if "obsolete" in filename_format:
        return "saaq_obsolete"
    if "saaq" in filename_format:
        if "_fr" in filename_format:
            return "saaq_fr"
        else:
            return "saaq"
    elif "ncdb" in filename_format:
        return "ncdb"
    else:
        return None


def load_data(
        data_dir: Path,
        years: list,
        filename_format,
        sentinel: Union[int, float, str, None],
) -> pd.DataFrame:
    # Initialize empty dataframe
    df = pd.DataFrame()

    print_first_line: bool = True
    for year in years:
        filename = filename_format.replace("yyyy", str(year))
        filepath = data_dir.joinpath(filename)
        try:
            df_year = pd.read_csv(filepath, low_memory=False)
            if print_first_line:
                print("\nFor each year we\n".upper())
                i = 1
                print(f"  {i}. Read csv into dataframe.")
                i += 1
                if sentinel is not None:
                    print(
                        f"  {i}. Replace NaN values with {sentinel}. May take a minute."
                    )
                    i += 1
                print(
                    f"  {i}. Replace strings 'x' by x if x is a number. May take a minute."
                )
                i += 1
                print(
                    f"  {i}. Concatenate resulting dataframe with previous years' dataframe.\n"
                )
                print_first_line = False
            print(f"{year}", end=" ")
            if sentinel is not None and df_year.isnull().any().any():
                df_year = df_year.fillna(sentinel)
            if filename_format == "saaq":
                df_year["SPD_LIM"] = df_year["SPD_LIM"].apply(convert_to_number)
            elif filename_format == "saaq_fr":
                df_year["VITESSE_AUTOR"] = df_year["VITESSE_AUTOR"].apply(
                    convert_to_number
                )
            else:
                convert_to_numeric_if_needed(df_year)
            df = pd.concat([df, df_year], ignore_index=True)
        except:  # We don't have a csv file for this year (or something else has gone wrong).
            pass

    if df.empty:
        print("\nNo data found".upper())
        return

    return df


def synopsis(df: pd.DataFrame) -> None:
    print(
        "\n\n" + "=" * 30,
    )
    for col in df.columns:
        print(f"\n{col}:", end=" ")
        value_counts = df[col].value_counts(dropna=False)
        if len(value_counts) > 20:
            value_counts = value_counts[:20]
        for idx, (value, count) in enumerate(value_counts.items()):
            if idx < len(value_counts) - 1:
                print(f"{value} ({count}), ", end="")
            else:
                print(f"{value} ({count})")
    print("\n" + "=" * 30)


def convert_to_numeric_if_needed(df: pd.DataFrame) -> None:
    for column in df.columns:
        if not pd.api.types.is_integer_dtype(
                df[column]
        ) and not pd.api.types.is_float_dtype(df[column]):
            df[column] = df[column].apply(convert_to_number)


def convert_to_number(value):
    try:
        # Try to convert to float
        numeric_value = float(value)

        # Check if the float value is numerically equivalent to an integer
        if numeric_value.is_integer():
            return int(numeric_value)
        else:
            return numeric_value
    except (ValueError, TypeError):
        # If conversion fails, return the original value
        return value


# PROCESS CLASS USED FOR PROCESSING SOURCE DATA


class process:
    def __init__(
            self,
            source: Type[primary],
            restrict_to: Union[dict, None],
            remove_if: Union[dict, None],
            drop_row_if_missing_value_in: Union[list, None],
            targets: list,
            features: list,
            test_size: Union[float, None],
            seed: Union[int, None],
            stratify: Union[bool, None],
            stratify_by: Union[list, None],
    ) -> None:

        self.source = source
        self.restrict_to = restrict_to
        self.remove_if = remove_if
        self.drop_row_if_missing_value_in = drop_row_if_missing_value_in
        self.targets = targets
        self.features = features
        # In case we forgot, "ID" etc. should not be among the features...
        for non_feature in ["ID", "C_CASE"]:
            if non_feature in self.features:
                print(f"Removing {non_feature} from self.features")
                self.features.remove(non_feature)
        # ... and obviously targets should not be among features...
        for target in self.targets:
            if target in self.features:
                print(f"Removing {target} (it's a target) from self.features")
                self.features.remove(target)
        # Continuing...
        self.test_size = test_size
        self.seed = seed
        self.stratify = stratify
        self.stratify_by = stratify_by
        # New attributes
        self.df = self.source.df
        self.sentinel = self.source.sentinel

        # Apply methods below

        # Restrict (if applicable)
        if self.restrict_to is not None or self.remove_if is not None:
            self.filtration()

        # Add new target columns (if applicable)
        if bool(
                set(["MULT_VEH", "VICTIMS", "TNRY_SEV"]).intersection(
                    set(self.targets).union(set(self.features))
                )
        ):
            self.add_columns()

        # Drop certain rows with null values (if applicable)
        # If a target variable column has null values, the corresponding record should be removed
        if self.df[self.targets].isnull().any().any():
            if self.drop_row_if_missing_value_in is None:
                self.drop_row_if_missing_value_in = []
            for target in self.targets:
                if self.df[target].isnull().any():
                    print(f"Adding {target} to self.drop_row_if_missing_value_in")
                    self.drop_row_if_missing_value_in.append(target)
        # ...
        if self.drop_row_if_missing_value_in is not None:
            self.drop_missing()

        # Remove a feature if it cannot be used to predict a target.
        # e.g. 'NUM_VICTIMS' cannot be used to predict 'SEVERITY', as 'NUM_VICTIMS' > 0 iff 'SEVERITY' > 1.
        # Also remove feature 'f' if we are restricting attention to just one value of 'f'.
        self.remove_features()

        # New attributes
        self.categorical_targets = list(
            set(categorical_cols).intersection(set(self.targets))
        )
        self.categorical_features = list(
            set(categorical_cols).intersection(set(self.features))
        )

        self.ordinal_targets = list(set(ordinal_cols).intersection(set(self.targets)))
        self.ordinal_features = list(set(ordinal_cols).intersection(set(self.features)))

        # Partition into train/test sets        
        if self.test_size is not None:
            self.split()

        print(f"\nself.ordinal_features = {self.ordinal_features}")
        print(f"\nself.ordinal_targets = {self.ordinal_targets}")
        print(f"\nself.categorical_features = {self.categorical_features}")
        print(f"\nself.categorical_targets = {self.categorical_targets}")

    def filtration(self) -> None:
        if self.restrict_to is not None:
            self.restrict_to = {
                k: v for k, v in self.restrict_to.items() if k in self.df.columns
            }
            if len(self.restrict_to) > 0:
                print(f"\nRemoving all records unless:")
                for k, v_list in self.restrict_to.items():
                    print(f"  {k} in {v_list}")
                # Generate the query string based on the self.restrict_to dictionary
                query_str = " & ".join(
                    [f"(`{k}` in {v_list})" for k, v_list in self.restrict_to.items()]
                )
                self.df = self.df.query(query_str)
        if self.remove_if is not None:
            self.remove_if = {
                k: v for k, v in self.remove_if.items() if k in self.df.columns
            }
            if len(self.remove_if) > 0:
                print(f"\nRemoving all records if:")
                for k, v_list in self.remove_if.items():
                    print(f"  {k} in {v_list}")
                # Generate the query string based on the self.remove_if dictionary
                query_str = " | ".join(
                    [f"~(`{k}` in {v_list})" for k, v_list in self.remove_if.items()]
                )
                self.df = self.df.query(query_str)

    def drop_missing(self) -> None:
        # New attribute:
        if self.sentinel is None:
            print(
                f"\nDropping rows for which there is a missing value in a column from {self.drop_row_if_missing_value_in}."
            )
            self.df = self.df.dropna(subset=self.drop_row_if_missing_value_in).copy()
        else:
            print(
                f"\nDropping rows for which there is a sentinel value in a column from {self.drop_row_if_missing_value_in}."
            )
            self.df = self.df[
                ~self.df[self.drop_row_if_missing_value_in].isin([self.sentinel]).copy()
            ]

    def add_columns(self) -> None:
        if "MULT_VEH" in self.targets and "NUM_VEH" in self.df.columns:
            # 0 if single-vehicle accident, 1 if more than one vehicle involved
            print("\nInserting 'MULT_VEH' column.")
            self.df = self.df.assign(
                MULT_VEH=self.df["NUM_VEH"].map({1: 0, 2: 1, 9: 1})
            )
            # Put this column next to 'NUM_VEH' column
            num_veh_index = self.df.columns.get_loc("NUM_VEH")
            self.df.insert(num_veh_index + 1, "MULT_VEH", self.df.pop("MULT_VEH"))

        if "MULT_VEH" in self.targets and "C_VEHS" in self.df.columns:
            # 0 if single-vehicle accident, 1 if more than one vehicle involved
            print("\nInserting 'MULT_VEH' column.")
            mapdict = {1: 0}
            mapdict.update({m: 1 for m in range(2, 100)})
            self.df = self.df.assign(MULT_VEH=self.df["C_VEHS"].map(mapdict))
            # Put this column next to 'C_VEHS' column
            c_vehs_index = self.df.columns.get_loc("C_VEHS")
            self.df.insert(c_vehs_index + 1, "MULT_VEH", self.df.pop("MULT_VEH"))

        if "VICTIMS" in self.targets and "NUM_VICTIMS" in self.df.columns:
            # 0 if no person is injured or killed; 1 if at least one person is injured or killed
            print("\nInserting 'VICTIMS' column.")
            self.df = self.df.assign(
                VICTIMS=self.df["NUM_VICTIMS"].map({0: 0, 1: 1, 2: 1, 9: 1})
            )
            # Put this column next to 'NUM_VICTIMS' column
            num_victims_index = self.df.columns.get_loc("NUM_VICTIMS")
            self.df.insert(num_victims_index + 1, "VICTIMS", self.df.pop("VICTIMS"))

        if "VICTIMS" in self.targets and "P_ISEV" in self.df.columns:
            # 0 if no person is injured or killed; 1 if at least one person is injured or killed
            print("\nInserting 'VICTIMS' column.")
            self.df = self.df.assign(
                VICTIMS=self.df["P_ISEV"].map({1: 0, 2: 1, 3: 1, "N": 0})
            )
            # Put this column next to 'P_ISEV' column
            p_isev_index = self.df.columns.get_loc("P_ISEV")
            self.df.insert(p_isev_index + 1, "VICTIMS", self.df.pop("VICTIMS"))

        if "TNRY_SEV" in self.targets and "SEVERITY" in self.df.columns:
            # 0 or 1 if material damage only; 2 if at least one person is injured or killed.
            # In other words, combining the highest two levels of severity (minor with fatal/serious)
            print("\nInserting 'TNRY_SEV' column.")
            self.df = self.df.assign(
                TNRY_SEV=self.df["SEVERITY"].map(
                    {
                        "Material damage below the reporting threshold": 0,
                        "Material damage only": 0,
                        "Minor": 1,
                        "Fatal or serious": 2,
                    }
                )
            )
            # Put this column next to 'NUM_VEH' column
            severity_index = self.df.columns.get_loc("SEVERITY")
            self.df.insert(severity_index + 1, "TNRY_SEV", self.df.pop("TNRY_SEV"))

        if (
                "V_AGE" in self.features
                and "C_YEAR" in self.df.columns
                and "V_YEAR" in self.df.columns
        ):
            # Age of vehicle in years: C_YEAR - V_YEAR
            print("\nInserting 'V_AGE' column.")
            self.df = self.df.assign(V_AGE=self.df["C_YEAR"] - self.df["V_YEAR"])
            # Put this column next to 'V_YEAR' column
            v_year_index = self.df.columns.get_loc("V_YEAR")
            self.df.insert(v_year_index + 1, "V_AGE", self.df.pop("V_AGE"))

    def remove_features(self) -> None:
        potential_features = set(self.features).intersection(set(self.df.columns))
        for col in potential_features:
            if self.sentinel is None:
                non_null_values = self.df[col].nunique(dropna=True)
                text_to_print = f"\nRemoving {col} from self.features (but not from self.df) as the number of distinct non-null values in self.df['{col}'] is {non_null_values}."
            else:
                non_null_values = self.df[self.df[col] != self.sentinel][col].nunique()
                text_to_print = f"\nRemoving {col} from self.features (but not from self.df) as the number of distinct non-sentinel values in self.df['{col}'] is {non_null_values}."
            if (col == "RDWX" and non_null_values == 0) or (
                    col != "RDWX" and non_null_values < 2
            ):
                print(text_to_print)
                self.features.remove(col)

        # May not be exhaustive... always think carefully about features and targets for any given model fitting
        incompatible_target_feature = {
            "SEVERITY": ["NUM_VICTIMS", "TNRY_SEV", "VICTIMS"],
            "NUM_VICTIMS": ["SEVERITY", "TNRY_SEV", "VICTIMS"],
            "TNRY_SEV": ["SEVERITY", "NUM_VICTIMS", "VICTIMS"],
            "VICTIMS": ["SEVERITY", "TNRY_SEV", "NUM_VICTIMS", "P_ISEV", "C_SEV"],
            "ACCDN_TYPE": [
                "PED",
                "BICYC",
                "MTRCYC",
                "NUM_VEH",
                "HVY_VEH",
                "LT_TRK",
            ],
            "PED": ["ACCDN_TYPE"],
            "BICYC": ["ACCDN_TYPE"],
            "P_ISEV": ["VICTIMS", "C_SEV"],
            "C_SEV": ["P_ISEV", "VICTIMS"],
            "NUM_VEH": ["MULT_VEH"],
            "MULT_VEH": ["NUM_VEH", "C_VEHS"],
            "C_VEHS": ["MULT_VEH"],
        }

        for target, features in incompatible_target_feature.items():
            if target in self.targets:
                for feature in features:
                    if feature in self.features:
                        print(
                            f"\nRemoving {feature} from self.features (but not from self.df): can't use {feature} to predict {target}."
                        )
                        self.features.remove(feature)

        # Some features may be redundant/lead to multicolinearity issues etc.
        # If there are lots, will need to do something more sophisticated here.
        if "V_AGE" in self.features and "V_YEAR" in self.features:
            print(
                f"\nRemoving 'V_YEAR' from self.features (but not from self.df) as 'V_AGE' is in self.features."
            )
            self.features.remove("V_YEAR")

    def split(self) -> None:
        print(
            "\nPartitioning data into training/test sets: self.df_train/self.df_test."
        )
        if not self.stratify:
            # New attributes:
            self.df_train, self.df_test = train_test_split(
                self.df.copy(deep=True),
                test_size=self.test_size,
                random_state=self.seed,
                shuffle=True,
            )

        if self.stratify:
            if self.stratify_by is None:
                print(
                    f"\nstratify is {self.stratify} but stratify_by is {self.stratify_by}."
                )
                print(
                    "\nSwitching stratify to False and proceeding with non-stratified train/test split."
                )
                self.stratify = False
                self.split()

            else:
                # Maintain relative proportions of records from each value in "STRAT_TUPLE".
                STRAT_TUPLE = self.df[self.stratify_by].apply(tuple, axis=1)
                # New attributes:
                self.df_train, self.df_test = train_test_split(
                    self.df.copy(deep=True),
                    test_size=self.test_size,
                    random_state=self.seed,
                    stratify=STRAT_TUPLE,
                    shuffle=True,
                )
