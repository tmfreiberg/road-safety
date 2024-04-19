from saaq_data_dictionary import FR_EN

# SAAQ AND NCDB COLUMN NAME-VARIABLE TYPE DICTIONARIES

saaq_variable_types = {
    "YEAR": "categorical",
    "ID": None,
    "MONTH": "categorical",
    "HOUR": "categorical",
    "WKDY_WKND": "categorical",
    "SEVERITY": "ordinal",
    "NUM_VICTIMS": "ordinal",
    "NUM_VEH": "ordinal",
    "REGION": "categorical",
    "SPD_LIM": "ordinal",
    "ACCDN_TYPE": "categorical",
    "RD_COND": "categorical",
    "LIGHT": "ordinal",
    "ZONE": "categorical",
    "PUB_PRIV_RD": "categorical",
    "ASPECT": "categorical",
    "LONG_LOC": "categorical",
    "RD_CONFG": "categorical",
    "RDWX": "categorical",
    "WEATHER": "categorical",
    "LT_TRK": "categorical",
    "HVY_VEH": "categorical",
    "MTRCYC": "categorical",
    "BICYC": "categorical",
    "PED": "categorical",
    "MULT_VEH": "ordinal",  # Not in original dataframe but can be added as a target
    "VICTIMS": "ordinal",  # Not in original dataframe but can be added as a target
    "TNRY_SEV": "ordinal",  # Not in original dataframe but can be added as a target
}

ncdb_variable_types = {
    "C_YEAR": "categorical",
    "C_MNTH": "categorical",
    "C_WDAY": "categorical",
    "C_HOUR": "categorical",
    "C_SEV": "ordinal",
    "C_VEHS": "ordinal",
    "C_CONF": "categorical",
    "C_RCFG": "categorical",
    "C_WTHR": "categorical",
    "C_RSUR": "categorical",
    "C_RALN": "categorical",
    "C_TRAF": "categorical",
    "V_ID": "categorical",
    "V_TYPE": "categorical",
    "V_YEAR": "ordinal",
    "P_ID": "categorical",
    "P_SEX": "categorical",
    "P_AGE": "ordinal",
    "P_PSN": "categorical",
    "P_ISEV": "ordinal",
    "P_SAFE": "categorical",
    "P_USER": "categorical",
    "C_CASE": None,
    "V_AGE": "ordinal",  # Not in original dataframe but can be added as a feature
}
# NB: 'MULT_VEH' and 'VICTIMS' can be added as targets to ncdb as well, but for present purposes we don't want overlap between the above two dictionaries.

# Combine all categorical variables into a single list

saaq_categorical_variables = [
    k for k, v in saaq_variable_types.items() if v == "categorical"
]
ncdb_categorical_variables = [
    k for k, v in ncdb_variable_types.items() if v == "categorical"
]
categorical_cols = list(set(saaq_categorical_variables + ncdb_categorical_variables))

# Combine all ordinal variables into a single list

saaq_ordinal_variables = [k for k, v in saaq_variable_types.items() if v == "ordinal"]
ncdb_ordinal_variables = [k for k, v in ncdb_variable_types.items() if v == "ordinal"]
ordinal_cols = list(set(saaq_ordinal_variables + ncdb_ordinal_variables))


# Define mappings for each potential variable for input into models.
# Numerical values, of a similar order of magnitude, starting with 0 and increasing from there.
class_codes = dict.fromkeys(saaq_variable_types.keys())
# class_codes["ID"] =
# class_codes["YEAR"] =
class_codes["REGION"] = {
    "Montréal (06)": 6,
    "Laval (13)": 13,
    "Côte-Nord (09)": 9,
    "Nord-du-Québec (10)": 10,
    "Saguenay/-Lac-Saint-Jean (02)": 2,
    "Outaouais (07)": 7,
    "Laurentides (15)": 15,
    "Montérégie (16)": 16,
    "Capitale-Nationale (03)": 3,
    "Estrie (05)": 5,
    "Mauricie (04)": 4,
    "Gaspésie/-Îles-de-la-Madeleine (11)": 11,
    "Bas-Saint-Laurent (01)": 1,
    "Centre-du-Québec (17)": 17,
    "Chaudière-Appalaches (12)": 12,
    "Lanaudière (14)": 14,
    "Abitibi-Témiscamingue (08)": 8,
}
class_codes["ASPECT"] = {"Straight": 0, "Curve": 1}
class_codes["PUB_PRIV_RD"] = {1: 0, 2: 1}
class_codes["WEATHER"] = {
    11: 0,
    12: 1,
    13: 2,
    14: 3,
    15: 4,
    16: 5,
    17: 6,
    18: 7,
    19: 8,
    99: 9,
}
class_codes["RD_CONFG"] = {1: 0, 23: 1, 45: 2, 9: 3}
class_codes["LIGHT"] = {1: 3, 2: 2, 3: 1, 4: 0}
class_codes["ZONE"] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 9: 6}
class_codes["RD_COND"] = {
    11: 0,
    12: 1,
    13: 2,
    14: 3,
    15: 4,
    16: 5,
    17: 6,
    18: 7,
    19: 8,
    20: 9,
    99: 10,
}
class_codes["ACCDN_TYPE"] = {
    "vehicle": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "animal": 3,
    "fixed object": 4,
    "no collision": 5,
    "other": 6,
}
class_codes["LONG_LOC"] = {12: 0, 33: 1, 34: 2, 40: 3, 69: 4, 99: 5}
class_codes["RDWX"] = {"N": 0, "Y": 1}
class_codes["SEVERITY"] = {
    "Material damage below the reporting threshold": 0,
    "Material damage only": 1,
    "Minor": 2,
    "Fatal or serious": 3,
}
class_codes["TNRY_SEV"] = {0: 0, 1: 1, 2: 2}  # Potential new variable
class_codes["HOUR"] = {
    "00:00:00-03:59:00": 0,
    "04:00:00-07:59:00": 1,
    "08:00:00-11:59:00": 2,
    "12:00:00-15:59:00": 3,
    "16:00:00-19:59:00": 4,
    "20:00:00-23:59:00": 5,
}
class_codes["LT_TRK"] = {"N": 0, "Y": 1}
class_codes["MTRCYC"] = {"N": 0, "Y": 1}
class_codes["PED"] = {"N": 0, "Y": 1}
class_codes["HVY_VEH"] = {"N": 0, "Y": 1}
class_codes["BICYC"] = {"N": 0, "Y": 1}
class_codes["WKDY_WKND"] = {"WKDY": 0, "WKND": 1}
class_codes["MONTH"] = {m: m for m in range(1, 13)}
class_codes["NUM_VEH"] = {1: 0, 2: 1, 9: 2}
class_codes["MULT_VEH"] = {0: 0, 1: 1}  # Potential new variable
class_codes["NUM_VICTIMS"] = {0: 0, 1: 1, 2: 2, 9: 3}
class_codes["VICTIMS"] = {0: 0, 1: 1}  # Potential new variable
class_codes["SPD_LIM"] = {"<50": 0, 50: 5, 60: 6, 70: 7, 80: 8, 90: 9, 100: 10}

# And now for the ncdb data...
class_codes.update(dict.fromkeys(ncdb_variable_types.keys()))

# class_codes['C_YEAR'] = { }
# class_codes['C_MNTH'] = { } # already integers ranging from 1 to 12
class_codes["C_WDAY"] = {}  # already integers ranging from 1 to 7
class_codes["C_HOUR"] = {}  # already integers ranging from 0 to 23

class_codes["C_SEV"] = {
    1: 1,
    2: 0,
}  # Originally, 1 means at least one fatal injury, 2 means non-fatal injury. Now, 1 means fatal, 0 non-fatal.

class_codes["C_VEHS"] = {1: 0, 2: 1, 3: 2}
class_codes["C_VEHS"].update({m: 3 for m in range(4, 100)})

class_codes["C_CONF"] = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    21: 4,
    22: 4,
    23: 5,
    24: 5,
    25: 5,
    31: 6,
    32: 7,
    33: 8,
    34: 9,
    35: 10,
    36: 9,
    41: 11,
    "QQ": 11,
}

class_codes["C_RCFG"] = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 5,
    7: 5,
    8: 5,
    9: 5,
    10: 6,
    11: 6,
    12: 6,
    "QQ": 5,
}

class_codes["C_WTHR"] = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, "Q": 8}

class_codes["C_RSUR"] = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 6,
    8: 6,
    9: 6,
    "Q": 6,
}

class_codes["C_RALN"] = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    "Q": 3,
}

class_codes["C_TRAF"] = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 6,
    9: 7,
    10: 8,
    11: 8,
    12: 8,
    13: 9,
    14: 9,
    15: 10,
    16: 10,
    17: 11,
    18: 11,
    "QQ": 11,
}

class_codes["V_ID"] = {1: 1, 2: 2, 3: 3, 99: 4}
class_codes["V_ID"].update({m: 5 for m in range(4, 99)})

class_codes["V_TYPE"] = {
    1: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 6,
    11: 7,
    14: 8,
    16: 9,
    17: 10,
    18: 9,
    19: 9,
    20: 9,
    21: 9,
    22: 9,
    23: 9,
    "QQ": 9,
}

# class_codes['V_YEAR'] = { } # We'll just do 'V_AGE'.
class_codes["V_AGE"] = {m: m // 3 for m in range(31)}
class_codes["V_AGE"].update({m: 11 for m in range(31, 105)})

class_codes["P_ID"] = {1: 1, 2: 2, 3: 3}
class_codes["P_ID"].update({m: 4 for m in range(4, 100)})

class_codes["P_SEX"] = {"M": 0, "F": 1}

class_codes["P_AGE"] = {m: m // 10 for m in range(100)}

class_codes["P_PSN"] = {
    11: 1,
    12: 2,
    13: 2,
    21: 3,
    22: 3,
    23: 3,
    31: 3,
    32: 3,
    33: 3,
    96: 4,
    97: 5,
    98: 4,
    99: 0,
    "QQ": 4,
}

class_codes["P_ISEV"] = {1: 0, 2: 1, 3: 2}

class_codes["P_SAFE"] = {
    1: 1,
    2: 2,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 7,
    "QQ": 7,
}

# class_codes['P_USER'] = { } # already integers ranging from 1 to 5

# class_codes['C_CASE'] = { }

inverse_class_codes = {
    variable: {value: key for key, value in class_codes[variable].items()}
    for variable in class_codes.keys()
    if class_codes[variable] is not None
}
