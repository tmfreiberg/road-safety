from typing import Union

def explain(language: Union[str, None] = None, terms: Union[list, None] = None) -> None:
    global FR_EN
    if language is None:
        language = "EN"

    if language.lower()[0] == "f":
        if terms is None:
            terms = FR_EN.keys()
        explainable_terms = [term for term in FR_EN.keys() if term in terms]
        explain = explain_fr
    else:
        if terms is None:
            terms = FR_EN.values()
        explainable_terms = [term for term in FR_EN.values() if term in terms]
        explain = explain_en

    for idx, term in enumerate(explainable_terms):
        print(term, "\n")

        explanation = explain[term]
        
        if bool(explanation["summary"]):
            summary_txt = f"{explanation['summary']}"
            print(summary_txt, "\n")
        if bool(explanation["values"]) and bool(explanation["elaborate"]):
            for value, elaborate in zip(
                explanation["values"], explanation["elaborate"]
            ):
                print(value, ":", elaborate, "\n")
        elif bool(explanation["values"]):
            for value in explanation["values"]:
                print(value, "\n")
        elif bool(explanation["elaborate"]):
            for elaborate in explanation["elaborate"]:
                print(elaborate, "\n")

selection_dictionary = {'MONTH': {1: 1,
  2: 2,
  3: 3,
  4: 4,
  5: 5,
  6: 6,
  7: 7,
  8: 8,
  9: 9,
  10: 10,
  11: 11,
  12: 12},
 'HOUR': {'00:00:00-03:59:00': 0,
  '04:00:00-07:59:00': 1,
  '08:00:00-11:59:00': 2,
  '12:00:00-15:59:00': 3,
  '16:00:00-19:59:00': 4,
  '20:00:00-23:59:00': 5},
 'WKDY_WKND': {'WKDY': 0, 'WKND': 1},
 'NUM_VEH': {1: 0, 2: 1, 9: 2},
 'SPD_LIM': {'<50': 0, 50: 5, 60: 6, 70: 7, 80: 8, 90: 9, 100: 10},
 'ACCDN_TYPE': {'vehicle': 0,
  'pedestrian': 1,
  'cyclist': 2,
  'animal': 3,
  'fixed object': 4,
  'no collision': 5,
  'other': 6},
 'RD_COND': {11: 0,
  12: 1,
  13: 2,
  14: 3,
  15: 4,
  16: 5,
  17: 6,
  18: 7,
  19: 8,
  20: 9,
  99: 10},
 'LIGHT': {1: 3, 2: 2, 3: 1, 4: 0},
 'ZONE': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 9: 6},
 'PUB_PRIV_RD': {1: 0, 2: 1},
 'ASPECT': {'Straight': 0, 'Curve': 1},
 'LONG_LOC': {12: 0, 33: 1, 34: 2, 40: 3, 69: 4, 99: 5},
 'RD_CONFG': {1: 0, 23: 1, 45: 2, 9: 3},
 'RDWX': {'N': 0, 'Y': 1},
 'WEATHER': {11: 0,
  12: 1,
  13: 2,
  14: 3,
  15: 4,
  16: 5,
  17: 6,
  18: 7,
  19: 8,
  99: 9},
 'LT_TRK': {'N': 0, 'Y': 1},
 'HVY_VEH': {'N': 0, 'Y': 1},
 'MTRCYC': {'N': 0, 'Y': 1},
 'BICYC': {'N': 0, 'Y': 1},
 'PED': {'N': 0, 'Y': 1}}

# FRENCH TO ENGLISH DICTIONARY: SAAQ SOURCE DATAFRAME COLUMN NAMES

FR_EN = {
    "AN": "YEAR",
    "CD_ASPCT_ROUTE": "ASPECT",
    "CD_CATEG_ROUTE": "PUB_PRIV_RD",
    "CD_COND_METEO": "WEATHER",
    "CD_CONFG_ROUTE": "RD_CONFG",
    "CD_ECLRM": "LIGHT",
    "CD_ENVRN_ACCDN": "ZONE",
    "CD_ETAT_SURFC": "RD_COND",
    "CD_GENRE_ACCDN": "ACCDN_TYPE",
    "CD_LOCLN_ACCDN": "LONG_LOC",
    "CD_ZON_TRAVX_ROUTR": "RDWX",
    "GRAVITE": "SEVERITY",
    "HR_ACCDN": "HOUR",
    "IND_AUTO_CAMION_LEGER": "LT_TRK",
    "IND_MOTO_CYCLO": "MTRCYC",
    "IND_PIETON": "PED",
    "IND_VEH_LOURD": "HVY_VEH",
    "IND_VELO": "BICYC",
    "JR_SEMN_ACCDN": "WKDY_WKND",
    "MS_ACCDN": "MONTH",
    "NB_VEH_IMPLIQUES_ACCDN": "NUM_VEH",
    "NB_VICTIMES_TOTAL": "NUM_VICTIMS",
    "NO_SEQ_COLL": "ID",
    "REG_ADM": "REGION",
    "VITESSE_AUTOR": "SPD_LIM",
}

# And the English-French dictionary

EN_FR = {en: fr for (fr, en) in FR_EN.items()}

# For translating the French column values to English:

FR_EN_map_default = {
    "CD_ASPCT_ROUTE": {"Droit": "Straight", "Courbe": "Curve"},
    "CD_GENRE_ACCDN": {
        "véhicule": "vehicle",
        "piéton": "pedestrian",
        "cycliste": "cyclist",
        "objet fixe": "fixed object",
        "sans collision": "no collision",
        "autre": "other",
    },
    "CD_ZON_TRAVX_ROUTR": {"O": "Y"},
    "GRAVITE": {
        "Dommages matériels seulement": "Material damage only",
        "Dommages matériels inférieurs au seuil de rapportage": "Material damage below the reporting threshold",
        "Léger": "Minor",
        "Mortel ou grave": "Fatal or serious",
    },
    "IND_AUTO_CAMION_LEGER": {"O": "Y", "N": "N"},
    "IND_MOTO_CYCLO": {"O": "Y", "N": "N"},
    "IND_PIETON": {"O": "Y", "N": "N"},
    "IND_VEH_LOURD": {"O": "Y", "N": "N"},
    "IND_VELO": {"O": "Y", "N": "N"},
    "JR_SEMN_ACCDN": {"SEM": "WKDY", "FDS": "WKND"},
}

# And the English-French translation:

EN_FR_map_default = {
    FR_EN[key]: {en: fr for (fr, en) in FR_EN_map_default[key].items()}
    for key in FR_EN_map_default.keys()
}

# For each column name in the French SAAQ source dataframe, we put all possible values in a dictionary (French keys, English values):

values = {
    "AN": {},
    "CD_ASPCT_ROUTE": {"Droit": "Straight", "Courbe": "Curve"},
    "CD_CATEG_ROUTE": {1: 1, 2: 2, "": ""},
    "CD_COND_METEO": {
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 18,
        19: 19,
        99: 99,
        "": "",
    },
    "CD_CONFG_ROUTE": {1: 1, 23: 23, 45: 45, 9: 9, "": ""},
    "CD_ECLRM": {1: 1, 2: 2, 3: 3, 4: 4, "": ""},
    "CD_ENVRN_ACCDN": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 9: 9, "": ""},
    "CD_ETAT_SURFC": {
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 18,
        19: 19,
        20: 20,
        99: 99,
        "": "",
    },
    "CD_GENRE_ACCDN": {
        "véhicule": "vehicle",
        "piéton": "pedestrian",
        "cycliste": "cyclist",
        "animal": "animal",
        "objet fixe": "fixed object",
        "sans collision": "no collision",
        "autre": "other",
    },
    "CD_LOCLN_ACCDN": {12: 12, 33: 33, 34: 34, 40: 40, 69: 69, 99: 99, "": ""},
    "CD_ZON_TRAVX_ROUTR": {"O": "Y"},
    "GRAVITE": {
        "Mortel ou grave": "Fatal or serious",
        "Léger": "Minor",
        "Dommages matériels seulement": "Material damage only",
        "Dommages matériels inférieurs au seuil de rapportage": "Material damage below the reporting threshold",
    },
    "HR_ACCDN": {
        "00:00:00-03:59:00": "00:00:00-03:59:00",
        "04:00:00-07:59:00": "04:00:00-07:59:00",
        "08:00:00-11:59:00": "08:00:00-11:59:00",
        "12:00:00-15:59:00": "12:00:00-15:59:00",
        "16:00:00-19:59:00": "16:00:00-19:59:00",
        "20:00:00-23:59:00": "20:00:00-23:59:00",
    },
    "IND_AUTO_CAMION_LEGER": {"O": "Y", "N": "N"},
    "IND_MOTO_CYCLO": {"O": "Y", "N": "N"},
    "IND_PIETON": {"O": "Y", "N": "N"},
    "IND_VEH_LOURD": {"O": "Y", "N": "N"},
    "IND_VELO": {"O": "Y", "N": "N"},
    "JR_SEMN_ACCDN": {"SEM": "WKDY", "FDS": "WKND"},
    "MS_ACCDN": {m: m for m in range(1, 13)},
    "NB_VEH_IMPLIQUES_ACCDN": {1: 1, 2: 2, 9: 9, "": ""},
    "NB_VICTIMES_TOTAL": {0: 0, 1: 1, 2: 2, 9: 9},
    "NO_SEQ_COLL": {},
    "REG_ADM": {
        "Montréal (06)": "Montréal (06)",
        "Laval (13)": "Laval (13)",
        "Côte-Nord (09)": "Côte-Nord (09)",
        "Nord-du-Québec (10)": "Nord-du-Québec (10)",
        "Saguenay/-Lac-Saint-Jean (02)": "Saguenay/-Lac-Saint-Jean (02)",
        "Outaouais (07)": "Outaouais (07)",
        "Laurentides (15)": "Laurentides (15)",
        "Montérégie (16)": "Montérégie (16)",
        "Capitale-Nationale (03)": "Capitale-Nationale (03)",
        "Estrie (05)": "Estrie (05)",
        "Mauricie (04)": "Mauricie (04)",
        "Gaspésie/-Îles-de-la-Madeleine (11)": "Gaspésie/-Îles-de-la-Madeleine (11)",
        "Bas-Saint-Laurent (01)": "Bas-Saint-Laurent (01)",
        "Centre-du-Québec (17)": "Centre-du-Québec (17)",
        "Chaudière-Appalaches (12)": "Chaudière-Appalaches (12)",
        "Lanaudière (14)": "Lanaudière (14)",
        "Abitibi-Témiscamingue (08)": "Abitibi-Témiscamingue (08)",
        "": "",
    },
    "VITESSE_AUTOR": {
        "<50": "<50",
        50: 50,
        60: 60,
        70: 70,
        80: 80,
        90: 90,
        100: 100,
        "": "",
    },
}

# And the version with English keys and French values:
values_en = {
    FR_EN[key]: {en: fr for (fr, en) in values[key].items()} for key in values.keys()
}

# Define mappings for original SAAQ codes/values to something shorter and/or more human-readble (in some cases).
shorthand = dict.fromkeys(FR_EN.values())
shorthand["REGION"] = {
    "Montréal (06)": "Montréal",
    "Laval (13)": "Laval",
    "Côte-Nord (09)": "Côte-Nord",
    "Nord-du-Québec (10)": "Nord-du-Québec",
    "Saguenay/-Lac-Saint-Jean (02)": "Saguenay/Lac-Saint-Jean",
    "Outaouais (07)": "Outaouais",
    "Laurentides (15)": "Laurentides",
    "Montérégie (16)": "Montérégie",
    "Capitale-Nationale (03)": "Capitale-Nationale",
    "Estrie (05)": "Estrie",
    "Mauricie (04)": "Mauricie",
    "Gaspésie/-Îles-de-la-Madeleine (11)": "Gaspésie/Îles-de-la-Madeleine",
    "Bas-Saint-Laurent (01)": "Bas-Saint-Laurent",
    "Centre-du-Québec (17)": "Centre-du-Québec",
    "Chaudière-Appalaches (12)": "Chaudière-Appalaches",
    "Lanaudière (14)": "Lanaudière",
    "Abitibi-Témiscamingue (08)": "Abitibi-Témiscamingue",
    "": "",
}
shorthand["NUM_VEH"] = {1: 1, 2: 2, 9: "3+"}
shorthand["NUM_VICTIMS"] = {0: 0, 1: 1, 2: 2, 9: "3+"}
shorthand["HOUR"] = {
    "00:00:00-03:59:00": "00:00-04:00",
    "04:00:00-07:59:00": "04:00-08:00",
    "08:00:00-11:59:00": "08:00-12:00",
    "12:00:00-15:59:00": "12:00-16:00",
    "16:00:00-19:59:00": "16:00-20:00",
    "20:00:00-23:59:00": "20:00-24:00",
}
shorthand["SEVERITY"] = {
    "Material damage below the reporting threshold": "Mat < 2000",
    "Material damage only": "Mat",
    "Minor": "Minor",
    "Fatal or serious": "Fatal/serious",
}
shorthand["TNRY_SEV"] = { 0 : "Mat", 1: "Minor", 2: "Fatal/serious"}
shorthand["MONTH"] = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

shorthand["ACCDN_TYPE"] = {
    "vehicle": "veh",
    "pedestrian": "ped",
    "cyclist": "cyc",
    "animal": "anim",
    "fixed object": "fxd obj",
    "no collision": "no coll",
    "other": "oth",
}

shorthand["RD_COND"] = {
    11: "Dry",
    12: "Wet",
    13: "Water acc",
    14: "Sand/gravel",
    15: "Slush/sleet",
    16: "Snowy",
    17: "Hard snow",
    18: "Frozen",
    19: "Muddy",
    20: "Oily",
    99: "Other",
}
shorthand["LIGHT"] = {
    1: "Clear day",
    2: "Dawn/dusk",
    3: "Lit path",
    4: "Unlit path",
}
shorthand["ZONE"] = {
    1: "School",
    2: "Residential",
    3: "Business/commercial",
    4: "Industrial",
    5: "Rural",
    6: "Forest",
    9: "Other",
}
shorthand["PUB_PRIV_RD"] = {
    1: "Public",
    2: "Private",
}
shorthand["ASPECT"] = {
    "Straight": "Straight",
    "Curve": "Curve",
}
shorthand["LONG_LOC"] = {
    12: "Int'n/roundabout",
    33: "Near int'n/roundabout",
    34: "Btwn int'ns",
    40: "Shop centre",
    69: "Bridge etc.",
    99: "Other",
}
shorthand["RD_CONFG"] = {
    1: "One-way",
    23: "Two-way",
    45: "Sep by layout",
    9: "Other",
}
shorthand["WEATHER"] = {
    11: "Clear",
    12: "Overcast",
    13: "Fog/haze",
    14: "Rain/drizzle",
    15: "Downpour",
    16: "Strong wind",
    17: "Snow/hail",
    18: "Blowing snow/snowstorm",
    19: "Black ice",
    99: "Other",
}

shorthand["WKDY_WKND"] = { "WKDY" : "weekday", "WKND" : "weekend" }
shorthand["SPD_LIM"] = {
        "<50": "<50",
        50: 50,
        60: 60,
        70: 70,
        80: 80,
        90: 90,
        100: 100,}
shorthand["RDWX"] = { "Y" : "Y", "N" : "N" }
shorthand["LT_TRK"] = { "Y" : "Y", "N" : "N" }
shorthand["HVY_VEH"] = { "Y" : "Y", "N" : "N" }
shorthand["PED"] = { "Y" : "Y", "N" : "N" }
shorthand["BICYC"] = { "Y" : "Y", "N" : "N" }
shorthand["MTRCYC"] = { "Y" : "Y", "N" : "N" }

selection_dictionary_shorthand = { feature : shorthand[feature] for feature, dictionary in selection_dictionary.items() }

inverse_shorthand = {
    variable: {value: key for key, value in shorthand[variable].items()}
    for variable in shorthand.keys()
    if shorthand[variable] is not None
}

# EXPLANATION OF SAAQ SOURCE DATAFRAME CODES

explain_fr = {
    col_name: dict.fromkeys(["summary", "values", "elaborate"])
    for col_name in FR_EN.keys()
}
explain_en = {
    col_name: dict.fromkeys(["summary", "values", "elaborate"])
    for col_name in FR_EN.values()
}

for key in values.keys():
    explain_fr[key]["values"] = values[key].keys()
    explain_en[FR_EN[key]]["values"] = values[key].values()

# "AN": "YEAR"

explain_fr["AN"]["summary"] = "Année de l'accident (AAAA)."
explain_en[FR_EN["AN"]]["summary"] = "Year of accident (YYYY)."


# "NO_SEQ_COLL": "ID"

explain_fr["NO_SEQ_COLL"]["summary"] = "Numéro séquentiel identifiant l'accident."
explain_en[FR_EN["NO_SEQ_COLL"]][
    "summary"
] = "Sequential number identifying the accident."

elab_id_fr = [
    "Composé de l'année de l'accident et d'un numéro séquentiel.",
    "Exemple: AAAA _ 999, où l'année et le numéro séquentiel sont séparés par un espace, une barre de soulignement et un espace.",
]
elab_id_en = [
    "Composed of the year of the accident and a sequential number.",
    "Example: YYYY_999, where the year and sequential number are separated by a space, an underscore and a space.",
]

explain_fr["NO_SEQ_COLL"]["elaborate"] = [" ".join(elab_id_fr)]
explain_en[FR_EN["NO_SEQ_COLL"]]["elaborate"] = [" ".join(elab_id_en)]

# "MS_ACCDN": "MONTH"

explain_fr["MS_ACCDN"]["summary"] = "Mois de l'accident."
explain_en[FR_EN["MS_ACCDN"]]["summary"] = "Month of the accident."

elab_months_fr = [
    "Janvier",
    "Février",
    "Mars",
    "Avril",
    "Mai",
    "Juin",
    "Juillet",
    "Août",
    "Septembre",
    "Octobre",
    "Novembre",
    "Décembre",
]
elab_months_en = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

explain_fr["MS_ACCDN"]["elaborate"] = elab_months_fr
explain_en[FR_EN["MS_ACCDN"]]["elaborate"] = elab_months_en

# "HR_ACCDN": "HOUR"

explain_fr["HR_ACCDN"][
    "summary"
] = "Heure de l'accident. Intervalle de 4 heures, contenant l'heure réelle de l'accident."
explain_en[FR_EN["HR_ACCDN"]][
    "summary"
] = "Hour of accident. Four-hour interval, containing the actual time of the accident."

elab_hours_fr = [
    "Entre minuit et 4h00.",
    "Entre 4h00 et 8h00.",
    "Entre 8h00 et midi.",
    "Entre midi et 16h00.",
    "Entre 16h00 et 20h00.",
    "Entre 20h00 et minuit.",
]
elab_hours_en = [
    "Between midnight and 4:00am.",
    "Between 4:00am and 8:00am.",
    "Between 8:00am and noon.",
    "Between noon and 4:00pm.",
    "Between 4:00pm and 8:00pm.",
    "Between 8:00pm and midnight.",
]

explain_fr["HR_ACCDN"]["elaborate"] = elab_hours_fr
explain_en[FR_EN["HR_ACCDN"]]["elaborate"] = elab_hours_en

# "JR_SEMN_ACCDN": "WKDY_WKND"

explain_fr["JR_SEMN_ACCDN"]["summary"] = "Jour de la semaine de la date de l'accident."
explain_en[FR_EN["JR_SEMN_ACCDN"]]["summary"] = "Weekday of the date of the accident."

elab_wkdy_wknd_fr = ["Lundi au vendredi.", "Samedi ou dimanche."]
elab_wkdy_wknd_en = ["Monday through Friday", "Saturday or Sunday."]

explain_fr["JR_SEMN_ACCDN"]["elaborate"] = elab_wkdy_wknd_fr
explain_en[FR_EN["JR_SEMN_ACCDN"]]["elaborate"] = elab_wkdy_wknd_en

# "GRAVITE": "SEVERITY"

explain_fr["GRAVITE"][
    "summary"
] = "Gravité de l'accident: Indique la gravité de l'accident en fonction de la présence et de l'état des victimes."
explain_en[FR_EN["GRAVITE"]][
    "summary"
] = "Severity of the accident: Indicates the severity of the accident based on the presence and condition of victims"

elab_severity_fr = [
    "Au moins une victime décédée dans les 30 jours suivant l'accident ou aucun décès et au moins une victime blessée gravement (blessures nécessitant l'hospitalisation, incluant celles pour lesquelles la personne demeure en observation à l'hôpital).",
    "Seulement une ou plusieurs victimes blessées légèrement (blessures ne nécessitant pas l'hospitalisation ni la mise en observation de la personne, même si elles exigent des traitements chez un médecin ou dans un centre hospitalier).",
    "Aucune victime, et l'évaluation des dommages est supérieure au seuil de rapportage (seuil de 2 000$ depuis mars 2010).",
    "Aucune victime, et l'évaluation des dommages est inférieure ou égale au seuil de rapportage (seuil de 2 000$ depuis mars 2010).",
]

elab_severity_en = [
    "At least one victim died within 30 days following the accident or no death and at least one victim seriously injured (injuries requiring hospitalization, including those for which the person remains under observation in hospital).",
    "Only one or more victims lightly injured (injuries not requiring hospitalization or observation of the person, even if they require treatment by a doctor or in a hospital center).",
    "No victims, and the damage assessment is above the reporting threshold (threshold of $2,000 since March 2010).",
    "No victims, and the assessment of damage is less than or equal to the reporting threshold (threshold of $2,000 since March 2010).",
]

explain_fr["GRAVITE"]["elaborate"] = elab_severity_fr
explain_en[FR_EN["GRAVITE"]]["elaborate"] = elab_severity_en

# "NB_VICTIMES_TOTAL": "NUM_VICTIMS"

explain_fr["NB_VICTIMES_TOTAL"][
    "summary"
] = "Nombre total de victimes (décès, blessés graves et blessés légers) dans l'accident."
explain_en[FR_EN["NB_VICTIMES_TOTAL"]][
    "summary"
] = "Total number of victims (deaths, serious injuries and minor injuries) in the accident."

elab_num_victims_fr = [
    "Aucune victime est décédée ou n'a été blessée.",
    "Une victime est décédée ou a été blessée.",
    "Deux victimes sont décédées ou ont été blessées.",
    "Trois victimes ou plus sont décédées ou ont été blessées.",
]
elab_num_victims_en = [
    "No victims died or were injured.",
    "One victim died or was injured.",
    "Two victims died or were injured.",
    "Three or more victims died or were injured.",
]

explain_fr["NB_VICTIMES_TOTAL"]["elaborate"] = elab_num_victims_fr
explain_en[FR_EN["NB_VICTIMES_TOTAL"]]["elaborate"] = elab_num_victims_en

# "NB_VEH_IMPLIQUES_ACCDN": "NUM_VEH"

explain_fr["NB_VEH_IMPLIQUES_ACCDN"][
    "summary"
] = "Nombre de véhicules impliqués dans l'accident."
explain_en[FR_EN["NB_VEH_IMPLIQUES_ACCDN"]][
    "summary"
] = "Number of vehicles involved in the accident."

elab_num_veh_fr = [
    "Un seul véhicule routier a été impliqué.",
    "Deux véhicules routiers ont été impliqués.",
    "Trois véhicules routiers ou plus ont été impliqués.",
    "Non précisé.",
]
elab_num_veh_en = [
    "One road vehicle only was involved.",
    "Two road vehicles were involved.",
    "Three or more road vehicles were involved.",
    "Unspecified.",
]

explain_fr["NB_VEH_IMPLIQUES_ACCDN"]["elaborate"] = elab_num_veh_fr
explain_en[FR_EN["NB_VEH_IMPLIQUES_ACCDN"]]["elaborate"] = elab_num_veh_en

# "REG_ADM": "REGION"

explain_fr["REG_ADM"][
    "summary"
] = "Région administrative du Québec: La région administrative est déterminée à partir du code de municipalité."
explain_en[FR_EN["REG_ADM"]][
    "summary"
] = "Administrative region of Quebec: The administrative region is determined from the municipality code."

# "VITESSE_AUTOR": "SPD_LIM"

explain_fr["VITESSE_AUTOR"][
    "summary"
] = "Vitesse autorisée. Vitesse permise, en kilomètres/heure, par l'autorité compétente sur la route où est survenu l'accident. Pour un accident à une intersection, la vitesse la plus élevée parmi celles des routes qui se croisent sera inscrite."
explain_en[FR_EN["VITESSE_AUTOR"]][
    "summary"
] = "Speed limit. Speed permitted, in kilometers/hour, by the competent authority on the road where the accident occurred. For an accident at an intersection, the highest speed among those of the intersecting roads will be recorded."

elab_spd_lim_fr = [
    "Moins de 50 km/h.",
    "Vitesse permise en km/h.",
    "Vitesse permise en km/h.",
    "Vitesse permise en km/h.",
    "Vitesse permise en km/h.",
    "Vitesse permise en km/h.",
    "Vitesse permise en km/h.",
    "Non précisé.",
]
elab_spd_lim_en = [
    "Less than 50 km/h.",
    "Speed limit in km/h.",
    "Speed limit in km/h.",
    "Speed limit in km/h.",
    "Speed limit in km/h.",
    "Speed limit in km/h.",
    "Speed limit in km/h.",
    "Unspecified.",
]

explain_fr["VITESSE_AUTOR"]["elaborate"] = elab_spd_lim_fr
explain_en[FR_EN["VITESSE_AUTOR"]]["elaborate"] = elab_spd_lim_en

# "CD_GENRE_ACCDN": "ACCDN_TYPE"

explain_fr["CD_GENRE_ACCDN"][
    "summary"
] = "Genre d'accident: Sert à indiquer la nature de l'accident et le premier fait physique (impact)."
explain_en[FR_EN["CD_GENRE_ACCDN"]][
    "summary"
] = "Type of accident: Used to indicate the nature of the accident and the first physical event (impact)."

fixed_obj_bkdn_fr = [
    "Lampadaire: Support fixe servant à soutenir des équipements d'éclairage.",
    "Support/feu de signalisation: Support servant à soutenir de façon permanente des panneaux de signalisation ou des feux de signalisation.",
    "Poteau (service public): Support fixe servant à soutenir des équipements d'utilité publique, autre qu'un lampadaire ou un feu de signalisation.",
    "Arbre: Toute espèce d'arbres, excluant les haies.",
    "Section de glissière de sécurité: Section d'un dispositif de protection en tôle ondulée, en béton (New Jersey) ou en acier, servant à retenir les véhicules routiers quittant la chaussée.",
    "Atténuateur d'impact: Dispositif de sécurité installé devant des obstacles fixes le long d'une route pour réduire les préjudices corporels et les dommages matériels, lorsqu'un véhicule quitte la route devant l'obstacle.",
    "Extrémité de glissière de sécurité: Bout de la glissière, excluant les atténuateurs d'impact.",
    "Pilier (pont/tunnel): Partie d'une structure (pont, tunnel, viaduc) qui en supporte la charge.",
    "Amoncellement de neige: Accumulation de neige.",
    "Bâtiment/édifice/mur: Comprend, entre autres, toute construction servant à abriter des individus, des animaux ou des choses.",
    "Bordure/trottoir: Bande qui limite le bord de la chaussée ou l'accotement, ou chemin surélevé longeant une rue et réservé aux piétons.",
    "Borne-fontaine: Pièce d'équipement servant de point d'eau et généralement utilisée pour combattre un incendie.",
    "Clôture/barrière: Enceinte qui délimite un espace. Inclut aussi les haies.",
    "Fossé: Tranchée ou canal aménagé le long d'une route, servant à l'écoulement de l'eau.",
    "Paroi rocheuse: Mur de roc longeant une route.",
    "Ponceau: Conduite de drainage, généralement en béton ou en métal, servant au passage de l'eau sous une route, une entrée, un accès.",
    "Autre objet fixe: Tout objet impliqué dans une collision, autre que ceux décrits précédemment.",
]

join_fixed_obj_bkdn_fr = "\n* " + "\n* ".join(fixed_obj_bkdn_fr)

fixed_obj_bkdn_en = [
    "Lamp: Fixed support used to support lighting equipment.",
    "Support/signal light: Support used to permanently support traffic signs or traffic lights.",
    "Pole (public service): Fixed support used to support public utility equipment, other than a street lamp or a traffic light.",
    "Tree: Any species of tree, excluding hedges.",
    "Crash barrier section: Section of a protective device made of corrugated iron, concrete (New Jersey) or steel, used to retain road vehicles leaving the roadway.",
    "Impact attenuator: Safety device installed in front of fixed obstacles along a road to reduce bodily injury and property damage when a vehicle leaves the road in front of the obstacle.",
    "End of guardrail: End of guardrail, excluding impact attenuators.",
    "Pillar (bridge/tunnel): Part of a structure (bridge, tunnel, viaduct) which supports the load.",
    "Snow drift: Snow accumulation.",
    "Building/edifice/wall: Includes, among others, any construction used to shelter individuals, animals or things.",
    "Curb/sidewalk: Strip that limits the edge of the roadway or shoulder, or raised path running along a street and reserved for pedestrians.",
    "Fire hydrant: Piece of equipment serving as a water point and generally used to fight a fire.",
    "Fence/barrier: Enclosure which demarcates a space. Also includes hedges.",
    "Ditch: Trench or canal built along a road, used for the flow of water.",
    "Rock wall: Rock wall along a road.",
    "Culvert: Drainage pipe, generally made of concrete or metal, used to pass water under a road, an entrance, an access.",
    "Other fixed object: Any object involved in a collision, other than those described previously.",
]

join_fixed_obj_bkdn_en = "\n* " + "\n* ".join(fixed_obj_bkdn_en)

no_collision_bkdn_fr = [
    "Capotage: Lorsqu'un véhicule a culbuté ou capoté.",
    "Renversement: Lorsqu'un véhicule se retrouve sur un de ses côtés, sans avoir capoté.",
    "Submersion/cours d'eau: Lorsqu'un véhicule a plongé dans l'eau ou qu'il se retrouve dans un cours d'eau (ex.: rivière, lac).",
    "Feu/explosion: Lorsqu'un véhicule a pris feu ou a explosé.",
    "Quitte la chaussée: Lorsqu'un véhicule quitte la surface de roulement à la suite de la perte de contrôle du conducteur.",
    "Autre: Tout événement sans collision, autre que ceux décrits précédemment.",
]

join_no_collision_bkdn_fr = "\n* " + "\n* ".join(no_collision_bkdn_fr)

no_collision_bkdn_en = [
    "Rollover: When a vehicle has overturned or rolled over.",
    "Rollover: When a vehicle ends up on one of its sides, without having rolled over.",
    "Submersion/watercourse: When a vehicle has plunged into water or finds itself in a watercourse (e.g.: river, lake).",
    "Fire/explosion: When a vehicle catches fire or explodes.",
    "Leaves the roadway: When a vehicle leaves the running surface following the loss of driver control.",
    "Other: Any non-collision event, other than those described previously.",
]

join_no_collision_bkdn_en = "\n* " + "\n* ".join(no_collision_bkdn_en)

elab_accdn_type_fr = [
    "Collision avec véhicule routier. Véhicule routier: automobile ou camion léger, camion, tracteur routier, véhicule outil, véhicule d'équipement, autobus, minibus, taxi, véhicule d'urgence, motocyclette, cyclomoteur, véhicule récréatif, motoneige, VHR, motocyclette visée par la loi VHR.",
    "Collision avec piéton. Toute personne qui circule à pied, tire, pousse un objet ou se trouve sur ou dans cet objet. Toute personne qui utilise un équipement qui n'est pas autorisé à circuler sur un chemin public est assimilée à un piéton.",
    "Collision avec cycliste. Toute personne qui circule à bicyclette (assistée ou non), en tricycle, en monocycle ou en quadricycle.",
    "Collision avec animal. Animal domestique ou sauvage.",
    join_fixed_obj_bkdn_fr,
    join_no_collision_bkdn_fr,
    "Autre genre de collision. Tout genre de collision, autre que ceux décrits précédemment. Exemples: une collision avec un train, un obstacle temporaire ou un objet projeté/détaché.",
    "Non précisé.",
]

elab_accdn_type_en = [
    "Collision with road vehicle. Road vehicle: automobile or light truck, truck, road tractor, tool vehicle, equipment vehicle, bus, minibus, taxi, emergency vehicle, motorcycle, moped, recreational vehicle, snowmobile, VHR, motorcycle covered by the VHR law.",
    "Collision with pedestrian. Any person who walks, pulls, pushes an object or is on or in this object. Any person who uses equipment that is not authorized to travel on a public road is considered a pedestrian.",
    "Collision with cyclist. Any person who rides a bicycle (assisted or not), tricycle, unicycle or quadricycle.",
    "Collision with animal. Domestic or wild animal.",
    join_fixed_obj_bkdn_en,
    join_no_collision_bkdn_en,
    "Other type of collision. Any kind of collision, other than those described above. Examples: a collision with a train, a temporary obstacle or an object projected/detached.",
    "Unspecified.",
]

explain_fr["CD_GENRE_ACCDN"]["elaborate"] = elab_accdn_type_fr
explain_en[FR_EN["CD_GENRE_ACCDN"]]["elaborate"] = elab_accdn_type_en

# "CD_ETAT_SURFC": "RD_COND"

explain_fr["CD_ETAT_SURFC"][
    "summary"
] = "État de la surface de roulement lors de l'accident."
explain_en[FR_EN["CD_ETAT_SURFC"]][
    "summary"
] = "Condition of the road surface during the accident."

elab_rd_cond_fr = [
    "Sèche. Surface qui n'a reçu aucun liquide ou aucun matériau nuisant à l'adhérence des pneus.",
    "Mouillée. Surface qui a reçu un liquide de nature à diminuer l'adhérence entre le véhicule et la surface (autre qu'une substance huileuse ou graisseuse).",
    "Accumulation d'eau (aquaplanage). Surface où une pellicule d'eau entre la chaussée et les pneus entraîne la perte complète d'adhérence d'un véhicule.",
    "Sable, gravier sur la chaussée. Surface couverte de sable ou de gravier. ",
    "Gadoue/neige fondante. Surface qui est couverte de neige mouillée ou de neige fondante.",
    "Enneigée. Surface qui est couverte de neige.",
    "Neige durcie. Surface couverte de neige qui est compactée et qui a durci.",
    "Glacée. Surface qui a perdu son adhérence à la suite de l'apparition de glace. ",
    "Boueuse. Surface d'un chemin de terre à la suite d'une pluie ou toute autre surface qui a perdu de son adhérence par la présence de boue.",
    "Huileuse. Présence d'huile ou de produit graisseux sur la chaussée.",
    "Autre. Tout état de la surface, autre que ceux décrits précédemment.",
    "Non précisé.",
]

elab_rd_cond_en = [
    "Dry. Surface that has not received any liquid or any material that affects tire grip.",
    "Wet. Surface which has received a liquid likely to reduce adhesion between the vehicle and the surface (other than an oily or greasy substance).",
    "Water accumulation (hydroplaning). Surface where a film of water between the road surface and the tires causes loss complete grip of a vehicle.",
    "Sand, gravel on the roadway. Surface covered with sand or gravel.",
    "Slush/sleet. Surface that is covered with wet snow or slush.",
    "Snowy. Surface that is covered with snow.",
    "Hard snow. Surface covered with snow that is compacted and hardened.",
    "Frozen. Surface that has lost its grip following the appearance of ice.",
    "Muddy. Surface of a dirt road following rain or any other surface that has lost its grip due to the presence of mud.",
    "Oily. Presence of oil or greasy product on the road.",
    "Other. Any surface condition, other than those described previously.",
    "Unspecified.",
]

explain_fr["CD_ETAT_SURFC"]["elaborate"] = elab_rd_cond_fr
explain_en[FR_EN["CD_ETAT_SURFC"]]["elaborate"] = elab_rd_cond_en

# "CD_ECLRM": "LIGHT"

explain_fr["CD_ECLRM"][
    "summary"
] = "Éclairement: Degré de clarté des lieux au moment de l'accident. L'éclairement fait référence à deux périodes d'une journée, soit le jour et la nuit."
explain_en[FR_EN["CD_ECLRM"]][
    "summary"
] = "Illumination: Degree of clarity of the premises at the time of the accident. Illumination refers to two periods of a day, day and night."

elab_light_fr = [
    "Jour et clarté. Jour: Période comprise entre une demi-heure avant le lever du soleil et une demi-heure après son coucher. Clarté: Fait référence à la période comprise entre le lever du soleil et son coucher.",
    "Jour et demi-obscurité. Jour: Période comprise entre une demi-heure avant le lever du soleil et une demi-heure après son coucher. Demi-obscurité: Fait référence à la période entre la nuit et le lever du soleil et à la période entre le coucher du soleil et la nuit.",
    "Nuit et chemin éclairé. Nuit: Période comprise entre une demi-heure après le coucher du soleil et une demi-heure avant son lever. Chemin éclairé: Chemin le long duquel sont installés des équipements d'éclairage qui fonctionnaient au moment de l'accident.",
    "Nuit et chemin non éclairé. Nuit: Période comprise entre une demi-heure après le coucher du soleil et une demi-heure avant son lever. Chemin non éclairé: Chemin le long duquel, dans la région immédiate de l'accident, aucun équipement d'éclairage n'est installé ou l'équipement en place ne fonctionnait pas.",
    "Non précisé.",
]

elab_light_en = [
    "Day and clarity. Day: Period between half an hour before sunrise and half an hour after sunset. Clarity: Refers to the period between sunrise and sunset.",
    "Day and half darkness. Day: Period between half an hour before sunrise and half an hour after sunset. Half-darkness: Refers to the period between night and sunrise and the period between sunset and night.",
    "Night and lit path. Night: Period between half an hour after sunset and one half an hour before getting up. Illuminated path: Path along which lighting equipment is installed which was operating at the time of the accident.",
    "Night and unlit path. Night: Period between half an hour after sunset and one half an hour before getting up. Unlit path: Path along which, in the immediate area of the accident, no lighting equipment is installed or the equipment in place was not functioning.",
    "Unspecified.",
]

explain_fr["CD_ECLRM"]["elaborate"] = elab_light_fr
explain_en[FR_EN["CD_ECLRM"]]["elaborate"] = elab_light_en

# "CD_ENVRN_ACCDN": "ZONE"

explain_fr["CD_ENVRN_ACCDN"][
    "summary"
] = "Environnement: Activité dominante du secteur où l'accident s'est produit."
explain_en[FR_EN["CD_ENVRN_ACCDN"]][
    "summary"
] = "Zone: Dominant activity in the sector where the accident occurred."

elab_zone_fr = [
    "Scolaire. Région immédiate d'un établissement d'enseignement.",
    "Résidentiel. Secteur domiciliaire principalement.",
    "Affaires / commercial. Secteur où l'activité principale est d'ordre commercial, administratif ou d'affaires.",
    "Industriel / manufacturier. Secteur où l'activité principale est d'ordre industriel, manufacturier.",
    "Rural. Secteur hors des limites des cités, villes et villages, sauf le secteur forestier.",
    "Forestier. Secteur où l'activité principale est l'exploitation forestière ou forêt, même si on y trouve quelques habitations.",
    "Autre (ex. lac, récréatif, parc, camping). Toute activité dominante du secteur, autre que celles mentionnées précédemment.",
    "Non précisé.",
]
elab_zone_en = [
    "School. Immediate area of an educational institution.",
    "Residential. Mainly residential sector.",
    "Business/commercial. Sector where the main activity is commercial, administrative or business.",
    "Industrial / manufacturing. Sector where the main activity is industrial, manufacturing.",
    "Rural. Sector outside the limits of cities, towns and villages, except the forestry sector.",
    "Forest road. Sector where the main activity is logging or forestry, even if there is found some homes.",
    "Other (e.g. lake, recreational, park, campsite). Any dominant activity in the sector, other than those mentioned previously.",
    "Unspecified.",
]

explain_fr["CD_ENVRN_ACCDN"]["elaborate"] = elab_zone_fr
explain_en[FR_EN["CD_ENVRN_ACCDN"]]["elaborate"] = elab_zone_en

# "CD_CATEG_ROUTE": "PUB_PRIV_RD"

explain_fr["CD_CATEG_ROUTE"][
    "summary"
] = "Catégorie de route sur laquelle le premier fait physique (impact) s'est produit."
explain_en[FR_EN["CD_CATEG_ROUTE"]][
    "summary"
] = "Category of road on which the first physical event (impact) occurred."

elab_rd_type_fr = [
    "Chemin public. Exemples: route numérotée, bretelle, collecteur d'autoroute, voie de service, artère principale, rue résidentielle, chemin, rang, ruelle.",
    "Hors chemin public. Exemples: terrain de stationnement, terrain privé, chemin privé, chemin forestier, sentier balisé.",
    "Non précisé.",
]
elab_rd_type_en = [
    "Public road. Examples: numbered road, ramp, highway collector, service road, main artery, residential street, path, row, alley.",
    "Off public roads. Examples: parking lot, private land, private road, forest road, marked trail.",
    "Unspecified.",
]

explain_fr["CD_CATEG_ROUTE"]["elaborate"] = elab_rd_type_fr
explain_en[FR_EN["CD_CATEG_ROUTE"]]["elaborate"] = elab_rd_type_en

# "CD_ASPCT_ROUTE": "ASPECT"

explain_fr["CD_ASPCT_ROUTE"][
    "summary"
] = "Aspect de la route sur le lieu de l'accident au moment de l'impact et dans son entourage immédiat en fonction du champ de vision d'un conducteur assis au volant de son véhicule."
explain_en[FR_EN["CD_ASPCT_ROUTE"]][
    "summary"
] = "Appearance of the road at the scene of the accident at the time of impact and in its immediate surroundings depending on the field of vision of a driver seated at the wheel of their vehicle."

elab_aspect_fr = [
    "Chaussée où la direction de la circulation est relativement droite.",
    "Chaussée où la direction de la circulation tourne vers la gauche ou vers la droite.",
    "Non précisé.",
]
elab_aspect_en = [
    "Roadway where the direction of traffic is relatively straight.",
    "Roadway where the direction of traffic turns to the left or right.",
    "Unspecified.",
]

explain_fr["CD_ASPCT_ROUTE"]["elaborate"] = elab_aspect_fr
explain_en[FR_EN["CD_ASPCT_ROUTE"]]["elaborate"] = elab_aspect_en

# "CD_LOCLN_ACCDN": "LONG_LOC"

explain_fr["CD_LOCLN_ACCDN"][
    "summary"
] = "Localisation longitudinale (le long de la route) du premier fait physique (impact)."
explain_en[FR_EN["CD_LOCLN_ACCDN"]][
    "summary"
] = "Longitudinal location (along the road) of the first physical event (impact)."

six_nine_bkdn_fr = [
    "Pont (au-dessus d'un cours d'eau). Structure permettant de traverser un cours d'eau.",
    "Autre pont (viaduc). Structure permettant de traverser une voie de circulation routière ou ferroviaire, ou tout autre obstacle, autre qu'un cours d'eau.",
    "Tunnel. Galerie souterraine de grande section permettant le passage d'une voie de communication.",
    "Sous un pont ou un viaduc. En dessous d'une structure permettant de traverser des obstacles.",
]

join_six_nine_bkdn_fr = "\n* " + "\n* ".join(six_nine_bkdn_fr)

six_nine_bkdn_en = [
    "Bridge (over a stream). Structure allowing a river to be crossed.",
    "Another bridge (viaduct). Structure allowing crossing of a road or rail traffic lane, or any other obstacle, other than a watercourse.",
    "Tunnel. Large section underground gallery allowing the passage of a communication route.",
    "Under a bridge or viaduct. Below a structure allowing you to cross obstacles.",
]

join_six_nine_bkdn_en = "\n* " + "\n* ".join(six_nine_bkdn_en)

elab_long_loc_fr = [
    "En intersection (moins de 5 mètres) ou dans un carrefour giratoire.",
    "Près d'une intersection/carrefour giratoire.",
    "Entre intersections (100 mètres et +).",
    "Centre commercial.",
    join_six_nine_bkdn_fr,
    "Autres. Toute localisation, autre que celles décrites précédemment.",
    "Non précisé.",
]
elab_long_loc_en = [
    "At an intersection (less than 5 meters) or in a roundabout.",
    "Near an intersection/roundabout.",
    "Between intersections (100 meters and more).",
    "Shopping centre.",
    join_six_nine_bkdn_en,
    "Others. Any location, other than those described above.",
    "Unspecified.",
]

explain_fr["CD_LOCLN_ACCDN"]["elaborate"] = elab_long_loc_fr
explain_en[FR_EN["CD_LOCLN_ACCDN"]]["elaborate"] = elab_long_loc_en

# "CD_CONFG_ROUTE": "RD_CONFG"

explain_fr["CD_CONFG_ROUTE"][
    "summary"
] = "Configuration de la route: Caractéristiques des voies. Si l'accident a lieu à une intersection, la rue la plus importante de cette intersection est décrite."
explain_en[FR_EN["CD_CONFG_ROUTE"]][
    "summary"
] = "Road configuration: Lane characteristics. If the accident takes place at an intersection, the most important street of this intersection is described."

elab_rd_config_fr = [
    "Sens unique. La circulation des véhicules est autorisée dans une direction seulement, indiquée par une flèche.",
    "Deux sens. Les véhicules se déplacent dans les deux directions.",
    "Séparée par aménagement. Les courants de circulation sont séparés par un aménagement physique.",
    "Autre. Toute configuration, autre que celles mentionnées précédemment. Exemples: balises, voie de virage à gauche dans les deux sens. ",
    "Non précisé.",
]
elab_rd_config_en = [
    "One-way. Vehicle traffic is authorized in one direction only, indicated by an arrow.",
    "Two-way. Vehicles move in both directions.",
    "Separated by layout. Traffic flows are separated by a physical layout.",
    "Other. Any configuration, other than those mentioned previously. Examples: markers, left turn lane in both directions.",
    "Unspecified.",
]

explain_fr["CD_CONFG_ROUTE"]["elaborate"] = elab_rd_config_fr
explain_en[FR_EN["CD_CONFG_ROUTE"]]["elaborate"] = elab_rd_config_en


# "CD_ZON_TRAVX_ROUTR": "RDWX"

explain_fr["CD_ZON_TRAVX_ROUTR"][
    "summary"
] = "Indicateur si l'accident a eu lieu près ou dans une zone de travaux."
explain_en[FR_EN["CD_ZON_TRAVX_ROUTR"]][
    "summary"
] = "Indicates that the accident took place near or in a work zone."

elab_work_fr = [
    "Aux approches de la zone. Zone en amont des travaux où les conducteurs sont avisés des changements de voies, de la réduction de vitesse, des interdictions de passer, etc.",
    "Dans la zone. Zone où il y a des modifications de la configuration de la route ou de la vitesse autorisée afin de permettre les travaux, ou dans laquelle il y a des travaux mobiles.",
]

elab_work_en = [
    "Approaching the area. Area upstream of the work where drivers are notified of changes in lanes, speed reduction, passing bans, etc.",
    "In the zone. Area where there are changes to the road configuration or authorized speed to allow work, or where there is mobile work.",
]

explain_fr["CD_ZON_TRAVX_ROUTR"]["elaborate"] = ["\n* " + "\n* ".join(elab_work_fr)]
explain_en[FR_EN["CD_ZON_TRAVX_ROUTR"]]["elaborate"] = [
    "\n* " + "\n* ".join(elab_work_en)
]

# "CD_COND_METEO": "WEATHER"

explain_fr["CD_COND_METEO"][
    "summary"
] = "Conditions météorologiques: Conditions atmosphériques présentes lors de l'accident."
explain_en[FR_EN["CD_COND_METEO"]][
    "summary"
] = "Meteorological conditions: Atmospheric conditions present during the accident."

elab_weather_fr = [
    "Clair. Absence totale de nuages ou présence de nuages qui n'a pas pour effet d'assombrir ou de rendre la vision moins distincte.",
    "Couvert (nuageux/sombre). Ciel couvert de nuages sombres et épais ayant pour effet d'assombrir et de rendre la vision moins distincte.",
    "Brouillard/brume. « Fumée blanche opaque » formée de très petites gouttelettes d'eau en suspension dans l'air.",
    "Pluie/bruine. Tombée régulière et continue de gouttelettes d'eau venant des nuages.",
    "Averse (pluie forte). Pluie soudaine et abondante.",
    "Vent fort (pas de poudrerie, pas de pluie). Déplacement d'air qui a pour effet de rendre un véhicule moins stable sur la route.",
    "Neige/grêle. Tombée de gouttelettes d'eau cristallisées sous forme de neige ou de grêle.",
    "Poudrerie/tempête. de neige Neige chassée par le vent (souvent en rafales) ou chute de neige accompagnée d'un vent violent.",
    "Verglas. Couche de glace, généralement très mince, qui se forme lorsque tombe une pluie surfondue venant en contact avec des corps solides au-dessous de zéro degré Celsius.",
    "Autre. Toute condition atmosphérique, autre que celles décrites précédemment.",
    "Non précisé.",
]
elab_weather_en = [
    "Clear. Total absence of clouds or presence of clouds which does not have the effect of darkening or making vision less distinct.",
    "Overcast (cloudy/dark). Sky covered with dark, thick clouds which have the effect of darkening and making vision less distinct.",
    'Fog/haze. "Opaque white smoke" formed from very small water droplets suspended in the air.',
    "Rain/drizzle. Regular and continuous fall of water droplets coming from the clouds.",
    "Downpour (heavy rain). Sudden and heavy rain.",
    "Strong wind (no blowing snow, no rain). Air movement which has the effect of making a vehicle less stable on the road.",
    "Snow/hail. Falling water droplets crystallized in the form of snow or hail.",
    "Blowing snow/snowstorm. Windblown snow (often gusty) or snowfall accompanied by strong wind.",
    "Black ice. A layer of ice, generally very thin, which forms when supercooled rain falls in contact with solid bodies below zero degrees Celsius.",
    "Other. Any atmospheric condition, other than those described above.",
    "Unspecified.",
]

explain_fr["CD_COND_METEO"]["elaborate"] = elab_weather_fr
explain_en[FR_EN["CD_COND_METEO"]]["elaborate"] = elab_weather_en

# "IND_AUTO_CAMION_LEGER": "LT_TRK"

explain_fr["IND_AUTO_CAMION_LEGER"][
    "summary"
] = "Indicateur d'au moins une automobile ou un camion léger impliqué dans l'accident. "
explain_en[FR_EN["IND_AUTO_CAMION_LEGER"]][
    "summary"
] = "Indicator of at least one automobile or light truck involved in the accident."

YesNo_fr = ["Oui.", "Non."]
YesNo_en = ["Yes.", "No."]

explain_fr["IND_AUTO_CAMION_LEGER"]["elaborate"] = YesNo_fr
explain_en[FR_EN["IND_AUTO_CAMION_LEGER"]]["elaborate"] = YesNo_en

# "IND_VEH_LOURD": "HVY_VEH"

explain_fr["IND_VEH_LOURD"][
    "summary"
] = "Indicateur d'au moins un véhicule lourd impliqué dans l'accident. Véhicule lourd: camion lourd, tracteur routier, autobus, autobus scolaire, minibus, véhicule-outil ou d'équipement."
explain_en[FR_EN["IND_VEH_LOURD"]][
    "summary"
] = "Indicator of at least one heavy vehicle involved in the accident. Heavy vehicle: heavy truck, road tractor, bus, school bus, minibus, tool or equipment vehicle."

explain_fr["IND_VEH_LOURD"]["elaborate"] = YesNo_fr
explain_en[FR_EN["IND_VEH_LOURD"]]["elaborate"] = YesNo_en

# "IND_MOTO_CYCLO": "MTRCYC"

explain_fr["IND_MOTO_CYCLO"][
    "summary"
] = "Indicateur d'au moins une motocyclette ou un cyclomoteur impliqué dans l'accident."
explain_en[FR_EN["IND_MOTO_CYCLO"]][
    "summary"
] = "Indicator of at least one motorcycle or moped involved in the accident."

explain_fr["IND_MOTO_CYCLO"]["elaborate"] = YesNo_fr
explain_en[FR_EN["IND_MOTO_CYCLO"]]["elaborate"] = YesNo_en

# "IND_VELO": "BICYC"

explain_fr["IND_VELO"][
    "summary"
] = "Indicateur d'au moins une bicyclette impliquée dans l'accident."
explain_en[FR_EN["IND_VELO"]][
    "summary"
] = "Indicator of at least one bicycle involved in the accident."

explain_fr["IND_VELO"]["elaborate"] = YesNo_fr
explain_en[FR_EN["IND_VELO"]]["elaborate"] = YesNo_en

# "IND_PIETON": "PED"

explain_fr["IND_PIETON"][
    "summary"
] = "Indicateur d'au moins une victime piéton (blessée ou décédée) dans l'accident."
explain_en[FR_EN["IND_PIETON"]][
    "summary"
] = "Indicator of at least one pedestrian victim (injured or died) in the accident."

explain_fr["IND_PIETON"]["elaborate"] = YesNo_fr
explain_en[FR_EN["IND_PIETON"]]["elaborate"] = YesNo_en

