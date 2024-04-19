from typing import Union
from utils import display, clear_output


def explain(terms: Union[list, None] = None) -> None:
    global ncdb_columns
    global ncdb_explain
    if terms is None:
        terms = ncdb_columns
    explainable_terms = [term for term in ncdb_columns if term in terms]
    for idx, term in enumerate(explainable_terms):
        print(term, "\n")

        for value, elaborate in ncdb_explain[term].items():
            print(value, ":", elaborate, "\n")

        if idx < len(explainable_terms) - 1:
            print("=" * 3)
            response = input("Enter 'c' to cancel, or anything else to proceed.")
            if response.lower() == "c":
                break
            clear_output()


ncdb_columns = [
    "C_YEAR",
    "C_MNTH",
    "C_WDAY",
    "C_HOUR",
    "C_SEV",
    "C_VEHS",
    "C_CONF",
    "C_RCFG",
    "C_WTHR",
    "C_RSUR",
    "C_RALN",
    "C_TRAF",
    "V_ID",
    "V_TYPE",
    "V_YEAR",
    "P_ID",
    "P_SEX",
    "P_AGE",
    "P_PSN",
    "P_ISEV",
    "P_SAFE",
    "P_USER",
    "C_CASE",
]

ncdb_explain = dict.fromkeys(ncdb_columns)

ncdb_explain["C_YEAR"] = dict.fromkeys(["19yy-20yy"])
ncdb_explain["C_MNTH"] = dict.fromkeys([m for m in range(1, 13)] + ["UU", "XX"])
ncdb_explain["C_WDAY"] = dict.fromkeys([d for d in range(1, 8)] + ["U", "X"])
ncdb_explain["C_HOUR"] = dict.fromkeys([h for h in range(24)] + ["UU", "XX"])
ncdb_explain["C_SEV"] = dict.fromkeys([1, 2] + ["U", "X"])
ncdb_explain["C_VEHS"] = dict.fromkeys(["01 - 98"] + [99] + ["UU", "XX"])
ncdb_explain["C_CONF"] = dict.fromkeys(
    [c for c in range(1, 7)]
    + [c for c in range(21, 26)]
    + [c for c in range(31, 37)]
    + [41]
    + ["QQ", "UU", "XX"]
)
ncdb_explain["C_RCFG"] = dict.fromkeys([r for r in range(1, 13)] + ["QQ", "UU", "XX"])
ncdb_explain["C_WTHR"] = dict.fromkeys([w for w in range(1, 8)] + ["Q", "U", "X"])
ncdb_explain["C_RSUR"] = dict.fromkeys([s for s in range(1, 10)] + ["Q", "U", "X"])
ncdb_explain["C_RALN"] = dict.fromkeys([l for l in range(1, 7)] + ["Q", "U", "X"])
ncdb_explain["C_TRAF"] = dict.fromkeys([t for t in range(1, 19)] + ["QQ", "UU", "XX"])
ncdb_explain["V_ID"] = dict.fromkeys(["01 - 98"] + [99] + ["UU"])
ncdb_explain["V_TYPE"] = dict.fromkeys(
    [1, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23]
    + ["QQ", "UU", "XX", "NN"]
)
ncdb_explain["V_YEAR"] = dict.fromkeys(["19yy-20yy"] + ["UUUU", "XXXX", "NNNN"])
ncdb_explain["P_ID"] = dict.fromkeys(["01 - 99"] + ["UU", "NN"])
ncdb_explain["P_SEX"] = dict.fromkeys(["M", "F"] + ["U", "X", "N"])
ncdb_explain["P_AGE"] = dict.fromkeys([0] + ["01 - 98"] + [99] + ["UU", "XX", "NN"])
ncdb_explain["P_PSN"] = dict.fromkeys(
    [p + i for p in [10, 20, 30] for i in [1, 2, 3]]
    + ["etc."]
    + [96, 97, 98, 99]
    + ["QQ", "UU", "XX", "NN"]
)
ncdb_explain["P_ISEV"] = dict.fromkeys([1, 2, 3] + ["U", "X", "N"])
ncdb_explain["P_SAFE"] = dict.fromkeys(
    [1, 2, 9, 10, 11, 12, 13] + ["QQ", "UU", "XX", "NN"]
)
ncdb_explain["P_USER"] = dict.fromkeys([1, 2, 3, 4, 5] + ["U"])
ncdb_explain["C_CASE"] = dict.fromkeys([])

ncdb_explain["C_YEAR"][
    "19yy-20yy"
] = "yy = last two digits of the calendar year. (e.g. 90, 91, 92)"

ncdb_explain["C_MNTH"][1] = "January"
ncdb_explain["C_MNTH"][2] = "February"
ncdb_explain["C_MNTH"][3] = "March"
ncdb_explain["C_MNTH"][4] = "April"
ncdb_explain["C_MNTH"][5] = "May"
ncdb_explain["C_MNTH"][6] = "June"
ncdb_explain["C_MNTH"][7] = "July"
ncdb_explain["C_MNTH"][8] = "August"
ncdb_explain["C_MNTH"][9] = "September"
ncdb_explain["C_MNTH"][10] = "October"
ncdb_explain["C_MNTH"][11] = "November"
ncdb_explain["C_MNTH"][12] = "December"
ncdb_explain["C_MNTH"]["UU"] = "Unknown"
ncdb_explain["C_MNTH"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_WDAY"][1] = "Monday"
ncdb_explain["C_WDAY"][2] = "Tuesday"
ncdb_explain["C_WDAY"][3] = "Wednesday"
ncdb_explain["C_WDAY"][4] = "Thursday"
ncdb_explain["C_WDAY"][5] = "Friday"
ncdb_explain["C_WDAY"][6] = "Saturday"
ncdb_explain["C_WDAY"][7] = "Sunday"
ncdb_explain["C_WDAY"]["U"] = "Unknown"
ncdb_explain["C_WDAY"]["X"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_HOUR"][0] = "Midnight to 0:59"
for h in range(1, 24):
    ncdb_explain["C_HOUR"][h] = f"{h}:00 to {h}:59"
ncdb_explain["C_HOUR"]["UU"] = "Unknown"
ncdb_explain["C_HOUR"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_SEV"][1] = "Collision producing at least one fatality"
ncdb_explain["C_SEV"][2] = "Collision producing non-fatal injury"
ncdb_explain["C_SEV"]["U"] = "Unknown"
ncdb_explain["C_SEV"]["X"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_VEHS"]["01 - 98"] = "01 - 98 vehicles involved"
ncdb_explain["C_VEHS"][99] = "99 or more vehicles involved"
ncdb_explain["C_VEHS"]["UU"] = "Unknown"
ncdb_explain["C_VEHS"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_CONF"][
    1
] = "Single Vehicle in Motion: Hit a moving object. E.g. a person or an animal."
ncdb_explain["C_CONF"][
    2
] = "Single Vehicle in Motion: Hit a stationary object. E.g. a tree."
ncdb_explain["C_CONF"][
    3
] = "Single Vehicle in Motion: Ran off left shoulder. Including rollover in the left ditch."
ncdb_explain["C_CONF"][
    4
] = "Single Vehicle in Motion: Ran off right shoulder. Including rollover in the right ditch."
ncdb_explain["C_CONF"][5] = "Single Vehicle in Motion: Rollover on roadway."
ncdb_explain["C_CONF"][
    6
] = "Single Vehicle in Motion: Any other single vehicle collision configuration."
ncdb_explain["C_CONF"][
    21
] = "Two Vehicles in Motion - Same Direction of Travel: Rear-end collision"
ncdb_explain["C_CONF"][
    22
] = "Two Vehicles in Motion - Same Direction of Travel: Side swipe"
ncdb_explain["C_CONF"][
    23
] = "Two Vehicles in Motion - Same Direction of Travel: One vehicle passing to the left of the other, or left turn conflict"
ncdb_explain["C_CONF"][
    24
] = "Two Vehicles in Motion - Same Direction of Travel: One vehicle passing to the right of the other, or right turn conflict"
ncdb_explain["C_CONF"][
    25
] = "Two Vehicles in Motion - Same Direction of Travel: Any other two vehicle - same direction of travel configuration"
ncdb_explain["C_CONF"][
    31
] = "Two Vehicles in Motion - Different Direction of Travel: Head-on collision"
ncdb_explain["C_CONF"][
    32
] = "Two Vehicles in Motion - Different Direction of Travel: Approaching side-swipe"
ncdb_explain["C_CONF"][
    33
] = "Two Vehicles in Motion - Different Direction of Travel: Left turn across opposing traffic"
ncdb_explain["C_CONF"][
    34
] = "Two Vehicles in Motion - Different Direction of Travel: Right turn, including turning conflicts"
ncdb_explain["C_CONF"][
    35
] = "Two Vehicles in Motion - Different Direction of Travel: Right angle collision"
ncdb_explain["C_CONF"][
    36
] = "Two Vehicles in Motion - Different Direction of Travel: Any other two-vehicle - different direction of travel configuration"
ncdb_explain["C_CONF"][41] = "Two Vehicles - Hit a Parked Motor Vehicle"
ncdb_explain["C_CONF"]["QQ"] = "Choice is other than the preceding values"
ncdb_explain["C_CONF"]["UU"] = "Unknown"
ncdb_explain["C_CONF"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_RCFG"][1] = "Non-intersection e.g. 'mid-block'"
ncdb_explain["C_RCFG"][2] = "At an intersection of at least two public roadways"
ncdb_explain["C_RCFG"][
    3
] = "Intersection with parking lot entrance/exit, private driveway or laneway"
ncdb_explain["C_RCFG"][4] = "Railroad level crossing"
ncdb_explain["C_RCFG"][5] = "Bridge, overpass, viaduct"
ncdb_explain["C_RCFG"][6] = "Tunnel or underpass"
ncdb_explain["C_RCFG"][7] = "Passing or climbing lane"
ncdb_explain["C_RCFG"][8] = "Ramp"
ncdb_explain["C_RCFG"][9] = "Traffic circle"
ncdb_explain["C_RCFG"][10] = "Express lane of a freeway system"
ncdb_explain["C_RCFG"][11] = "Collector lane of a freeway system"
ncdb_explain["C_RCFG"][12] = "Transfer lane of a freeway system"
ncdb_explain["C_RCFG"]["QQ"] = "Choice is other than the preceding values"
ncdb_explain["C_RCFG"]["UU"] = "Unknown"
ncdb_explain["C_RCFG"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_WTHR"][1] = "Clear and sunny"
ncdb_explain["C_WTHR"][2] = "Overcast, cloudy but no precipitation"
ncdb_explain["C_WTHR"][3] = "Raining"
ncdb_explain["C_WTHR"][4] = "Snowing, not including drifting snow"
ncdb_explain["C_WTHR"][5] = "Freezing rain, sleet, hail"
ncdb_explain["C_WTHR"][
    6
] = "Visibility limitation e.g. drifting snow, fog, smog, dust, smoke, mist"
ncdb_explain["C_WTHR"][7] = "Strong wind"
ncdb_explain["C_WTHR"]["Q"] = "Choice is other than the preceding values"
ncdb_explain["C_WTHR"]["U"] = "Unknown"
ncdb_explain["C_WTHR"]["X"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_RSUR"][1] = "Dry, normal"
ncdb_explain["C_RSUR"][2] = "Wet"
ncdb_explain["C_RSUR"][3] = "Snow (fresh, loose snow)"
ncdb_explain["C_RSUR"][4] = "Slush, wet snow"
ncdb_explain["C_RSUR"][5] = "Icy. Includes packed snow."
ncdb_explain["C_RSUR"][
    6
] = "Sand/gravel/dirt. Refers to the debris on the road, not the material used to construct the road."
ncdb_explain["C_RSUR"][7] = "Muddy"
ncdb_explain["C_RSUR"][8] = "Oil. Includes spilled liquid or road application."
ncdb_explain["C_RSUR"][9] = "Flooded"
ncdb_explain["C_RSUR"]["Q"] = "Choice is other than the preceding values"
ncdb_explain["C_RSUR"]["U"] = "Unknown"
ncdb_explain["C_RSUR"]["X"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_RALN"][1] = "Straight and level"
ncdb_explain["C_RALN"][2] = "Straight with gradient"
ncdb_explain["C_RALN"][3] = "Curved and level"
ncdb_explain["C_RALN"][4] = "Curved with gradient"
ncdb_explain["C_RALN"][5] = "Top of hill or gradient"
ncdb_explain["C_RALN"][6] = "Bottom of hill or gradient. 'Sag'."
ncdb_explain["C_RALN"]["Q"] = "Choice is other than the preceding values"
ncdb_explain["C_RALN"]["U"] = "Unknown"
ncdb_explain["C_RALN"]["X"] = "Jurisdiction does not provide this data element"

ncdb_explain["C_TRAF"][1] = "Traffic signals fully operational"
ncdb_explain["C_TRAF"][2] = "Traffic signals in flashing mode"
ncdb_explain["C_TRAF"][3] = "Stop sign"
ncdb_explain["C_TRAF"][4] = "Yield sign"
ncdb_explain["C_TRAF"][5] = "Warning sign. Yellow diamond shape sign."
ncdb_explain["C_TRAF"][6] = "Pedestrian crosswalk"
ncdb_explain["C_TRAF"][7] = "Police officer"
ncdb_explain["C_TRAF"][8] = "School guard, flagman"
ncdb_explain["C_TRAF"][9] = "School crossing"
ncdb_explain["C_TRAF"][10] = "Reduced speed zone"
ncdb_explain["C_TRAF"][11] = "No passing zone sign"
ncdb_explain["C_TRAF"][12] = "Markings on the road e.g. no passing"
ncdb_explain["C_TRAF"][13] = "School bus stopped with school bus signal lights flashing"
ncdb_explain["C_TRAF"][
    14
] = "School bus stopped with school bus signal lights not flashing"
ncdb_explain["C_TRAF"][15] = "Railway crossing with signals, or signals and gates"
ncdb_explain["C_TRAF"][16] = "Railway crossing with signs only"
ncdb_explain["C_TRAF"][17] = "Control device not specified"
ncdb_explain["C_TRAF"][18] = "No control present"
ncdb_explain["C_TRAF"]["QQ"] = "Choice is other than the preceding values"
ncdb_explain["C_TRAF"]["UU"] = "Unknown"
ncdb_explain["C_TRAF"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["V_ID"]["01 - 98"] = "01 - 98"
ncdb_explain["V_ID"][99] = "Vehicle sequence number assigned to pedestrians"
ncdb_explain["V_ID"][
    "UU"
] = "Unknown. In cases where a person segment cannot be correctly matched with the vehicle that he/she was riding in, the Vehicle Sequence Number is set to UU."

ncdb_explain["V_TYPE"][
    1
] = "Light Duty Vehicle (Passenger car, Passenger van, Light utility vehicles and light duty pick up trucks)"
ncdb_explain["V_TYPE"][
    5
] = "Panel/cargo van <= 4536 KG GVWR. Panel or window type of van designed primarily for carrying goods."
ncdb_explain["V_TYPE"][
    6
] = "Other trucks and vans <= 4536 KG GVWR. Unspecified, or any other types of LTVs that do not fit into the above categories (e.g. delivery or service vehicles, chip wagons, small tow trucks etc.)"
ncdb_explain["V_TYPE"][
    7
] = "Unit trucks > 4536 KG GVWR. All heavy unit trucks, with or without a trailer."
ncdb_explain["V_TYPE"][8] = "Road tractor With or without a semi-trailer"
ncdb_explain["V_TYPE"][9] = "School bus. Standard large type."
ncdb_explain["V_TYPE"][10] = "Smaller school bus. Smaller type, seats < 25 passengers."
ncdb_explain["V_TYPE"][11] = "Urban and Intercity Bus"

ncdb_explain["V_TYPE"][
    14
] = "Motorcycle and moped Motorcycle and limited-speed motorcycle"
ncdb_explain["V_TYPE"][
    16
] = "Off road vehicles Off road motorcycles (e.g. dirt bikes) and all terrain vehicles"
ncdb_explain["V_TYPE"][17] = "Bicycle"
ncdb_explain["V_TYPE"][18] = "Purpose-built motorhome. Exclude pickup campers."
ncdb_explain["V_TYPE"][19] = "Farm equipment"
ncdb_explain["V_TYPE"][20] = "Construction equipment"
ncdb_explain["V_TYPE"][21] = "Fire engine"
ncdb_explain["V_TYPE"][22] = "Snowmobile"
ncdb_explain["V_TYPE"][23] = "Streetcar"
ncdb_explain["V_TYPE"][
    "NN"
] = "Data element is not applicable. e.g. 'dummy' vehicle record created for the pedestrian."
ncdb_explain["V_TYPE"]["QQ"] = "Choice is other than the preceding values"
ncdb_explain["V_TYPE"]["UU"] = "Unknown"
ncdb_explain["V_TYPE"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["V_YEAR"][
    "19yy-20yy"
] = "Model Year 19YY to 20YY where 00 <= YY <= Current Year + 1"
ncdb_explain["V_YEAR"][
    "NNNN"
] = "Data element is not applicable. e.g. 'dummy' vehicle record created for the pedestrian."
ncdb_explain["V_YEAR"]["UUUU"] = "Unknown"
ncdb_explain["V_YEAR"]["XXXX"] = "Jurisdiction does not provide this data element"

ncdb_explain["P_ID"]["01 - 99"] = "01 - 99"
ncdb_explain["P_ID"][
    "NN"
] = "Data element is not applicable. e.g. 'dummy' person record created for parked cars."
ncdb_explain["P_ID"]["UU"] = "Unknown"

ncdb_explain["P_SEX"]["F"] = "Female"
ncdb_explain["P_SEX"]["M"] = "Male"
ncdb_explain["P_SEX"][
    "N"
] = "Data element is not applicable. e.g. 'dummy' person record created for parked cars."
ncdb_explain["P_SEX"]["U"] = "Unknown"
ncdb_explain["P_SEX"]["X"] = "Jurisdiction does not provide this data element"

ncdb_explain["P_AGE"][0] = "Less than 1 Year old"
ncdb_explain["P_AGE"]["01 - 98"] = "1 to 98 Years old"
ncdb_explain["P_AGE"][99] = "99 Years or older"
ncdb_explain["P_AGE"][
    "NN"
] = "Data element is not applicable. e.g. 'dummy' person record created for parked cars."
ncdb_explain["P_AGE"]["UU"] = "Unknown"
ncdb_explain["P_AGE"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["P_PSN"][11] = "Driver"
ncdb_explain["P_PSN"][12] = "Front row, center"
ncdb_explain["P_PSN"][
    13
] = "Front row, right outboard, including motorcycle passenger in sidecar"
ncdb_explain["P_PSN"][21] = "Second row, left outboard, including motorcycle passenger"
ncdb_explain["P_PSN"][22] = "Second row, center"
ncdb_explain["P_PSN"][23] = "Second row, right outboard"
ncdb_explain["P_PSN"][31] = "Third row, left outboard"
ncdb_explain["P_PSN"][32] = "Third row, center"
ncdb_explain["P_PSN"][33] = "Third row, right outboard"
ncdb_explain["P_PSN"]["etc."] = " "
ncdb_explain["P_PSN"][
    96
] = "Position unknown, but the person was definitely an occupant"
ncdb_explain["P_PSN"][97] = "Sitting on someone\'s lap"
ncdb_explain["P_PSN"][
    98
] = "Outside passenger compartment. e.g. riding in the back of a pick-up truck."
ncdb_explain["P_PSN"][99] = "Pedestrian"
ncdb_explain["P_PSN"][
    "NN"
] = "Data element is not applicable. e.g. 'dummy' person record created for parked cars."
ncdb_explain["P_PSN"]["QQ"] = "Choice is other than the preceding value"
ncdb_explain["P_PSN"]["UU"] = "Unknown. e.g. applies to runaway cars."
ncdb_explain["P_PSN"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["P_ISEV"][1] = "No Injury"
ncdb_explain["P_ISEV"][2] = "Injury"
ncdb_explain["P_ISEV"][3] = "Fatality. Died immediately or within the time limit."
ncdb_explain["P_ISEV"][
    "N"
] = "Data element is not applicable. e.g. 'dummy' person record created for parked cars."
ncdb_explain["P_ISEV"]["U"] = "Unknown. e.g. applies to runaway cars."
ncdb_explain["P_ISEV"]["X"] = "Jurisdiction does not provide this data element"

ncdb_explain["P_SAFE"][1] = "No safety device used, or No child restraint used"
ncdb_explain["P_SAFE"][2] = "Safety device used or child restraint used"
ncdb_explain["P_SAFE"][
    9
] = "Helmet worn. For motorcyclists, bicyclists, snowmobilers, all-terrain vehicle riders."
ncdb_explain["P_SAFE"][
    10
] = "Reflective clothing worn. For motorcyclists, bicyclists, snowmobilers, all-terrain vehicle riders."
ncdb_explain["P_SAFE"][
    11
] = "Both helmet and reflective clothing used. For motorcyclists, bicyclists, snowmobilers, all-terrain vehicle riders."
ncdb_explain["P_SAFE"][12] = "Other safety device used"
ncdb_explain["P_SAFE"][13] = "No safety device equipped. e.g. buses."
ncdb_explain["P_SAFE"][
    "NN"
] = "Data element is not applicable. e.g. 'dummy' person record created for parked cars."
ncdb_explain["P_SAFE"]["QQ"] = "Choice is other than the preceding value"
ncdb_explain["P_SAFE"]["UU"] = "Unknown. e.g. applies to runaway cars."
ncdb_explain["P_SAFE"]["XX"] = "Jurisdiction does not provide this data element"

ncdb_explain["P_USER"][1] = "Motor Vehicle Driver"
ncdb_explain["P_USER"][2] = "Motor Vehicle Passenger"
ncdb_explain["P_USER"][3] = "Pedestrian"
ncdb_explain["P_USER"][4] = "Bicyclist"
ncdb_explain["P_USER"][5] = "Motorcyclist"
ncdb_explain["P_USER"]["U"] = "Not stated / Other / Unknown"