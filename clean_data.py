from __future__ import print_function # Python 3 print function easier to use

"""
Created on 2017-07-26

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Clean_data.py takes a CSV of data from an ACS-1 database. 
Cleans the data and stores it in a Pandas dataframe.

"""


''' Imports '''

import pandas as pd # data manipulation, version 0.20.2
import numpy  as np # matrix and math tools, version 1.13.1
import sys # get arguments from command line
import pickle # store cleaned data frame


''' Column names '''


# Pick names of columns to import
# Shock through death are the complications

column_names = ["age", "femal", "weight", "ethnicity",
                "hypothyroid", "diabetes", "hypertension", "pepticulcerdisease",
                "peripheralarterialdisease", "renaldisease", "liverdisease", "hyperlipidemia", 
                "priorpci", "priorrevasc", "priorcabg", "priorcvatia",
                "prioraf", "bbonadmission", "bbatadmission", "anychronotropic", 
                "dxondischarge", "stemi", "mi",
                "INHOSPCABG", "asa", "dapt", "optimalmed", "bb", "stat", 
                "ACE_inhibitor", "angiotensin_receptor_blocker", 
                "psychotropic",
                "cath", "cathacesssite", "locationofstenoses", "typeofstent", 
                "angioplasty", "numberofstents", "intraaorticballoon",
                "hoursbeforecoronaryangio", "thienopyridine", "thrombolytic", "gp2b3ainhibitor",
                "inhospitalunfracheparin", "inhospitalLMWheparin","inhospenox",
                "baselinecreat", "peakcreat", "clearanceofcreatinine", 
                "baselinehb", "nadirhemoglobin", "platelets", 
                "lowdensitychol", "hdl", "typetrop", "peaktrop", 
                "initialglycemia", "peakglycemia", "hbglycosylated",
                "lvef",
                "vascularcomplicationpostangio", 
                
                "shock", "stroke", "mechanicalventilation", "chf", 
                "cardiogenicshock", "ventriculararrythmia", "atrialfibrillation", 
                "bradyarrythmia", "arrrythmia", "cardiacarrest", "timibleed", 
                "gibleed", "infection", "death"]
                


column_types = {"femal": "category", "ethnicity": "category",
                "hypothyroid": "category", "diabetes": "category", "hypertension": "category", "pepticulcerdisease": "category",
                "peripheralarterialdisease": "category", "renaldisease": "category", "liverdisease": "category", "hyperlipidemia": "category", 
                "priorpci": "category", "priorrevasc": "category", "priorcabg": "category", "priorcvatia": "category",
                "prioraf": "category", "bbonadmission": "category", "bbatadmission": "category", "anychronotropic": "category", 
                "dxondischarge": "category", "stemi": "category", "mi": "category",
                "INHOSPCABG": "category", "asa": "category", "dapt": "category", "optimalmed": "category", "bb": "category", "stat": "category", 
                "ACE_inhibitor": "category", "angiotensin_receptor_blocker": "category", 
                "psychotropic": "category",
                "cath": "category", "cathacesssite": "category", "locationofstenoses": "category", "typeofstent": "category", 
                "angioplasty": "category", "intraaorticballoon": "category",
                "thienopyridine": "category", "thrombolytic": "category", "gp2b3ainhibitor": "category",
                "inhospitalunfracheparin": "category", "inhospitalLMWheparin": "category","inhospenox": "category",
                "typetrop": "category", 
                "vascularcomplicationpostangio": "category",
                
                "shock":"category", "stroke":"category", "mechanicalventilation":"category", "chf":"category", 
                "cardiogenicshock":"category", "ventriculararrythmia":"category", "atrialfibrillation":"category", 
                "bradyarrythmia":"category", "arrrythmia":"category", "cardiacarrest":"category", "timibleed":"category", 
                "gibleed":"category", "infection":"category", "death":"category"}

                
# If there are incorrectly coded floats as strings, replace them with NaN
def float_error(x):
    try: return float(x)
    except ValueError: return np.nan
    
float_converters = {"age": float_error, "weight": float_error, 
                    "hoursbeforecoronaryangio": float_error, "numberofstents": float_error,
                    "baselinecreat": float_error, "peakcreat": float_error, "clearanceofcreatinine": float_error,
                    "baselinehb": float_error, "nadirhemoglobin": float_error, "platelets": float_error, 
                    "lowdensitychol": float_error, "hdl": float_error, "peaktrop": float_error, 
                    "initialglycemia": float_error, "peakglycemia": float_error, "hbglycosylated": float_error,
                    "lvef": float_error}
                
''' Read data '''


# Obtain file name from command line
try:
    file_name = sys.argv[1]
except IndexError:
    sys.exit("Please enter a file name to clean.\nFormat: python clean_data.py file_name_here")
    
print("Reading " + file_name)


# Convert file to data frame
# usecols to select only certain columns (reduces memory footprint)
# low_memory = False to prevent type errors
# error_bad_lines = False to skip invalid lines
# na_values to include other representations of missing data
data_frame = pd.read_csv(file_name, 
                         header          = 0,
                         usecols         = column_names,
                         dtype           = column_types,
                         low_memory      = False,
                         error_bad_lines = False,
                         converters      = float_converters,
                         na_values       = [""," ","ND", "UNKNOWN"])
                         
                         
''' Process data '''


# Rename some columns
print("Renaming some columns")
data_frame.rename(columns = {"dxondischarge": "ACS_type",
                             "cathacesssite": "cath_entry",
                             "locationofstenoses" :"coronaries_affected",
                            }, inplace = True)



print("Changing some encodings")
# Convert 1/0 encoding from str to int
data_frame.replace(to_replace = "1", value = 1, inplace = True)
data_frame.replace(to_replace = "0", value = 0, inplace = True)

# Convert YES/NO encoding to 1/0
data_frame.replace(to_replace = "YES", value = 1, inplace = True)
data_frame.replace(to_replace = "NO",  value = 0, inplace = True)
data_frame.replace(to_replace = "yes", value = 1, inplace = True)
data_frame.replace(to_replace = "no",  value = 0, inplace = True)


print("Removing mis-entered weights and ages")
# Nobody should weigh less than 10 kg, and there are many weights in the 0-2 kg range
data_frame.loc[data_frame["weight"] < 10] = np.nan
# Remove the many <1 and negative age individuals
data_frame.loc[data_frame["age"] < 1] = np.nan



print("Fixing mis-entered values for some medical history")

# Clean high, meaningless values
### Many values that should be coded as "1" are instead a very high number
### np.where() takes a condition, and returns a vector of 1 if true, 0 if false
data_frame["bbonadmission"]       = np.where(data_frame["bbonadmission"]       >= 1, 1, 0)
data_frame["priorrevasc"]         = np.where(data_frame["priorrevasc"]         >= 1, 1, 0)
data_frame["priorcabg"]           = np.where(data_frame["priorcabg"]           >= 1, 1, 0)

# Store the data as categories
for col in ["bbonadmission", "priorrevasc", "priorcabg"]:
    data_frame[col] = data_frame[col].astype("category")



# Remove numerical (mis-entered) data from ethnicity, P12Y_inhibitor 1 and 2
# Replace numerical data with np.nan, otherwise keep as it was
def is_numeric(s):
    try: float(s)
    except TypeError: return False
    else: return True
    
data_frame["ACS_type"]  = np.where(is_numeric(data_frame["ACS_type"]),  np.nan, data_frame["ACS_type"])
data_frame["ethnicity"] = np.where(is_numeric(data_frame["ethnicity"]), np.nan, data_frame["ethnicity"])



# Encode ACS type as categorical variable
print("Encoding ACS type as a categorical variable")
def ACS_data(string):
    # Returns boolean list of ACS type given a string   
    # ACS will be returned in order: UA, NSTEMI, STEMI
    # (Unstable angina, non-ST-elevation myocardial infarction, ST-elevation myocardial infarction)
    ACS_types = [np.nan, np.nan, np.nan]
    ACS_encodings = ["UNSTABLE ANGINA", "NSTEMI", "STEMI"]
    
    # Temporarily convert function input to string
    # This is needed since some data points are incorrectly numbers, which throws an error
    string = str(string)

    # Check if there is ACS data present
    # If not, return list of NaN
    data_present = False
    for ACS in (ACS_encodings + ["NON-STEMI"]):
        if ACS in string: 
            data_present = True
            break
        else: data_present = False
    if data_present == False: return ACS_types
    
    # Encode the ACS types
    for i in range(len(ACS_types)):
    
        if ACS_encodings[i] == string: 
            ACS_types[i] = 1
        # Some NSTEMI is encoded as "NON-STEMI"
        elif (ACS_encodings[i] == "NSTEMI") and ("NON-STEMI" == string):
            ACS_types[i] = 1
        else: ACS_types[i] = 0
        
    return ACS_types
 
# ACS_series becomes a matrix of ACS type 
ACS_series = data_frame["ACS_type"].apply(ACS_data) 

# Convert the matrix into data_frame columns
data_frame["unstable_angina"] = [row[0] for row in ACS_series]
data_frame["NSTEMI"]          = [row[1] for row in ACS_series]
data_frame["STEMI"]           = [row[2] for row in ACS_series]


    
# Encode coronaries affected
print("Encoding affected coronaries as a categorical variable")
def coronary_data(string):
    # Returns boolean list of affected coronaries given a string
    
    # Coronaries will be returned in order: LAD, LCX, RCA, Other
    coronaries = [np.nan, np.nan, np.nan, np.nan]
    coronary_encodings = ["LEFT", "CIRCUMFLEX", "RIGHT", "OTHER"]
    
    # Temporarily convert function input to string
    # This is needed since some data points are incorrectly numbers, which throws an error
    string = str(string)

    # Check if there is coronary data present
    # If not, return list of NaN
    data_present = False
    for coronary in coronary_encodings:
        if coronary in string: 
            data_present = True
            break
        else: return tuple(coronaries)
        
    # Encode the coronaries
    for i in range(len(coronary_encodings)):
        if coronary_encodings[i] in string: coronaries[i] = 1
        else: coronaries[i] = 0
        
    return tuple(coronaries)

coronary_series = []    
for i in range(len(data_frame["coronaries_affected"])):
    coronary_series.append(coronary_data(data_frame["coronaries_affected"].iloc[i]))
    
# coronary_series becomes a matrix of affected coronaries
#coronary_series = data_frame["coronaries_affected"].apply(coronary_data)

# Convert the matrix into data_frame columns
data_frame["left_main"]       = [row[0] for row in coronary_series]
data_frame["left_circumflex"] = [row[1] for row in coronary_series]
data_frame["right_coronary"]  = [row[2] for row in coronary_series]
data_frame["other_coronary"]  = [row[3] for row in coronary_series]



# Encode type of stent (bare-metal, drug-eluting, or both)
print("Encoding stents as a categorical variable")
def stent_data(string):
    # Returns boolean tuple of stent type given a string
    # Stent data will be returned in order: bare-metal, drug-eluting
    
    # Temporarily convert function input to string
    # This is needed since some data points are incorrectly numbers, which throws an error
    string = str(string)

    # Encode the case with "BOTH":
    if   string == "BOTH":               return (1,1)
    elif string == "BARE METAL STENT":   return (1,0)
    elif string == "DRUG ELUTING STENT": return (0,1)
    else: return (np.nan, np.nan)

# stent_series becomes a matrix of affected coronaries
stent_series = []
for i in range(len(data_frame["typeofstent"])):
    stent_series.append(stent_data(data_frame["typeofstent"].iloc[i]))

# Convert the matrix into data_frame columns
data_frame["bare_metal_stent"]   = [row[0] for row in stent_series]
data_frame["drug_eluting_stent"] = [row[1] for row in stent_series] 


# Encode the two different types of troponins in two different columns
print("Separating measurements of troponin I and C")


data_frame["TnI"] = np.where((data_frame["peaktrop"] != np.nan) & (data_frame["typetrop"] == "I"), data_frame["peaktrop"], np.nan)
data_frame["TnT"] = np.where((data_frame["peaktrop"] != np.nan) & (data_frame["typetrop"] == "T"), data_frame["peaktrop"], np.nan)

    

# Add a column for any complications
print("Adding a column for any complication")
data_frame["any_complication"] = data_frame[["shock", "stroke", "mechanicalventilation", "chf", 
                                             "cardiogenicshock", "ventriculararrythmia", "atrialfibrillation", 
                                             "bradyarrythmia", "arrrythmia", "cardiacarrest", "timibleed", 
                                             "gibleed", "infection", "death"]].any(axis = 1, skipna = True)
                                             
data_frame["any_complication"].replace(to_replace = [True, False], value = [1,0], inplace = True)



# Get dummy variables for other string-based columns
print("One-hot encoding remaining string variables")
data_frame = pd.concat([data_frame, pd.get_dummies(data_frame["ethnicity"])], axis = 1)
data_frame = pd.concat([data_frame, pd.get_dummies(data_frame["cath_entry"])],axis = 1)

print("Dropping unnecessary rows and columns")
# Drop string variables that have been encoded other ways
data_frame = data_frame.drop(["ACS_type", "coronaries_affected", 
                              "ethnicity", "cath_entry", "typeofstent",
                              "typetrop", "peaktrop", "stemi"], axis = 1)                            
                              


''' Verify the final data frame before writing to file '''


print("Done.")
view_data = raw_input("See cleaned data?\nEnter Y for yes or anything else (or Enter) for no.")

if view_data == "Y": print(data_frame)
#data_frame.to_csv("test.csv")



''' Write cleaned data to file '''


# Remove original file extension, then add _cleaned
with open(file_name[:-4] + "_cleaned.pickle","wb") as f:
    pickle.dump(data_frame, f)
data_frame.to_csv(file_name[:-4] + "_cleaned.csv")
print('Wrote pickled data to "' + file_name[:-4] + '_cleaned.pickle".')
print('Wrote CSV data to "'     + file_name[:-4] + '_cleaned.csv".')
