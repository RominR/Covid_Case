#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:13:42 2022

@author: rominrajbhandari
"""

#Importing all the required libraries.

#Standard Libraries
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from numpy.random import randn

#Statistics Libraries
from scipy import stats

#Plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

#Setting directry (if needed)
import os
os.getcwd()
os.chdir('/Users/romin/Library/Mobile Documents/com~apple~CloudDocs/Python/Python_Bootcamp_10-7-2020/Datasets')

#Datetime
import datetime

#Regular Expression
import re

#SimpleImputer
from sklearn.impute import SimpleImputer

#KNNImputer
from sklearn.impute import KNNImputer

#Iterative Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Label Encoding or One Hot Encoding
from sklearn.preprocessing import LabelEncoder

#Grab the data from web (HTTP capabilities)
import requests
import urllib.request #creates a Request object specifying the URL we want.

# We'll also use StringIO to work with the csv file, the DataFrame will require a .read() method. StringIO provides a convenient means of working with text in memory using the file API
from io import StringIO
import io

#if the file to be imported in on zipfile
import zipfile

import xarray as xr # opens up the bytes file as a xarray dataset.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Now, let us read the dataset directly from the web.

'''Key Indicators of Heart Disease
2020 annual CDC survey data of 400k adults related to their health status'''
#url = https://healthdata.gov/dataset/Heart-Disease-Mortality-Data-Among-US-Adults-35-by/pwn5-iqp5

url = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-2020-12-14.xlsx"

#The dataset is in xlsx format. Therefore,

#Using 'requests' to get the information.
source = requests.get(url)
open('covidcase.xlsx', 'wb').write(source.content)

#Now that we have our data, we can set it as a DataFrame.
Covid_Case = pd.read_excel('covidcase.xlsx')
Covid_Case.info() #Checking the basic information of each columns
Covid_Case.head(10) #Viewing the first 10 rows of the dataset.

#Renaming the columns because '-' is not compatible with pandas.
Covid_Case.rename(columns = {'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000': 'Cumulative_number_for_14_days_of_COVID19_cases_per_100000'}, inplace = True)


#def data_check():
#    for col in Covid_Case.columns:
#        if (Covid_Case[col].dtypes == 'object'):
#            if (Covid_Case[(Covid_Case[col] == '') | (Covid_Case[col].str.findall("[^a-zA-Z0-9 ]+"))][[col]]).empty:
#                print(f"Column '{col}' is empty.\n\n")
#            elif (len(Covid_Case[(Covid_Case[col] == '') | (Covid_Case[col].str.findall("[^a-zA-Z0-9 ]+"))][[col]]) ==
#                  len(Covid_Case[(Covid_Case[col] == '') | (Covid_Case[col].str.findall("[\:\.\(\)\-\&\'\,\/\{\}\~\`\|\!\@\#\$\%\^\*\_\=\+\;\"\'\<\>\?]+"))][[col]])):
#                print(f'''Column '{col}' is not empty.\nBut the length of both the columns matches.
#                      Now, let's check if there is just a special character in any of the rows.\n\n"''')
#                for spec_char in list(":.()-&',/{}~`|!@#$%^*_=+;'<>?"):
#                    if not (Covid_Case[Covid_Case[col] == spec_char][col]).empty:
#                        print(f"There is just a '{spec_char}' character in one of the rows. See the result below:")
#                        print(Covid_Case[Covid_Case[col] == spec_char][col],"\n\n")
#                    else:
#                        print(f"There is no just '{spec_char}' character in any row. See the empty series below:")
#                        print(Covid_Case[Covid_Case[col] == spec_char][col],"\n\n")
#            else:
#                print(f'''The length of {col} columns don't match. There may be a different special character than
#                      what is provided in the 'elif' above.\n\n''')
#              
#data_check()




#Checking to see if the column 'countriesAnd Territoris' has any special characters.
all(c.isalnum() for c in Covid_Case.countriesAndTerritories)

#Let us find out what sorts of special characters are present in the column.
Covid_Case.countriesAndTerritories.unique()

'''
#The code is correct but this code was not useful for removing all the special characters if the names(countriesAndTerritories) has multiple different special characters.
It didn't work for the values such as 'Falkland_Islands_(Malvinas)' because once it removes '_' sign under for loop using "names.replace(char, '')",
it moves to another names i.e. 'Faroe Islands'.
def replace_char(names):
    if names.isalnum():
        return names
    else:
        for char in list('_(),Ã§ '):
            if char in names:
                return names.replace(char, '')
            else:
                continue
'''            
'''
#Removes all the the special characters from 'countriesAndTerritories' column.
def replace_char(names):
    if names.isalnum():
        return names
    else:
        return re.sub(pattern = '[^a-zA-Z0-9]+', repl = '', string = names)
'''
#It looks like there are unnecessary special characters such as '_ in the columns. Also, there are other special characters like ',', '()' which are required for the values. Therefore let us remove only '_' character.
def replace_char(names):
    if names.isalnum():
        return names
    else:
        return re.sub(pattern = '\_', repl = ' ', string = names)

#Calling the user defined function i.e. replace_char using apply().
Covid_Case.countriesAndTerritories =  Covid_Case.countriesAndTerritories.apply(replace_char)

#Now, checking the following code again.
all(c.isalnum() for c in Covid_Case.countriesAndTerritories)
#We would get True result only if all the special characters were removed. But, here we have just removed '_'. I have tried removing all the special characters(code in the comment section). And it worked.

'''
import re
my_list= ["on@3", "two#", "thre%e"]
[re.sub(pattern = '[^a-zA-Z0-9]+', repl = '', string = i) for i in my_list]


dframe = DataFrame(data=np.arange(1, 26, 1).reshape((5, 5)), index=[
                   'NYC', 'LA', 'SF', 'DC', 'Chi'], columns=['A', 'B', 'C', 'D', 'E'])
dframe.A.replace({1: 'one@', 6: 'si_x', 11: 'ele,ven', 16: 'six tee@@n', 21: 'tw, enty(one)'}, inplace = True)
dframe

def rc(names):
    if names.isalnum():
        return names
    else:
        return re.sub(pattern = '[^a-zA-Z0-9]+', repl = '', string = names)
        
dframe.A = dframe.A.apply(rc)
dframe'''    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
#There are 275 null values in geoId column
Covid_Case[Covid_Case.geoId.isnull()][['countriesAndTerritories', 'geoId']]

#Let us see what the geoId column looks like. 
Covid_Case[Covid_Case.geoId.notnull()][['countriesAndTerritories', 'geoId']]

#Ahh! It is just the country code. Let's see if it is just Namibia which is missing the geoId.
Covid_Case[Covid_Case.countriesAndTerritories == 'Namibia']

#Since the number of rows for above two codes match, it is Namibia whose geoId is missing. Now, let us give Namibia its geoId. ISO Code for Namibia is 'NA'.
#First, let us make sure there are not any 'NA' ISO code in the column.
Covid_Case[Covid_Case.geoId == 'NA'][['countriesAndTerritories', 'geoId']]

#Looks like there are not any 'NA' given for other countries. Hence, we can now provide the ID for Namibia.
Covid_Case.geoId.replace({np.nan : 'NA'}, inplace = True)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#There are 123 null values in 'countryterritoryCode' column
Covid_Case[Covid_Case.countryterritoryCode.isnull()][['countryterritoryCode', 'geoId', 'countriesAndTerritories']]
#Let us see the unique values 
Covid_Case[Covid_Case.countryterritoryCode.isnull()]['countriesAndTerritories'].unique() #There are 59 'Wallis and Futuna' and 64 'Cases on an international conveyance Japan'.

#Let us see what the countryterritoryCode column looks like. 
Covid_Case[Covid_Case.countryterritoryCode.notnull()][['countryterritoryCode', 'geoId', 'countriesAndTerritories']]

#Based on google, we can provide the countryterritory code to Wallis and Futuna as WLF.
#Covid_Case[Covid_Case.countriesAndTerritories == 'Wallis and Futuna'][['countryterritoryCode', 'geoId', 'countriesAndTerritories']]
Covid_Case.loc[Covid_Case.countriesAndTerritories == 'Wallis and Futuna', 'countryterritoryCode'] = 'WLF'

#we can use where function, in this types of condition if there are only two categories to be modified. For e.g. Gender: Male and Female.

#Now, since there is not any country territory code for "International Conveyance Japan" since it is a cruise ship, we can just say JPG, taking the first 3 characters from column geoId.
#Covid_Case[Covid_Case.countriesAndTerritories == 'Cases on an international conveyance Japan'][['countryterritoryCode', 'geoId', 'countriesAndTerritories']]
Covid_Case.loc[Covid_Case.countriesAndTerritories == "Cases on an international conveyance Japan", 'countryterritoryCode'] = 'JPG'

#Let us now check if there are any null values present in this condition.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Now, let us work on another column i.e. PopData2019 which is missing 123 values.
Covid_Case[Covid_Case.popData2019.isnull()][['geoId', 'popData2019']] #It looks like 'Wallis and Fortuna' and 'Cases on an international conveyance Japan' are missing the data.
#Let us check the unique values to make sure 'Wallis and Fortuna' and 'Cases on an international conveyance Japan' are missing the data.
Covid_Case[Covid_Case.popData2019.isnull()]['geoId'].unique() #Yes, it was just 'WF' and 'JPG11668'.

#In order to provide the population data for 'WLF' and 'JPG11668', we have to first make sure the population for each country for each date is same.
#Let us see if the population for each date is the same.
'''
for i in Covid_Case.geoId:
    if i == 'WF' or i == 'JPG11668':
        continue
    elif len(Covid_Case[Covid_Case.geoId == i]['popData2019'].unique()) == 1:
        print(True)
    else:
        print(False)
'''

all(len(Covid_Case[Covid_Case.geoId == i]['popData2019'].unique())  == 1 for i in Covid_Case[(Covid_Case.geoId != 'WF') & (Covid_Case.geoId != 'JPG11668')].geoId)
'''In the above script, we filtered out the data of 'WF' and 'JPG11668' so that, when we obtain the length of unique value of popData2019 column,
we just get the length from the rest of the countries.
Here, we try to make sure that the length of each unique popData2019 is 1. If any of these data comes out to be False, then 'all()' will return false.'''

#Since the result from above script is True, it appears that the population for all the date is same i.e. population of 2019 is provided in each date for the respective countries.
Covid_Case[Covid_Case.geoId == 'NP']['popData2019'].unique() #Confirming from google that the population in the dataset matches.

#Now, let us provide the population for 'WLF' and 'JPG11668'
#According to google, the population for Wallis and Futuna in 2019 is 11,432 and 4061 people in Diamond Princess Cruise Ship.
#url = https://www.google.com/search?q=total+population+in+Wallis+and+Futuna+in+2019&oq=total+population+in+Wallis+and+Futuna+in+2019&aqs=chrome.0.69i59.513j0j9&sourceid=chrome&ie=UTF-8
#url = https://www.statista.com/statistics/1099517/japan-coronavirus-patients-diamond-princess/

#Let us give them these two values.
Covid_Case.loc[(Covid_Case.geoId == 'WF'), 'popData2019'] = 11432
Covid_Case.loc[Covid_Case.geoId == 'JPG11668', 'popData2019'] = 4061

#Checking the info.
Covid_Case.info()
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Now, let us work on another column i.e. Cumulative_number_for_14_days_of_COVID19_cases_per_100000 which is missing 2879 values.

Covid_Case[Covid_Case.geoId == 'NP']['Cumulative_number_for_14_days_of_COVID19_cases_per_100000'].tail(13)
#We came to find out that the first 13 days for each country has nan values. Let us see if the first 13 days for each country have nan values.
grpby_all_geoId = Covid_Case[Covid_Case['Cumulative_number_for_14_days_of_COVID19_cases_per_100000'].isnull()].groupby('geoId').count() #count()function does not count NaN values, therefore we see 0 in the last column counts.
#grpby.iloc[0]
grpby_all_geoId[grpby_all_geoId.dateRep != 13] #It looks like except for 'JPG11668' and 'WF' all other countries have nan values in the first 13 days.

#Let us ignore 'WF' and 'JPG11668' and see if all the countries have nan values in their first 13 days.
no_WF_JPG = Covid_Case[(Covid_Case.geoId != 'WF') & (Covid_Case.geoId != 'JPG11668')].copy()
len(no_WF_JPG['geoId'].unique()) #length is 212
len(no_WF_JPG[no_WF_JPG['Cumulative_number_for_14_days_of_COVID19_cases_per_100000'].isnull()]) #length is 2756
#If we divide 2756 by 212, we get 13, which proves that all the countries have nan values in their first 13 days.

#In order to prove it further let us do the following.
all(i == 13 for i in grpby_all_geoId.day) #Results False becuase it includes 'WF' and 'JPG11668' whose counts are 59 and 64 resp.

grpby_without_WF_JPG = no_WF_JPG[no_WF_JPG['Cumulative_number_for_14_days_of_COVID19_cases_per_100000'].isnull()].groupby('geoId').count() #count()function does not count NaN values, therefore we see 0 in the last column counts.
all(i == 13 for i in grpby_without_WF_JPG.day) #Results True becuase it doesn't include 'WF' and 'JPG11668'

#Now, since we want to remove NaN values from the column 'Cumulative...100000', we will fill the value as 0 except for  'WF' and 'JPG1668'.

#Covid_Case[(Covid_Case.geoId != 'WF') & (Covid_Case.geoId != 'JPG11668')]['Cumulative_number_for_14_days_of_COVID19_cases_per_100000'].fillna(0, inplace = True)
'''The above code doesn't work. It throws the following error:
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame
See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
return self._update_inplace(result)
'''
#Read the article below to learn more about Setting with copy warning:
#URL = https://www.dataquest.io/blog/settingwithcopywarning/

#no_WF_JPG.info()
#no_WF_JPG.Cumulative_number_for_14_days_of_COVID19_cases_per_100000.fillna(0, inplace = True)
#The above code works even though there is hidden chaining because we have applied .copy() i.e. we have explicitly tell pandas to make a copy when we create the new dataframe

#Therefore, using the following one:

#Covid_Case['Cumulative_number_for_14_days_of_COVID19_cases_per_100000'] = Covid_Case.Cumulative_number_for_14_days_of_COVID19_cases_per_100000.fillna(0)
#The above code works, but we are trying to exclude 'WF' and 'JPG11668' from the dataset.

Covid_Case.loc[Covid_Case[(Covid_Case.geoId != 'WF') & (Covid_Case.geoId != 'JPG11668') & (Covid_Case.Cumulative_number_for_14_days_of_COVID19_cases_per_100000.isnull())],
'Cumulative_number_for_14_days_of_COVID19_cases_per_100000']  = 0
#The above codes gives ValueError: Cannot index with multidimensional key. It is because, we are creating another dataframe inside .loc[] i.e. Covid_Case[...]

Covid_Case.loc[((Covid_Case.geoId != 'WF') & (Covid_Case.geoId != 'JPG11668') & (Covid_Case.Cumulative_number_for_14_days_of_COVID19_cases_per_100000.isnull())),
'Cumulative_number_for_14_days_of_COVID19_cases_per_100000'] = 0

#Let us chekc if only 'JPG11668' and 'WF' has null values.
Covid_Case[Covid_Case.Cumulative_number_for_14_days_of_COVID19_cases_per_100000.isnull()]['geoId'].unique()
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#It looks like there is no way we can just leave the values for 'WF' and 'JPG11668' as null values, therefore, let us give them 0 too.
Covid_Case.Cumulative_number_for_14_days_of_COVID19_cases_per_100000.fillna(0, inplace = True)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Now, let us work on visualization.

#pull test

















