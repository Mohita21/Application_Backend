import Untitled as un# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Press ⌘F8 to toggle the breakpoint.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    s = un.final_frame()
    un.Generating_dates_and_columns(s)
    un.Mois_and_Temp_Imputation("SOIL_MOISTURE_5_DAILY", s)
    un.Mois_and_Temp_Imputation("SOIL_MOISTURE_10_DAILY", s)
    un.Mois_and_Temp_Imputation("SOIL_MOISTURE_20_DAILY", s)
    un.Mois_and_Temp_Imputation("SOIL_MOISTURE_50_DAILY", s)
    un.Mois_and_Temp_Imputation("SOIL_MOISTURE_100_DAILY", s)
    un.Mois_and_Temp_Imputation("SOIL_TEMP_5_DAILY", s)
    un.Mois_and_Temp_Imputation("SOIL_TEMP_10_DAILY", s)
    un.Mois_and_Temp_Imputation("SOIL_TEMP_20_DAILY", s)
    un.Mois_and_Temp_Imputation("SOIL_TEMP_50_DAILY", s)
    un.Mois_and_Temp_Imputation("SOIL_TEMP_100_DAILY", s)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
