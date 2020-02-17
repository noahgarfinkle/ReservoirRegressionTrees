# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:13:02 2015

@author: userx
"""

"""
Get either pydot or graphviz
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
import pydot
import os

plt.ioff() # turn off interactive plotting


path = "C:\\RBMProject\\Final Presentation"
# courtesy of http://stackoverflow.com/questions/141291/how-to-list-only-top-level-directories-in-python
reservoirs = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
print reservoirs


    
def test_errors(treeName,tree,merged):
    errors = []
    ar = np.array(merged)
    for i in range(0,len(ar)):
        r = ar[i]
        prediction = tree.predict([r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13]])
        actual = r[14]
        error = actual - prediction
        if actual == 0.0:
            actual += 0.00001
        error /= actual
        errors.append(error)
    errorsArray = np.array(errors)
    print("Min error: " + str(np.min(abs(errorsArray))))
    print("Max error: " + str(np.max(abs(errorsArray))))
    print("Median error: " + str(np.median(errorsArray)))
    plt.figure()
    plt.suptitle("Histogram of Error: " + treeName +"\nMinError: " + str(round(np.min(abs(errorsArray)),4)) + ", Max error: " + str(round(np.max(abs(errorsArray)),4)) + ", Median error: " + str(round(np.median(errorsArray),4)))
    plt.hist(errorsArray,bins=100)
    fileName = treeName + "\\" + treeName + "ErrorHistogram.png"
    plt.savefig(fileName)
    return errors

def autoDecisionTree(df,xcols,ycol,directory,fileName,maxDepth=4,minSamplesLeaf=5):
    dfx = df[xcols] 
#    print dfx.columns
    xmat = np.array(dfx)
    yvec = np.array(df[ycol])
    
    # create the merged df to send back
    merged = dfx
    merged[ycol] = df[ycol]
    
    clf = DecisionTreeRegressor(max_depth=maxDepth,min_samples_leaf=minSamplesLeaf)
    clf.fit(xmat,yvec)
    dotPath = directory + "\\" + fileName + "_tree.dot"
    tree.export_graphviz(clf,out_file=dotPath)
    
    # edit tree.dot to replace variable names
    f = open(dotPath,'r')
    toEdit = f.read()
    f.close()
    
#    print(xcols)
    for i,col in enumerate(xcols):
        toReplace = "X[" + str(i) + "]"
#        print(toReplace + ": " + col)
        toEdit = toEdit.replace(toReplace,col)
#    input("Paused")
        
    toEdit = toEdit.replace("value",ycol)
    
    w = open(dotPath,"w")
    w.write(toEdit)
    w.close()
    
    graph = pydot.graph_from_dot_file(dotPath)
    treePath = directory + "\\" + fileName + "_tree.png"
    graph.write_png(treePath)
    
#    graphvars = xcols
#    graphvars.append(ycol)
#    for col in graphvars:
#        plt.figure()
#        plt.suptitle(col + ": " + fileName)
#        plt.ylabel(col)
#        _ = plt.plot(df[col])
#        plt.savefig(directory + "\\" + fileName + "_" + col + ".png")
#        plt.close("all")
        
    df.to_csv(directory + "\\" + fileName + "_dataframe.csv",index=False)

    return merged
    
    
def processReservoir(reservoirPath,reservoirParameters):
    print reservoirPath
    inflow = pd.read_csv(reservoirPath + "\\Inflow_daily.txt")
    outflow = pd.read_csv(reservoirPath + "\\Outflow_daily.txt")
    storage = pd.read_csv(reservoirPath + "\\Storage_daily.txt")

    inflow = inflow.drop("PST",1)
    outflow = outflow.drop("PST",1)
    storage = storage.drop("PST",1)
    

    # create a monthly dataset
    monthlyInflow = inflow
    monthlyInflow[monthlyInflow < 0] = np.nan
    monthlyInflow = monthlyInflow.replace('m',np.nan)
    monthlyInflow = monthlyInflow.dropna()
    monthlyInflow["Date"] = monthlyInflow["Date"].astype(int)
    monthlyInflow["Date"] = monthlyInflow["Date"].astype(str)
    monthlyInflow["Inflow"] = monthlyInflow["Inflow"].astype(float)
    yyyymm = monthlyInflow["Date"].str[:6].astype(int) # courtesy of http://stackoverflow.com/questions/12604909/pandas-how-to-change-all-the-values-of-a-column
    monthlyInflow["yyyymm"] = yyyymm
    monthlyInflow = monthlyInflow.drop("Date",1)    
    monthlyInflow = monthlyInflow.groupby("yyyymm").aggregate(np.average)
    monthlyInflow.reset_index(level=0, inplace=True)

 
    monthlyOutflow = outflow
    monthlyOutflow[monthlyOutflow < 0] = np.nan
    monthlyOutflow = monthlyOutflow.replace('m',np.nan)
    monthlyOutflow = monthlyOutflow.dropna()
    monthlyOutflow["Date"] = monthlyOutflow["Date"].astype(int)
    monthlyOutflow["Date"] = monthlyOutflow["Date"].astype(str)
    monthlyOutflow["Outflow"] = monthlyOutflow["Outflow"].astype(float)
    yyyymm = monthlyOutflow["Date"].str[:6].astype(int) # courtesy of http://stackoverflow.com/questions/12604909/pandas-how-to-change-all-the-values-of-a-column
    monthlyOutflow["yyyymm"] = yyyymm
    monthlyOutflow = monthlyOutflow.drop("Date",1)        
    monthlyOutflow = monthlyOutflow.groupby("yyyymm").aggregate(np.average)
    monthlyOutflow.reset_index(level=0, inplace=True)

    monthlyStorage = storage
    monthlyStorage[monthlyStorage < 0] = np.nan
    monthlyStorage = monthlyStorage.replace('m',np.nan)
    monthlyStorage = monthlyStorage.dropna()
    monthlyStorage["Date"] = monthlyStorage["Date"].astype(int)
    monthlyStorage["Date"] = monthlyStorage["Date"].astype(str)
    monthlyStorage["Storage"] = monthlyStorage["Storage"].astype(float)
    yyyymm = monthlyStorage["Date"].str[:6].astype(int) # courtesy of http://stackoverflow.com/questions/12604909/pandas-how-to-change-all-the-values-of-a-column
    monthlyStorage["yyyymm"] = yyyymm    
    monthlyStorage = monthlyStorage.drop("Date",1)        
    monthlyStorage = monthlyStorage.groupby("yyyymm").aggregate(np.average)
    monthlyStorage.reset_index(level=0, inplace=True)

    
    monthlyMerged = pd.merge(monthlyInflow,monthlyOutflow,on="yyyymm")
    monthlyMerged = pd.merge(monthlyMerged,monthlyStorage,on="yyyymm")
    monthlyMerged[monthlyMerged < 0] = np.nan
    monthlyMerged = monthlyMerged.dropna()
    
#    return None,None,monthlyMerged

    # create the daily dataset
    m1 = pd.merge(inflow,storage,how="inner",on="Date")
    merged = pd.merge(m1,outflow,how="inner",on="Date")
    
    # remove any rows with string, courtesy of stack overflow
    """rows_with_strings = merged.apply(
    lambda row : 
        any([ isinstance(e,"m") for e in row])
        , axis=1)
    
    merged = merged[~rows_with_strings]
    merged = merged.dropna()
    """
    merged[merged < 0] = np.nan
    merged = merged.replace('m',np.nan)
    merged = merged.dropna()
    merged = merged.convert_objects(convert_numeric=True)
    merged["Date"] = merged["Date"].astype(int)
    merged["Date"] = merged["Date"].astype(str)
    # convert the seasons
    
    
    # standardize
    acreFootToCubicFoot = 43560.0
    secondsPerDay = 86400.0
    standardizeValue = reservoirParameters["Storage Capacity [ac-ft]"][0] * acreFootToCubicFoot
    merged["Inflow"] = merged["Inflow"].multiply(secondsPerDay).divide(standardizeValue)
    merged["Storage"] = merged["Storage"].multiply(acreFootToCubicFoot).divide(standardizeValue)
    merged["Outflow"] = merged["Outflow"].multiply(secondsPerDay).divide(standardizeValue)
    
    monthlyMerged["Inflow"] = monthlyMerged["Inflow"].multiply(secondsPerDay).divide(standardizeValue)
    monthlyMerged["Storage"] = monthlyMerged["Storage"].multiply(acreFootToCubicFoot).divide(standardizeValue)
    monthlyMerged["Outflow"] = monthlyMerged["Outflow"].multiply(secondsPerDay).divide(standardizeValue)
    
#    return None,None,monthlyMerged
    
    # remove any clear outliers where any standardized value exceeds one
    merged[merged < 0] = np.nan
    merged[merged["Inflow"] > 1] = np.nan
    merged[merged["Storage"] > 1] = np.nan
    merged[merged["Outflow"] > 1] = np.nan
    merged = merged.dropna()


    
    seasons = {"Spring":["0401","0630"],"Summer":["0701","0931"],"Fall":["1001","1231"],"Winter1":["0101","0331"]}
    seasonIDs = {"Spring":1,"Summer":2,"Fall":3,"Winter1":4}
    seasonsList = []
    Spring = []
    Summer = []
    Fall = []
    Winter = []

    for d in merged["Date"]:
        mmdd = int(d[4:])
        season = -9
        for k,v in seasons.iteritems():
            if int(v[0]) <= mmdd <= int(v[1]):
                season = seasonIDs[k]
                seasonsList.append(season)
        if season == -9:
                print str(mmdd) + " error!"
        if season == 1:
            Spring.append(1)
            Summer.append(0)
            Fall.append(0)
            Winter.append(0)
        if season == 2:
            Spring.append(0)
            Summer.append(1)
            Fall.append(0)
            Winter.append(0)
        if season == 3:
            Spring.append(0)
            Summer.append(0)
            Fall.append(1)
            Winter.append(0)
        if season == 4:
            Spring.append(0)
            Summer.append(0)
            Fall.append(0)
            Winter.append(1)
            
    merged["Season"] = seasonsList
    merged["Spring"] = Spring
    merged["Summer"] = Summer
    merged["Fall"] = Fall
    merged["Winter"] = Winter
    
    monthlySeasons = {"Spring":["04","06"],"Summer":["07","09"],"Fall":["10","12"],"Winter1":["1","3"]}
    monthlySeasonIDs = {"Spring":1,"Summer":2,"Fall":3,"Winter1":4}
    monthlySeasonsList = []
    monthlySpring = []
    monthlySummer = []
    monthlyFall = []
    monthlyWinter = []   
#    print monthlyMerged[0:5]
    for d in monthlyMerged["yyyymm"]:
        d = str(d)
        mm = int(d[4:])
        monthlySeason = -9
        for k,v in monthlySeasons.iteritems():
            if int(v[0]) <= mm <= int(v[1]):
                monthlySeason = monthlySeasonIDs[k]
                monthlySeasonsList.append(monthlySeason)
        if monthlySeason == -9:
                print str(mm) + " error!"
        if monthlySeason == 1:
            monthlySpring.append(1)
            monthlySummer.append(0)
            monthlyFall.append(0)
            monthlyWinter.append(0)
        if monthlySeason == 2:
            monthlySpring.append(0)
            monthlySummer.append(1)
            monthlyFall.append(0)
            monthlyWinter.append(0)
        if monthlySeason == 3:
            monthlySpring.append(0)
            monthlySummer.append(0)
            monthlyFall.append(1)
            monthlyWinter.append(0)
        if monthlySeason == 4:
            monthlySpring.append(0)
            monthlySummer.append(0)
            monthlyFall.append(0)
            monthlyWinter.append(1)       

    monthlyMerged["Season"] = monthlySeasonsList
    monthlyMerged["Spring"] = monthlySpring
    monthlyMerged["Summer"] = monthlySummer
    monthlyMerged["Fall"] = monthlyFall
    monthlyMerged["Winter"] = monthlyWinter  

    
    # Implement lagged and leading varaibles
    merged["Inflow(t-1)"] = merged["Inflow"].shift()
    merged["Inflow(t-2)"] = merged["Inflow"].shift(2)
    merged["Inflow(t-3)"] = merged["Inflow"].shift(3)
    merged["Inflow(3day)"] = merged["Inflow"] + merged["Inflow(t-1)"] + merged["Inflow(t-2)"] + merged["Inflow(t-3)"]
    merged["Inflow(t+1)"] = merged["Inflow"].shift(-1)
    merged["Inflow(t+2)"] = merged["Inflow"].shift(-2)
    merged["Inflow(t+3)"] = merged["Inflow"].shift(-3)
    
    # Implement summed lagged and leading variables
    merged = createLaggedRange(merged,"Inflow","Inflow(7day)",7,leading=False)
    merged = createLaggedRange(merged,"Inflow","Inflow(14day)",14,leading=False)
    merged = createLaggedRange(merged,"Inflow","Inflow(21day)",21,leading=False)
    merged = createLaggedRange(merged,"Inflow","Inflow(28day)",28,leading=False)

    merged = createLaggedRange(merged,"Inflow","Inflow(7dayForecast)",7,leading=True)

    merged = createLaggedRange(merged,"Outflow","Outflow(7day)",7,leading=False)
    merged = createLaggedRange(merged,"Outflow","Outflow(14day)",14,leading=False)
    merged = createLaggedRange(merged,"Outflow","Outflow(21day)",21,leading=False)
    merged = createLaggedRange(merged,"Outflow","Outflow(28day)",28,leading=False)
    
    monthlyMerged = createLaggedRange(monthlyMerged,"Inflow","Inflow(1month)",1,leading=False)
    monthlyMerged = createLaggedRange(monthlyMerged,"Inflow","Inflow(2month)",2,leading=False)
    monthlyMerged = createLaggedRange(monthlyMerged,"Inflow","Inflow(1monthForecast)",1,leading=True)

    monthlyMerged = createLaggedRange(monthlyMerged,"Outflow","Outflow(1month)",1,leading=False)
    monthlyMerged = createLaggedRange(monthlyMerged,"Outflow","Outflow(2month)",2,leading=False)
    
    # include only the previous day's storage
    merged["Storage"] = merged["Storage"].shift()
    
    merged = merged.dropna()


    # Daily
    xcols = ["Spring","Summer","Fall","Winter","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage","Outflow(28day)","Outflow(21day)","Outflow(14day)","Outflow(7day)"]  
    ycol = "Outflow"
    
    dailyMerged = autoDecisionTree(merged,xcols,ycol,reservoirPath,reservoirPath + "_Daily")
    
    xcols = ["Spring","Summer","Fall","Winter","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage","Outflow(28day)","Outflow(21day)","Outflow(14day)","Outflow(7day)"]  
    ycol = "Outflow"
    
    dailyMerged = autoDecisionTree(merged,xcols,ycol,reservoirPath,reservoirPath + "_Daily_6deep",maxDepth=6)
    
    # monthly
    xcols = ["Spring","Summer","Fall","Winter","Inflow(1month)","Inflow(2month)","Inflow","Inflow(1monthForecast)","Storage","Outflow(1month)","Outflow(2month)"]  
    ycol = "Outflow"
    
    monthlyMerged = autoDecisionTree(monthlyMerged,xcols,ycol,reservoirPath,reservoirPath + "_Monthly")
      
    # Seasonally
    seasonalMerged = merged
    yyyy = seasonalMerged["Date"].str[:4].astype(int) # courtesy of http://stackoverflow.com/questions/12604909/pandas-how-to-change-all-the-values-of-a-column
    seasonalMerged["yyyy"] = yyyy
    seasonalMerged = seasonalMerged.groupby(["Season","yyyy"]).aggregate(np.average)
    
    xcols = ["Spring","Summer","Fall","Winter","Inflow","Storage"]
    ycol = "Outflow"

    seasonalMerged = autoDecisionTree(seasonalMerged,xcols,ycol,reservoirPath,reservoirPath + "_Seasonal")

    
    return merged,dailyMerged,monthlyMerged,seasonalMerged
    
def createLaggedRange(df,col,newcol,laggedRange,leading=False):
    ar = np.zeros(len(df))
    for i in range(1,laggedRange + 1):
        if leading:
            ar += df[col].shift(-i)
        else:
            ar += df[col].shift(i)    
            
    ar /= laggedRange
    df[newcol] = ar
    df = df.dropna()
    return df
    
    
def test_new(reservoir):
    reservoirData = pd.read_excel("Reservoir Information.xlsm")
    reservoirParameters = reservoirData[reservoirData["ID"]==reservoir]
    reservoirParameters = reservoirParameters.reset_index()
    merged,dailyMerged,monthlyMerged,seasonalMerged = processReservoir(reservoir,reservoirParameters)
    return merged
    
def run():
    reservoirData = pd.read_excel("Reservoir Information_2.xlsm")
    dailyColumns = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage","Outflow"]
    monthlyColumns = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(1month)","Inflow(2month)","Inflow","Inflow(1monthForecast)","Storage","Outflow"]
    seasonalColumns = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage","Outflow"]
    allDaily = pd.DataFrame(columns=dailyColumns)
    allMonthly = pd.DataFrame(columns=monthlyColumns)
    allSeasonally = pd.DataFrame(columns=seasonalColumns)
    
    for reservoir in reservoirs:
        reservoirParameters = reservoirData[reservoirData["ID"]==reservoir]
        reservoirParameters = reservoirParameters.reset_index()
        merged,dailyMerged,monthlyMerged,seasonalMerged = processReservoir(reservoir,reservoirParameters)

        # update the dataset
        dailyMerged["Main_C"] = reservoirParameters["Main_C"][0]
        dailyMerged["Main_S"] = reservoirParameters["Main_S"][0]
        dailyMerged["Main_H"] = reservoirParameters["Main_H"][0]
        dailyMerged["Main_I"] = reservoirParameters["Main_I"][0]

        dailyMerged["Capacity"] = reservoirParameters["Storage Capacity [ac-ft]"][0]
        dailyMerged["Capacity_Small"] = reservoirParameters["Capacity_Small"][0]
        dailyMerged["Capacity_Medium"] = reservoirParameters["Capacity_Medium"][0]
        dailyMerged["Capacity_Large"] = reservoirParameters["Capacity_Large"][0]

        allDaily = pd.concat([allDaily,dailyMerged])
        
        
        monthlyMerged["Main_C"] = reservoirParameters["Main_C"][0]
        monthlyMerged["Main_S"] = reservoirParameters["Main_S"][0]
        monthlyMerged["Main_H"] = reservoirParameters["Main_H"][0]
        monthlyMerged["Main_I"] = reservoirParameters["Main_I"][0]

        monthlyMerged["Capacity"] = reservoirParameters["Storage Capacity [ac-ft]"][0]
        monthlyMerged["Capacity_Small"] = reservoirParameters["Capacity_Small"][0]
        monthlyMerged["Capacity_Medium"] = reservoirParameters["Capacity_Medium"][0]
        monthlyMerged["Capacity_Large"] = reservoirParameters["Capacity_Large"][0]        
        
        allMonthly = pd.concat([allMonthly,monthlyMerged])
     
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allDaily,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs",maxDepth=6,minSamplesLeaf=50)
    
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(1month)","Inflow(2month)","Inflow","Inflow(1monthForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allMonthly,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_sizecategories",maxDepth=6,minSamplesLeaf=50)

    allCDaily = allDaily[allDaily["Main_C"] == 1]
    allSDaily = allDaily[allDaily["Main_S"] == 1]
    allHDaily = allDaily[allDaily["Main_H"] == 1]
    allIDaily = allDaily[allDaily["Main_I"] == 1]
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    
    autoDecisionTree(allCDaily,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainC",maxDepth=6,minSamplesLeaf=5)
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"    
    autoDecisionTree(allSDaily,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainS",maxDepth=6,minSamplesLeaf=5)
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"    
    autoDecisionTree(allHDaily,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainH",maxDepth=6,minSamplesLeaf=5)
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allIDaily,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainI",maxDepth=6,minSamplesLeaf=5)


    allCMonthly = allMonthly[allMonthly["Main_C"] == 1]
    allSMonthly = allMonthly[allMonthly["Main_S"] == 1]
    allHMonthly = allMonthly[allMonthly["Main_H"] == 1]
    allIMonthly = allMonthly[allMonthly["Main_I"] == 1]
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(1month)","Inflow(2month)","Inflow","Inflow(1monthForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allCMonthly,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","monthlyAllReservoirs_MainC",maxDepth=6,minSamplesLeaf=5)
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(1month)","Inflow(2month)","Inflow","Inflow(1monthForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allSMonthly,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","monthlyAllReservoirs_MainS",maxDepth=6,minSamplesLeaf=5)
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(1month)","Inflow(2month)","Inflow","Inflow(1monthForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allHMonthly,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","monthlyAllReservoirs_MainH",maxDepth=6,minSamplesLeaf=5)
    xcols = ["Capacity","Main_C","Main_S","Main_H","Main_I","Spring","Summer","Fall","Winter","Inflow(1month)","Inflow(2month)","Inflow","Inflow(1monthForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allIMonthly,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","monthlyAllReservoirs_MainI",maxDepth=6,minSamplesLeaf=5)
    
    # by use, by season
    allCDaily_Summer = allDaily[allDaily["Main_C"] == 1]
    allCDaily_Summer = allCDaily_Summer[allCDaily_Summer["Summer"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allCDaily_Summer,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainC_Summer",maxDepth=4,minSamplesLeaf=5)
        
    allCDaily_Fall = allDaily[allDaily["Main_C"] == 1]
    allCDaily_Fall = allCDaily_Fall[allCDaily_Fall["Fall"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allCDaily_Fall,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainC_Fall",maxDepth=4,minSamplesLeaf=5)
    
    allCDaily_Spring = allDaily[allDaily["Main_C"] == 1]
    allCDaily_Spring = allCDaily_Spring[allCDaily_Spring["Spring"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allCDaily_Spring,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainC_Spring",maxDepth=4,minSamplesLeaf=5)
    
    allCDaily_Winter = allDaily[allDaily["Main_C"] == 1]
    allCDaily_Winter = allCDaily_Winter[allCDaily_Winter["Winter"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allCDaily_Winter,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainC_Winter",maxDepth=4,minSamplesLeaf=5)

    allSDaily_Summer = allDaily[allDaily["Main_S"] == 1]
    allSDaily_Summer = allSDaily_Summer[allSDaily_Summer["Summer"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allSDaily_Summer,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainS_Summer",maxDepth=4,minSamplesLeaf=5)
        
    allSDaily_Fall = allDaily[allDaily["Main_S"] == 1]
    allSDaily_Fall = allSDaily_Fall[allSDaily_Fall["Fall"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allSDaily_Fall,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainS_Fall",maxDepth=4,minSamplesLeaf=5)
    
    allSDaily_Spring = allDaily[allDaily["Main_S"] == 1]
    allSDaily_Spring = allSDaily_Spring[allSDaily_Spring["Spring"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allSDaily_Spring,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainS_Spring",maxDepth=4,minSamplesLeaf=5)
    
    allSDaily_Winter = allDaily[allDaily["Main_S"] == 1]
    allSDaily_Winter = allSDaily_Winter[allSDaily_Winter["Winter"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allSDaily_Winter,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainS_Winter",maxDepth=4,minSamplesLeaf=5)


    allHDaily_Summer = allDaily[allDaily["Main_H"] == 1]
    allHDaily_Summer = allHDaily_Summer[allHDaily_Summer["Summer"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allHDaily_Summer,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainH_Summer",maxDepth=4,minSamplesLeaf=5)
        
    allHDaily_Fall = allDaily[allDaily["Main_H"] == 1]
    allHDaily_Fall = allHDaily_Fall[allHDaily_Fall["Fall"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allHDaily_Fall,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainH_Fall",maxDepth=4,minSamplesLeaf=5)
    
    allHDaily_Spring = allDaily[allDaily["Main_H"] == 1]
    allHDaily_Spring = allHDaily_Spring[allHDaily_Spring["Spring"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allHDaily_Spring,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainH_Spring",maxDepth=4,minSamplesLeaf=5)
    
    allHDaily_Winter = allDaily[allDaily["Main_H"] == 1]
    allHDaily_Winter = allHDaily_Winter[allHDaily_Winter["Winter"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allHDaily_Winter,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainH_Winter",maxDepth=4,minSamplesLeaf=5)

    allIDaily_Summer = allDaily[allDaily["Main_I"] == 1]
    allIDaily_Summer = allIDaily_Summer[allIDaily_Summer["Summer"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allIDaily_Summer,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainI_Summer",maxDepth=4,minSamplesLeaf=5)
        
    allIDaily_Fall = allDaily[allDaily["Main_I"] == 1]
    allIDaily_Fall = allIDaily_Fall[allIDaily_Fall["Fall"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allIDaily_Fall,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainI_Fall",maxDepth=4,minSamplesLeaf=5)
    
    allIDaily_Spring = allDaily[allDaily["Main_I"] == 1]
    allIDaily_Spring = allIDaily_Spring[allIDaily_Spring["Spring"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allIDaily_Spring,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainI_Spring",maxDepth=4,minSamplesLeaf=5)
    
    allIDaily_Winter = allDaily[allDaily["Main_I"] == 1]
    allIDaily_Winter = allIDaily_Winter[allIDaily_Winter["Winter"]==1]
    xcols = ["Capacity","Inflow(28day)","Inflow(21day)","Inflow(14day)","Inflow(7day)","Inflow","Inflow(7dayForecast)","Storage"]
    ycol = "Outflow"
    autoDecisionTree(allIDaily_Winter,xcols,ycol,"C:\\RBMProject\\MergedResults_Final","dailyAllReservoirs_MainI_Winter",maxDepth=4,minSamplesLeaf=5)

    
    return dailyMerged, monthlyMerged, seasonalMerged