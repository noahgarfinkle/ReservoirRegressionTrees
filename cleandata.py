# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:25:38 2015

@author: userx
"""
import os

path = "c:\\rbmproject\\test 3"
# courtesy of http://stackoverflow.com/questions/141291/how-to-list-only-top-level-directories-in-python
reservoirs = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

# courtesy http://stackoverflow.com/questions/20364396/how-to-delete-the-first-line-of-a-text-file-using-python
def editFile(fileName,newVar):
    with open(fileName,'r') as fin:
        data = fin.read().splitlines(True)
    data = data[3:]
    with open(fileName,'w') as fout:
        fout.write("Date,PST," + newVar + "\n")
        fout.writelines(data)
        

for reservoir in reservoirs:
    inflow = reservoir + "\\Inflow_daily.txt"
    outflow = reservoir + "\\Outflow_daily.txt"
    storage = reservoir + "\\Storage_daily.txt"   
    
    editFile(inflow,"Inflow")
    editFile(outflow,"Outflow")
    editFile(storage,"Storage")
  