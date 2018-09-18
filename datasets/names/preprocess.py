import glob
import os
import csv

path = 'nomes-censos-ibge.csv'
output_file = open('names.txt', 'w+')


csv_reader = csv.reader(open(path, 'r', encoding='latin-1'), delimiter=';')
next(csv_reader)

for row in csv_reader:
    output_file.write(row[0] + '\n')
    
    
