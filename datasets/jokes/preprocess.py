import glob
import os
import csv

path = 'shortjokes.csv'
output_file = open('jokes.txt', 'w+')
csv_reader = csv.reader(open(path, 'r', encoding='latin-1'), delimiter=',')
next(csv_reader)

for row in csv_reader:
    output_file.write(row[1] + '\n')
    
    
