import glob
import os
import csv

path = 'colors.csv'
output_file = open('colors2.txt', 'w+')
csv_reader = csv.reader(open(path, 'r', encoding='latin-1'), delimiter=',')
next(csv_reader)

for row in csv_reader:
    output_file.write(' '.join(row) + '\n')
