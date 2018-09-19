import glob
import os
import csv

path = '10000-MTV-Music-Artists-page-1.csv'
output_file = open('bands.txt', 'w+')
csv_reader = csv.reader(open(path, 'r', encoding='latin-1'), delimiter=',')
next(csv_reader)

for row in csv_reader:
    output_file.write(row[0].strip() + '\n')
