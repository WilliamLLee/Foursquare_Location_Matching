import os 
import csv 

def get_lines(file_name, line_count, delimiter=","):
    lines = []
    with open(file_name, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, line in enumerate(reader):
            if i == line_count:
                break
            lines.append(line)
    return lines