import sys

fname = sys.argv[1]
with open(fname) as file:
    gold = []
    for line in file.readlines():
        gold.append(line.split("\t")[1].strip() + "\n")

with open(fname, 'w') as file:
    file.writelines(gold[1:])