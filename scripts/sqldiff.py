import sys
import sqlite3

if len(sys.argv) < 3:
    print("Usage: python sqldiff.py <file1> <file2>")
    exit()

QUERY = "SELECT * FROM pci_ids"
COL_NAME_QUERY = "PRAGMA table_info(pci_ids)"

db1 = sys.argv[1]
db2 = sys.argv[2]

conn1 = sqlite3.connect(db1)
conn2 = sqlite3.connect(db2)

cursor1 = conn1.cursor()
cursor2 = conn2.cursor()

res1 = set(cursor1.execute(QUERY).fetchall())
res2 = set(cursor2.execute(QUERY).fetchall())

cols = cursor1.execute(COL_NAME_QUERY).fetchall()
cols = tuple(col[1] for col in cols)

print("### Additions")
print("```")
print(cols)
for row in res2 - res1:
    print(row)
print("```")

print("### Deletions")
print("```")
print(cols)
for row in res1 - res2:
    print(row)
print("```")
