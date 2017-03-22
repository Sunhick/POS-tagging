import sys

if len(sys.argv)!=3:
	print('Usage: python evalPOS.py goldPOS_filename systemPOS_filename')
	sys.exit(1)

gold = [line.strip() for line in open(sys.argv[1], 'r')]
system = [line.strip() for line in open(sys.argv[2], 'r')]

if len(gold)!=len(system):
	print('Number of lines between gold and system do not match!')
	sys.exit(1)

totalCnt = 0
correctCnt = 0
for sysLine, goldLine in zip(system, gold):      #loop through both lists of lines
	if not sysLine:                       #if the original was an empty line, skip
		continue
	totalCnt += 1
	if sysLine.split()[1] == goldLine.split()[1]:
                correctCnt += 1         #if the tags match, increment correct count

print('Accuracy is ', correctCnt/totalCnt)
