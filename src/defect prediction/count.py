with open('slurm-198841.out', 'r') as f:
    d = f.readlines()
with open('slurm-198840.out', 'r') as f:
    e = f.readlines()
with open('slurm-198761.out', 'r') as f:
    i = f.readlines()
with open('slurm-198764.out', 'r') as f:
    j = f.readlines()
with open('slurm-198765.out', 'r') as f:
    k = f.readlines()
with open('slurm-198878.out', 'r') as f:
    l = f.readlines()
with open('slurm-198879.out', 'r') as f:
    m = f.readlines()
with open('slurm-198944.out', 'r') as f:
    h = f.readlines()
with open('slurm-198947.out', 'r') as f:
    n = f.readlines()
with open('slurm-198948.out', 'r') as f:
    o = f.readlines()
with open('slurm-198967.out', 'r') as f:
    p = f.readlines()
with open('slurm-198968.out', 'r') as f:
    q = f.readlines()
with open('slurm-198971.out', 'r') as f:
    r = f.readlines()
with open('slurm-198972.out', 'r') as f:
    s = f.readlines()

d = [line for line in d if line.startswith('tp,')]
h = [line for line in h if line.startswith('tp,')]
e = [line for line in e if line.startswith('tp,')]
i = [line for line in i if line.startswith('tp,')]
j = [line for line in j if line.startswith('tp,')]
k = [line for line in k if line.startswith('tp,')]
l = [line for line in l if line.startswith('tp,')]
m = [line for line in m if line.startswith('tp,')]
n = [line for line in n if line.startswith('tp,')]
o = [line for line in o if line.startswith('tp,')]
p = [line for line in p if line.startswith('tp,')]
q = [line for line in q if line.startswith('tp,')]
s = [line for line in s if line.startswith('tp,')]
r = [line for line in r if line.startswith('tp,')]

print('\nWPDP:\n----')
print('part1 (198944): ' + str(len(h)) + " done - " + str(len(h) * 100. / 1800) + '%')

print('\nCPDP:\n----')
print('part1 (198841): ' + str(len(d)) + " done - " + str(len(d) * 100. / 3000) + '%')
print('part2 (198840): ' + str(len(e)) + " done - " + str(len(e) * 100. / 4200) + '%')

print('\nF1:\n----')
print('part1 (198761): ' + str(len(i)) + " done - " + str(len(i) * 100. / 3000) + '%')
print('part2 (198967): ' + str(len(o)) + " done - " + str(len(o) * 100. / 1800) + '%')
print('part3 (198968): ' + str(len(p)) + " done - " + str(len(p) * 100. / 1200) + '%')

print('\npopt20:\n----')
print('part1 (198764): ' + str(len(j)) + " done - " + str(len(j) * 100. / 1200) + '%')
print('part2 (198765): ' + str(len(k)) + " done - " + str(len(k) * 100. / 3000) + '%')
print('part3 (198971): ' + str(len(r)) + " done - " + str(len(r) * 100. / 1800) + '%')

print('\nd2h:\n----')
print('part1 (198878): ' + str(len(l)) + " done - " + str(len(l) * 100. / 1200) + '%')
print('part2 (198879): ' + str(len(m)) + " done - " + str(len(m) * 100. / 3000) + '%')
print('part3 (198970): ' + str(len(q)) + " done - " + str(len(q) * 100. / 1800) + '%')

print('\nAUC:\n----')
print('part1 (198947): ' + str(len(n)) + " done - " + str(len(n) * 100. / 3000) + '%')
print('part2 (198948): ' + str(len(o)) + " done - " + str(len(o) * 100. / 3000) + '%')
print("")
