from pathlib import Path
HOME=str(Path.home())

import os
import sys
sys.path.insert(0, os.path.abspath(HOME+'/dev/autovis/bone/code'))

import select_bone as sel


nTimeSteps=150
distsFName=f'{HOME}/Desktop/dist_matrix_150.txt'

with open(distsFName) as f:
    lines = [line.rstrip().split(' ') for line in f]
    print(len(lines), len(lines[0]))
    dists=[float(e) for l in lines for e in l]


#dists=[e if e>0 else -1000000000. for e in dists]

def d(b0, b1):
    assert b0<=b1
    return dists[b1+nTimeSteps*b0]

selections=[]
for k in range(2,17):
    selection,v=sel.select_dynProg(k, dists, nTimeSteps)
    selections.append((selection,v))

#print(f'selections {selections}')

for sel,v in selections:
    print(f'k={len(sel)} v={v} | selection: {sel}')


#
# test uniform
# 

for k in range(2,17):

    step=nTimeSteps//(k-1)

    selection=[]
    for i in range(k-1):
        selection.append(i*step)
    selection.append(nTimeSteps-1)

    assert len(selection)==k
    v=0

    for i in range(1,k):
        v+=d(selection[i-1],selection[i])


    print(f'UNIFORM k={k} v={v} | selection: {selection}')

    selections.append(selection)


print('selections', selections)
