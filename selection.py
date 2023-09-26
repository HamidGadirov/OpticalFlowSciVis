from pathlib import Path
HOME=str(Path.home())

import os
import sys
# sys.path.insert(0, os.path.abspath(HOME+'/dev/helper/helper_timesel'))
sys.path.insert(0, os.path.abspath('/Users/hamidgadirov/Desktop/OpticalFlow/FlowSciVis/helper_timesel'))

# python3.11 selection.py

import helper_timesel as sel

nTimeSteps = 169 # 70 169
# distsFName = f'{HOME}/Desktop/OpticalFlow/FlowSciVis/dist_matrix_70.txt'
# distsFName=f'/Users/hamidgadirov/Desktop/OpticalFlow/FlowSciVis/dist_matrix_169.txt'
distsFName=f'/Users/hamidgadirov/Desktop/OpticalFlow/FlowSciVis/dist_matrix_169_1.txt'

# k=7 v=4173.755582 | selection: [0, 21, 72, 98, 130, 158, 168]
# k=8 v=4219.463857 | selection: [0, 21, 72, 98, 131, 147, 159, 168]
# k=9 v=4260.105192 | selection: [0, 33, 64, 74, 98, 131, 147, 159, 168]
# k=10 v=4300.339904 | selection: [0, 33, 64, 74, 98, 120, 135, 147, 159, 168]
# k=11 v=4315.807689 | selection: [0, 21, 33, 64, 74, 98, 120, 135, 147, 159, 168]
# k=12 v=4327.717379000001 | selection: [0, 21, 33, 64, 74, 98, 115, 120, 135, 147, 159, 168]
# k=16 v=4336.2659380000005 | selection: [0, 21, 29, 37, 56, 65, 74, 98, 115, 119, 131, 135, 142, 147, 159, 168]

# dist_matrix_169_1
# k=8 v=4238.963533 | selection: [0, 30, 56, 84, 115, 142, 160, 168]

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
    selection,v=sel.select_dynProg_max(k, dists, nTimeSteps)
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
