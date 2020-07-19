import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import networkx as nx
import pandas as pd

with open('../subj_dist.json') as f:
    data = json.load(f)

with open('../jrnl_subj.json') as f:
    jrl = json.load(f)

name = pd.read_csv('../subjects.csv')

m = np.zeros([len(data.keys()),len(data.keys())])

sub = dict(zip(data.keys(),np.arange(len(data.keys()))))
subx = dict(zip(np.arange(len(data.keys())),data.keys()))

for i in tqdm(range(len(data.keys()))):
    for j in list(jrl.values()):
        if subx[i] in j:
            for k in j:
                try:
                    m[i, sub[k]] += 1
                except:
                    print(i,j,k)

G=nx.from_numpy_matrix(m)
nx.set_node_attributes(G, {k: {'label': subx[k]} for k in subx.keys()})

pos = np.array(list(nx.spring_layout(G).values())).T

markers = []
for i in list(data.keys()):
    a = name[name.code == int(i)][' title'].values
    markers.append(a[0])

field = []
for i in list(data.keys()):
    a = name[name.code == int(i)][' field '].values
    field.append(a[0])

label = dict(zip(list(set(field)),np.arange(len(list(set(field))))))

x,y = pos[0]*5000+1000,pos[1]*5000+1000
new_x = []
for i in x:
    try:
        if i > 0 :
            new_x.append(np.log(i))
        elif i < 0 :
            new_x.append(-1* np.log(-1*i))
        else:
            new_x.append(0.001)
    except:
        print(i)

new_y = []
for i in y:
    try:
        if i > 0 :
            new_y.append(np.log(i))
        elif i < 0 :
            new_y.append(-1* np.log(-1*i))
        else:
            new_y.append(0.001)
    except:
        print(i)
        
x = np.dot(new_x,5)
y = np.dot(new_y,5)

y[y<32] = 32
x[x<32] = 32
y[y>38] = 38
x[x>38] = 38

N = pos[0].shape
NumOfPub = np.array(list(data.values()))
NumOfPub[NumOfPub<30] = 30
NumOfPub[(50>NumOfPub) & (NumOfPub>30)] = 50
radii = NumOfPub/5000

data = dict(x=x,y=y,radii1=radii*10000,radii2=radii*3,field=field,markers=markers,number=np.array(list(data.values())))
plt.title('Eigenvector Centrality')
sc = plt.scatter('x','y',s='radii1',c=nx.eigenvector_centrality(G).values(),data=data)
plt.colorbar(sc)
plt.show()
plt.title('Clustering Coefficient')
sc = plt.scatter('x','y',s='radii1',c=nx.clustering(G).values(),data=data)
plt.colorbar(sc)
plt.show()