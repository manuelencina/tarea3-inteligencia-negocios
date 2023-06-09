import numpy as np
def shannon_entropy(c):
    large=len(c)
    #print(large)
    n=int(np.sqrt(large))+1
    #print(np.sqrt(large))
    #print(round(3.88))
    intervals=np.linspace(min(c),max(c),n+1)
    p=[]
    print(intervals)
    for i in range(n):
        cont=0
        lb=intervals[i]
        ub=intervals[i+1]
        for j in range(large):
            if i==n-1:
                if c[j] >= lb and c[j] <= ub:
                    cont=cont+1
                continue
            if c[j] >= lb and c[j] < ub:
                cont=cont+1
                continue
        #print(cont)
        p.append(cont/large)
    #print(n,len(p))
    return -np.sum(p * np.log2(p))


      




x=np.array([7,8,9])
y=([2,3,4])
C=0
for i in range (len(x)):
    C=C+((x[i]-y[i])**2)/2
print(C/2)
print(np.sum(((x-y)**2)/2)/2)
