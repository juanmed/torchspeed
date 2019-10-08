import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.show()

import torch
from torch.autograd import grad
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set the parameters of our model:
g      = torch.tensor( [ 9.81], requires_grad = True).to(device)
#g.requires_grad = True
m      = torch.tensor( [ 15. ], requires_grad = True).to(device)
#m.requires_grad = True
source = torch.tensor( [0.,.5], requires_grad = True).to(device)
#source.requires_grad = True
target = torch.tensor( [7.,2.], requires_grad = True).to(device)
#target.requires_grad = True

def cost(m, g, P, windy = False):
    "Cost associated to a simple ballistic problem."

    def Emec(q,p) :
        "Particle of mass m in a gravitational field g."
        return m*g*q[1] + (p**2).sum() / (2*m)

    qt = source ; pt = P
    qts = [qt]

    for it in range(10) : # Simple Euler scheme
        [dq,dp] = grad(Emec(qt,pt), [qt,pt], create_graph=True)
        if windy :
            dq += qt[1] * 20 * (torch.randn(2, requires_grad = True).to(device))

        qt = qt + .1 * dp
        pt = pt - .1 * dq
        qts.append(qt)
    
    # Return the squared cost $|q_1 - target|^2_2$
    return ((qt - target)**2).sum(), qts

P = torch.tensor( [60., 30.], requires_grad = True).to(device)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, aspect='equal')

GRAV = False
WIND = False
if GRAV:
    lr = .1
else:
    lr = 5. 

for it in tqdm(range(101)):
    C, qts = cost(m,g,P, WIND)

    if GRAV:
        # optimize w.r.t gravity    
        dg = grad(C, [g])[0]
        g.data -= lr*dg.data
    else:
        # optimize w.r.t velocity
        dP = grad(C, [P])[0]
        P.data -= lr*dP.data        

    if it % 10 == 0 :
        ax.clear()
        a = P.data.cpu().numpy()
        a = np.arctan2(a[0], a[1])
    
    qts = torch.stack(qts).data.cpu().numpy()
    ax.plot(qts[:,0], qts[:,1], '-o', color='#0051FF')

    if WIND :
        for _ in range(10) :
            _, qts = cost(m,g,P, WIND)
            qts = torch.stack(qts).data.cpu().numpy()
            ax.plot(qts[:,0], qts[:,1], '-o', color='#88B0FF', zorder=1.5)

    circle1 = plt.Circle((7, 2), .8, color='#FF8A8A')
    circle2 = plt.Circle((7, 2), .6, color='#FFFFFF')
    circle3 = plt.Circle((7, 2), .4, color='#FF8A8A')
    circle4 = plt.Circle((7, 2), .2, color='#FFFFFF')

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    plt.ylim( 0,5)
    plt.xlim(-2,8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    if   GRAV == True : suff = "grav"
    elif WIND == True : suff = "windy"
    else :              suff = "simple"
    plt.savefig("output/OC_"+suff+"_"+str(it)+".png")
    print("Momentum: {}".format(P.data.cpu().numpy()))

plt.show(block=True)