import numpy as np
import sympy as sy

#VARIABLES
dz = -0.01
L = 4.0
N = int(abs(2*L/dz))
w0x = 1.0
w0y = 1.0
alpha_R = 0.1   ## alpha_R -> 2*alpha_R
r0 = -0.40

### FUNCTIONS
# Eigenstates (only confinament potential)
def h_c(nx1,nx2,ny1,ny2,s1,s2):
    return ((nx1+0.5) + (w0y/w0x)*(ny1+0.5))*sy.KroneckerDelta(nx1,nx2)*sy.KroneckerDelta(ny1,ny2)*sy.KroneckerDelta(s1,s2)

# Rashba (base sigma_x)
def h_Rx1(nx1,nx2,ny1,ny2,s1,s2):
    return 2*alpha_R*(-1j)*(np.sqrt(w0y/(2*w0x)))*(np.sqrt(ny2)*sy.KroneckerDelta(ny1,ny2-1) - np.sqrt(ny2+1)*sy.KroneckerDelta(ny1,ny2+1))*sy.KroneckerDelta(nx1,nx2)*sy.KroneckerDelta(s1,-s2)

def h_Rx2(nx1,nx2,ny1,ny2,s1,s2):
    return -2*alpha_R*sy.KroneckerDelta(nx1,nx2)*sy.KroneckerDelta(ny1,ny2)*(-1j)*s2*sy.KroneckerDelta(s1,-s2)

# Deformation
def alpha(z):
    if z<0: delta_z = r0*(z/L + 1)
    if z>0: delta_z = r0*(-z/L + 1)
    if z==0: delta_z = 0.0
    return 0.5*(1/(1+delta_z)**4 - 1)


def h_V1(nx1,nx2,ny1,ny2,s1,s2):
    return (2*nx1 + 2*ny1 + 1)*sy.KroneckerDelta(nx1,nx2)*sy.KroneckerDelta(ny1,ny2)*sy.KroneckerDelta(s1,s2)

def h_V2(nx1,nx2,ny1,ny2,s1,s2):
    return (np.sqrt(nx2*(nx2-1))*sy.KroneckerDelta(nx1,nx2-2) + np.sqrt((nx2+1)*(nx2+2))*sy.KroneckerDelta(nx1,nx2+2))*sy.KroneckerDelta(ny1,ny2)*sy.KroneckerDelta(s1,s2)

def h_V3(nx1,nx2,ny1,ny2,s1,s2):
    return (np.sqrt(ny2*(ny2-1))*sy.KroneckerDelta(ny1,ny2-2) + np.sqrt((ny2+1)*(ny2+2))*sy.KroneckerDelta(ny1,ny2+2))*sy.KroneckerDelta(nx1,nx2)*sy.KroneckerDelta(s1,s2)


# SUBSPACE
v0 = [[0,0,+1], [0,0,-1], [0,1,+1], [0,1,-1], [0,2,+1], [0,2,-1], [0,3,+1], [0,3,-1], [0,4,+1], [0,4,-1]]
v1 = [[1,0,+1], [1,0,-1], [1,1,+1], [1,1,-1], [1,2,+1], [1,2,-1], [1,3,+1], [1,3,-1]]
v2 = [[0, 2, 1], [0, 2, -1], [1, 2, 1], [1, 2, -1], [2, 2, 1], [2, 2, -1]]
v_coupling = [[0,0,+1], [0,0,-1], [0,1,+1], [0,1,-1], [0,2,+1], [0,2,-1], [0,3,+1], [0,3,-1]]

v = v_coupling

# MATRICES
E_c = np.array([[h_c(v[i][0],v[j][0],v[i][1],v[j][1],v[i][2],v[j][2]) for j in range(len(v))] for i in range(len(v))], dtype=complex)

H_Rx1 = np.array([[h_Rx1(v[i][0],v[j][0],v[i][1],v[j][1],v[i][2],v[j][2]) for j in range(len(v))] for i in range(len(v))], dtype=complex)

H_Rx2 = np.array([[h_Rx2(v[i][0],v[j][0],v[i][1],v[j][1],v[i][2],v[j][2]) for j in range(len(v))] for i in range(len(v))], dtype=complex)

H_V1 = np.array([[h_V1(v[i][0],v[j][0],v[i][1],v[j][1],v[i][2],v[j][2]) for j in range(len(v))] for i in range(len(v))], dtype=complex)

H_V2 = np.array([[h_V2(v[i][0],v[j][0],v[i][1],v[j][1],v[i][2],v[j][2]) for j in range(len(v))] for i in range(len(v))], dtype=complex)

H_V3 = np.array([[h_V3(v[i][0],v[j][0],v[i][1],v[j][1],v[i][2],v[j][2]) for j in range(len(v))] for i in range(len(v))], dtype=complex)


def RT(E):
    kk = np.array([np.sqrt(2*(E-E_c[i,i])) for i in range(len(v))])
    
    psi3 = np.array([[np.exp(1j*kk[i]*L)*sy.KroneckerDelta(i,j) for j in range(len(v))] for i in range(len(v))], dtype=complex)
    psi3 = np.array([psi3, psi3*1j*kk -(1j*H_Rx2/2).dot(psi3)])
    
    def g(f,z):
        H = E_c + H_Rx1 + alpha(z)*(H_V1 + H_V2 + H_V3)
        return np.array([f[1], (H - np.identity(len(v))*E).dot(f[0])-(1j*H_Rx2).dot(f[1])])
    
    f, z = psi3, L
    for i in range(N):
        r1 = g(f, z)
        r2 = g(f + dz*r1/2, z + dz/2)
        r3 = g(f + dz*r2/2, z + dz/2)
        r4 = g(f + dz*r3, z + dz)
        f = f + dz*(r1 + 2*r2 + 2*r3 + r4)/6
        z = z + dz
    
    psi2 = sy.Matrix.vstack(sy.Matrix(f[0]), sy.Matrix(f[1]))
    r = sy.Matrix(len(v),len(v), lambda i,j: sy.var('r_%d_%d' %(i,j)))
    t = sy.Matrix(len(v),len(v), lambda i,j: sy.var('t_%d_%d' %(i,j)))
    z = sy.var('z')
    
    psi1 = sy.Matrix(len(v),len(v), lambda i,j: sy.KroneckerDelta(i,j)*sy.exp(1j*kk[j]*z))+sy.Matrix(len(v),len(v), lambda i,j: sy.KroneckerDelta(i,j)*sy.exp(-1j*kk[j]*z))*r
    psi1 = sy.Matrix.vstack(psi1, sy.diff(psi1,z))
    
    M = sy.Matrix.vstack(psi1[0:len(v),0:len(v)] - psi2[0:len(v),0:len(v)]*t, psi1[len(v):2*len(v),0:len(v)] - psi2[len(v):2*len(v),0:len(v)]*t - (1j*H_Rx2/2)*psi2[0:len(v),0:len(v)]*t).subs(z,-L)
    
    eqs = [M[i] for i in range(len(M))]
    vabls = [r[i] for i in range(len(r))]+[t[i] for i in range(len(t))]
    
    S = sy.linear_eq_to_matrix(eqs,vabls)
    A = np.array(S[0],dtype=complex)
    B = np.array(S[1],dtype=complex)
    
    sol = np.linalg.inv(A).dot(B)
    
    sum_r, sum_t = 0,0
    for i in range(len(v)):
        for j in range(len(v)):
            if E>E_c[j,j] and E>E_c[i,i]:
                sum_r += abs(np.sqrt(kk[i]/kk[j])*sol[i*len(v)+j])**2
                sum_t += abs(np.sqrt(kk[i]/kk[j])*sol[i*len(v)+j+len(v)**2])**2
    
    return sum_r, sum_t


#F = float(input("E: "))
#print('{} channels'.format(len(v)))
#print('L = {}'.format(L))
#val = RT(F)
#print('{} {} {}'.format(val[0]/2, val[1]/2, val[0]/2+val[1]/2))


# LOOP
E, Emax = 0.00, 4.50
f = open('transport_harmonic_rashba%1.1f_L%1.1f_delta%1.1f_%dchannels_new.txt' %(alpha_R, L, r0, len(v)), 'w')
f.write('# alpha_R = %1.1f, w0x = %1.1f, w0y = %1.1f, L = %1.1f, r0=%1.1f\n' %(alpha_R, w0x, w0y, L, r0))
f.write('# channels = {}\n'.format(list(v)))

while E < Emax:
    val = RT(E)
    f.write('%3.2f %3.16f %3.16f\n' %(E, val[0], val[1]))
    E += 0.01
f.close()

