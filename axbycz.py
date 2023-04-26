import numpy as np
import GetF, Getc, GetJ, Getp, vlogR, rotationMatrix

def AXBYCZ(gA, gB, gC, RX_init, RY_init, RZ_init):
    N = gA.shape[2]

    RA = gA[:3,:3,:]
    RB = gB[:3,:3,:]
    RC = gC[:3,:3,:]

    tA = np.reshape(gA[:3, 3, :], (3, N))
    tB = np.reshape(gB[:3, 3, :], (3, N))
    tC = np.reshape(gC[:3, 3, :], (3, N))

    r = np.inf
    iter = 0

    while (np.linalg.norm(r) > 1e-10):
        F = GetF(RA, RB, RC, RX_init, RY_init, RZ_init)
        c = Getc(RA, RB, RC, RX_init, RY_init, RZ_init)
        r = np.linalg.solve(F, c)

        RX_init = np.dot(rotationMatrix(r[:3] / np.linalg.norm(r[:3]), np.linalg.norm(r[:3])), RX_init)
        RY_init = np.dot(rotationMatrix(r[3:6] / np.linalg.norm(r[3:6]), np.linalg.norm(r[3:6])), RY_init)
        RZ_init = np.dot(rotationMatrix(r[6:] / np.linalg.norm(r[6:]), np.linalg.norm(r[6:])), RZ_init)

        iter += 1
        if iter >= 100:
            break

    RX_sln = RX_init
    RY_sln = RY_init
    RZ_sln = RZ_init

    J = GetJ(RA, RY_sln, RC)
    p = Getp(RA, RX_sln, RY_sln, tA, tB, tC)

    t_sln = np.linalg.solve(J, p)
    tX_sln = t_sln[:3]
    tY_sln = t_sln[3:6]
    tZ_sln = t_sln[6:]

    gX = np.vstack((np.hstack((RX_sln, tX_sln[:, np.newaxis])), np.array([0, 0, 0, 1])))
    gY = np.vstack((np.hstack((RY_sln, tY_sln[:, np.newaxis])), np.array([0, 0, 0, 1])))
    gZ = np.vstack((np.hstack((RZ_sln, tZ_sln[:, np.newaxis])), np.array([0, 0, 0, 1])))

    errR = np.zeros(N)
    errt = np.zeros(N)

    for i in range(N):
        errg = np.dot(np.dot(gA[:,:,i], gX), np.linalg.inv(np.dot(gY, np.dot(gC[:,:,i], gZ))))
        errR[i] = np.linalg.norm(vlogR(errg[:3,:3]))
        errt[i] = np.linalg.norm(errg[:3,3])

    return gX, gY, gZ, errR, errt, iter

# Define GetF, Getc, GetJ, Getp, and vlogR functions here

# ...

if __name__ == '__main__':
    # Load your gA, gB, gC, RX_init, RY_init, and RZ_init matrices here
    # ...

    gX, gY, gZ, errR, errt, iter = AXBYCZ(gA, gB, gC, RX_init, RY_init, RZ_init)
    
    print("gX:", gX)
    print("gY:", gY)
    print("gZ:", gZ)
    print("errR:", errR)
    print("errt:", errt)
    print("iter:", iter)