import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.spatial.transform import Rotation as R
from roboticstoolbox import DHRobot, RevoluteDH
from axbycz import FindInitialXYZ, AXBYCZ, rotationTheta, rotationMatrix

# Generate data
# Rotation matrix of unknowns: X, Y, Z
simulation_cycle = 10

rotation_err_x = np.zeros(simulation_cycle)
rotation_err_y = np.zeros(simulation_cycle)
rotation_err_z = np.zeros(simulation_cycle)
translation_err_x = np.zeros(simulation_cycle)
translation_err_y = np.zeros(simulation_cycle)
translation_err_z = np.zeros(simulation_cycle)

err_matrix = np.zeros((19, 6))

j = 0

for N_motion in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

    j += 1

    for k in range(simulation_cycle):

        RX_true = R.from_rotvec(np.array([0, 0, pi / 2 + 0.01])).as_matrix()
        RY_true = R.from_rotvec(np.array([0, 0, pi - 0.02])).as_matrix()
        RZ_true = R.from_rotvec(np.array([0, 0, pi / 4 + 0.01])).as_matrix()

        tX_true = np.array([[0], [0], [0.200 - 0.003]])
        tY_true = np.array([[2.000 + 0.010], [0], [0]])
        tZ_true = np.array([[0], [0], [0.100 + 0.002]])

        # big noise
        rotation_noise_scale_A = 0.25 / 180 * pi
        rotation_noise_scale_B = 0.5 / 180 * pi
        rotation_noise_scale_C = 0.25 / 180 * pi

        translation_noise_scale_A = 1.0 / 1000
        translation_noise_scale_B = 2.0 / 1000
        translation_noise_scale_C = 1.0 / 1000

        RA_true = np.zeros((3, 3, N_motion))
        RB_true = np.zeros((3, 3, N_motion))
        RC_true = np.zeros((3, 3, N_motion))
        RA_noise = np.zeros((3, 3, N_motion))
        RB_noise = np.zeros((3, 3, N_motion))
        RC_noise = np.zeros((3, 3, N_motion))

        tA_true = np.zeros((3, N_motion))
        tB_true = np.zeros((3, N_motion))
        tC_true = np.zeros((3, N_motion))
        tA_noise = np.zeros((3, N_motion))
        tB_noise = np.zeros((3, N_motion))
        tC_noise = np.zeros((3, N_motion))

        gA_true = np.zeros((4, 4, N_motion))
        gC_true = np.zeros((4, 4, N_motion))

        gA_noise = np.zeros((4, 4, N_motion))
        gB_noise = np.zeros((4, 4, N_motion))
        gC_noise = np.zeros((4, 4, N_motion))

        # generate a puma 560 robot
        links = [
            RevoluteDH(a=0, alpha=pi / 2, d=0, offset=0),
            RevoluteDH(a=0.4318,alpha=0, d=0, offset=0),
            RevoluteDH(a=0.0203, alpha=-pi / 2, d=0.15005, offset=0),
            RevoluteDH(a=0, alpha=pi / 2, d=0.4318, offset=0),
            RevoluteDH(a=0, alpha=-pi / 2, d=0, offset=0),
            RevoluteDH(a=0, alpha=0, d=0, offset=0),
            ]
        puma = DHRobot(links, name="Puma 560")
        for i in range(N_motion):
            # randomly choose joint angles
            q = np.random.rand(6) * 2 * pi

            # forward kinematics
            T = puma.fkine(q)
            RA_true[:, :, i] = T[:3, :3]
            tA_true[:, i] = T[:3, 3]

            # apply rotation matrices
            RB_true[:, :, i] = RA_true[:, :, i] @ RX_true
            RC_true[:, :, i] = RA_true[:, :, i] @ RY_true

            # apply translations
            tB_true[:, i] = tA_true[:, i] + tX_true.ravel()
            tC_true[:, i] = tA_true[:, i] + tY_true.ravel()

            # add noise
            RA_noise[:, :, i] = RA_true[:, :, i] @ rotationMatrix(rotation_noise_scale_A)
            RB_noise[:, :, i] = RB_true[:, :, i] @ rotationMatrix(rotation_noise_scale_B)
            RC_noise[:, :, i] = RC_true[:, :, i] @ rotationMatrix(rotation_noise_scale_C)

            tA_noise[:, i] = tA_true[:, i] + np.random.randn(3) * translation_noise_scale_A
            tB_noise[:, i] = tB_true[:, i] + np.random.randn(3) * translation_noise_scale_B
            tC_noise[:, i] = tC_true[:, i] + np.random.randn(3) * translation_noise_scale_C

            # construct noisy transformations
            gA_noise[:, :, i] = np.vstack([np.hstack([RA_noise[:, :, i], tA_noise[:, i].reshape(-1, 1)]), [0, 0, 0, 1]])
            gB_noise[:, :, i] = np.vstack([np.hstack([RB_noise[:, :, i], tB_noise[:, i].reshape(-1, 1)]), [0, 0, 0, 1]])
            gC_noise[:, :, i] = np.vstack([np.hstack([RC_noise[:, :, i], tC_noise[:, i].reshape(-1, 1)]), [0, 0, 0, 1]])

# solve AX=XB
RX_est, tX_est = FindInitialXYZ(gA_noise, gB_noise)
RY_est, tY_est = FindInitialXYZ(gA_noise, gC_noise)

# optimization
RX_opt, tX_opt, RY_opt, tY_opt = AXBYCZ(gA_noise, gB_noise, gC_noise, RX_est, tX_est, RY_est, tY_est, 1e-9)

# compute errors
rotation_err_x[k] = rotationTheta(RX_true, RX_opt)
rotation_err_y[k] = rotationTheta(RY_true, RY_opt)
rotation_err_z[k] = rotationTheta(RZ_true, RY_opt @ RX_opt)

translation_err_x[k] = np.linalg.norm(tX_true - tX_opt)
translation_err_y[k] = np.linalg.norm(tY_true - tY_opt)
translation_err_z[k] = np.linalg.norm(tZ_true - tY_opt + tX_opt)

# Compute mean and standard deviation of errors
rotation_err_x_mean = np.mean(rotation_err_x)
rotation_err_x_std = np.std(rotation_err_x)
rotation_err_y_mean = np.mean(rotation_err_y)
rotation_err_y_std = np.std(rotation_err_y)
rotation_err_z_mean = np.mean(rotation_err_z)
rotation_err_z_std = np.std(rotation_err_z)

translation_err_x_mean = np.mean(translation_err_x)
translation_err_x_std = np.std(translation_err_x)
translation_err_y_mean = np.mean(translation_err_y)
translation_err_y_std = np.std(translation_err_y)
translation_err_z_mean = np.mean(translation_err_z)
translation_err_z_std = np.std(translation_err_z)

# Print results
print("Rotation error X (mean, std):", rotation_err_x_mean, rotation_err_x_std)
print("Rotation error Y (mean, std):", rotation_err_y_mean, rotation_err_y_std)
print("Rotation error Z (mean, std):", rotation_err_z_mean, rotation_err_z_std)

print("Translation error X (mean, std):", translation_err_x_mean, translation_err_x_std)
print("Translation error Y (mean, std):", translation_err_y_mean, translation_err_y_std)
print("Translation error Z (mean, std):", translation_err_z_mean, translation_err_z_std)