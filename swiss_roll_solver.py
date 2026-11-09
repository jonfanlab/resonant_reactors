import argparse
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


epsilon = 8.85E-12 # F/m
mu_0 = 4E-7 * np.pi # H/m


# Integrates the square of the current across a turn
def P_fn(x, J_l_prefac, J_r_prefac, delta, t):
    J_l = J_l_prefac * np.exp(-x / delta) * np.exp(-1j*x / delta)
    J_r = - J_r_prefac * np.exp((x - t) / delta) * np.exp(1j*(x - t) / delta)
    return np.abs(J_l + J_r)**2 / 2 # Factor of two converts from peak to rms

def solve_swiss_roll(rs, thick, sigma, loss_tangent=0):
    """Solves for resonance frequency and Q factor of swiss roll, accepting a list of radii.

    Args:
        rs: a list of the inner diameters of the cylinders after discretization [m]
        thick: metal thickness [m]
        N: number of turns
        sigma: conductivity [S/m]
    """
    N = len(rs)
    # Calculate capacitances based upon concentric cylinders
    Cs = []
    for ii in range(N-1):
        C = 2 * epsilon * np.pi / np.log(rs[ii+1] / (rs[ii] + thick))
        Cs.append(C)

    # Calculate total non-metal areas (for flux) and annular ring areas (for energy)
    ls = []
    for ii in range(0, N-1):
        ls.append(((rs[ii] + thick) + rs[ii+1])/2)

    ring_areas = [np.pi * (ls[0]**2 - ((rs[0] + thick)**2 - rs[0]**2))]
    tot_areas = [ring_areas[0]]
    for ii in range(1, N-1):
        ring_area = np.pi * (ls[ii]**2-ls[ii-1]**2 - ((rs[ii] + thick)**2 - (rs[ii])**2))
        ring_areas.append(ring_area)
        tot_areas.append(tot_areas[-1] + ring_areas[-1])

    # Iteratively solve expression for currents from an initial guess of uniform
    Is = np.ones(N-1)
    new_Is = np.zeros(N-1)
    for iteration in range(35):
        new_Is = np.zeros(N-1)
        for ii in range(N-2, 0-1, -1):
            prefac = mu_0 * Cs[ii]
            sum_1 = tot_areas[ii] * sum(Is[ii:N-1])
            sum_2 = 0
            for jj in range(0, ii):
                sum_2 += tot_areas[jj] * Is[jj]
            new_Is[ii] = prefac * (sum_1 + sum_2)
        # Keep it so outside turn has current of 1 for convenience
        Is = new_Is / new_Is[-1]
    res_freq = 1 / np.sqrt(new_Is[-1]) / (2 * np.pi)

    # Compute stored energy from electric field when current in outside turn is 1A
    E_field_storage = 0
    dielectric_Ps = []
    for I, C in zip(Is, Cs):
        V = I / (res_freq * 2 * np.pi * C)
        E_field_storage += C * V**2 / 2
        # Loss tangent = 1/Q = power / ( 2 * pi * energy stored * freq)
        dielectric_Ps.append(2 * np.pi * loss_tangent * C * V**2 / 2 * res_freq)

    # Compute magnetic fields and stored energy from magnetic fields when current in outside turn is 1A
    Bs = np.zeros(N)
    B_field_storage = 0
    for ii in range(N-1):
        Bs[ii] = sum(Is[ii:]) * mu_0
        B_field_storage += ring_areas[ii] * Bs[ii]**2 / (2 * mu_0)

    # Calc skin depth and prefactor to relate current to magnetic field
    skin_depth = 503 / np.sqrt(res_freq * sigma)
    alpha = res_freq * 2 * np.pi * sigma * skin_depth / np.sqrt(2)

    # calc out the powers
    t_max = thick / skin_depth
    Ps = []
    normalized_dielectric_Ps = []
    overall_P = 0
    for ii in range(N-1):
        # Calculate the prefactors to the current exponential decay/growth from B field
        J_l_prefac = alpha * (Bs[ii] - Bs[ii+1] * np.exp(-(1+1j)*t_max)) / (1-np.exp(-2*(1+1j)*t_max))
        J_r_prefac = alpha * (Bs[ii+1] - Bs[ii] * np.exp(-(1+1j)*t_max)) / (1-np.exp(-2*(1+1j)*t_max))
        # Integrate power density across thickness of metal
        Ps.append(quad(P_fn, 0, thick, args=(J_l_prefac, J_r_prefac, skin_depth, thick))[0] / sigma)
        normalized_dielectric_Ps.append(dielectric_Ps[ii] / (2 * np.pi * (rs[ii+1] + rs[ii]) / 2))
        # Power is [circumference of ring * power per unit length of ring]
        overall_P += Ps[-1] * (rs[ii+1] + rs[ii]) / 2 * 2 * np.pi + dielectric_Ps[ii]

    # Compute energy dissipated per cycle
    energy_per_cyc = overall_P / res_freq
    Q_factor = 2 * np.pi* E_field_storage / energy_per_cyc

    return Q_factor, res_freq, skin_depth, Is, Bs, Ps, normalized_dielectric_Ps


def solve_swiss_roll_uniform(r, delta_r, thick, N, sigma=1e7, loss_tangent=0):
    """Solves for resonance frequency and Q factor of a uniform swiss roll.

    Just a wrapper around solve_swiss_roll for the uniform case

    
    Args:
        r: inner radius of swiss roll [m]
        delta_r: change in diameter between consecutive layers [m]
        thick: metal thickness [m]
        N: number of turns
        sigma: conductivity [S/m]
        loss_tangent: the loss tangent of the material
    """
    rs = [r + (ii + 1/2) * delta_r for ii in range(N)]

    return solve_swiss_roll(rs, thick, sigma, loss_tangent)


if __name__ == '__main__':
    # Random plotting code, stuff down here doesn't really mean anything
    parser = argparse.ArgumentParser()
    parser.add_argument('--r',  default=.003)
    parser.add_argument('--delta_r', default=.00175)
    parser.add_argument('--thick', default=None)
    parser.add_argument('--max_thick', default=1e-3)
    parser.add_argument('--min_thick', default=1e-6)
    parser.add_argument('--sigma', default=1e6)
    parser.add_argument('--N', default=25)

    args = parser.parse_args()
    N = int(args.N)
    r = float(args.r)
    delta_r = float(args.delta_r)
    sigma = float(args.sigma)
        
    I_fig, I_ax = plt.subplots()
    I_ax.set_xlabel('Turn Number')
    I_ax.set_ylabel('Normalized Charging Current [u.l.]')

    B_fig, B_ax = plt.subplots()
    B_ax.set_xlabel('Turn Number')
    B_ax.set_ylabel('Normalized Magnetic Field [u.l.]')

    P_fig, P_ax = plt.subplots()
    P_ax.set_xlabel('Turn Number')
    P_ax.set_ylabel('Normalized Power [u.l.]')

    if args.thick:
        thicknesses = [float(args.thick)]
    else:
        thicknesses = np.logspace(np.log10(float(args.min_thick)), np.log10(float(args.max_thick)))

    Q_factors = []
    res_freqs = []
    skin_depths = []
    pwr_stds = []
    rs = [(r + (ii+1) * delta_r)**2 for ii in range(N - 1)]
    plot_color = len(thicknesses) - 1
    # Iterate through thicknesses and calc out the powers
    for ii, thick in enumerate(thicknesses):

        Q_factor, res_freq, skin_depth, Is, Bs, Ps, _ = solve_swiss_roll_uniform(r, delta_r, thick, N, sigma)

        print('thickness ', thick, 'skin_depth', skin_depth, 'Resonance frequency:', res_freq)

        if ii % 5 == 0:
            # Plot charging current curve
            I_ax.plot(Is, label=f't={int(thick * 1e6)}[$\mu$m]', color=plt.cm.viridis(ii/plot_color))

            # Plot B field curve
            B_ax.plot(Bs / max(Bs), label=f't={int(thick * 1e6)}[$\mu$m]', color=plt.cm.viridis(ii/plot_color))

            # Plot Ps
            P_ax.plot(np.array(Ps) / max(Ps), label=f't={int(thick * 1e6)}[$\mu$m]', color=plt.cm.viridis(ii/plot_color))

        Q_factors.append(Q_factor)
        res_freqs.append(res_freq)
        skin_depths.append(skin_depth)

    print('highest Q factor: ', max(Q_factors), '  at thickness: ', thicknesses[np.argmax(Q_factors)])

    if not args.thick:
        # Plot Q factors
        plt.figure()
        plt.plot(np.array(thicknesses) * 1000, Q_factors)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Metal Thickness [mm]')
        plt.ylabel('Q Factor [u.l.]')

    I_ax.legend()
    B_ax.legend()
    P_ax.legend()

    plt.show()
