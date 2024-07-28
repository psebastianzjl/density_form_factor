import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plot_spectra import plot_polar_signal

tsteps = np.arange(0, 1, 0.5)
plane = str(sys.argv[1]) # plane to which the X-ray pulse is supposed to be perpendicular
num_traj = 200
levels = 111 # Number of levels for contourplot

def plot_polar_signal(r, phi, values, levels, vmin=-1.0, vmax=1.0, color_code='rainbow', outname='polar_spectrum.png'):
    """

    Parameters
    ----------
    r : list or array, float
    Radius of polar coordinates in Angstrom
    phi : list or array, float
    Angles of polar coordiantes
    values : list or array, float
    Corresponding values for the intensities
    levels : int
    Number of levels in the color scheme
    outname : str
    Name of the output-file

    Returns
    -------
    PNG-file containing the spectrum.
    """
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    c = ax.contourf(phi, r, values, np.linspace(vmin, vmax, levels, endpoint=True), cmap=color_code)
    cbar = fig.colorbar(c, ax=ax, pad=0.07)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks(np.linspace(vmin, vmax, 5, endpoint=True))
    ax.set_rticks([])  # Radial ticks
    plt.tight_layout()
    plt.savefig(outname, bbox_inches='tight', dpi=400)
    plt.close()


def read_density(filename):
    """
    Function that reads in the output of Turbomole in the xyz format
    Parameters
    ----------
    filename : str
    Name of the file from which the density is to be read

    Returns
    -------
    Gridpoints and the density
    """
    density = []
    grid_start = []
    grid_delta = []
    grid_points = []
    with open(filename, 'r') as f:
        for _ in range(4):
            next(f)
        for line in f:
            if line.startswith('#grid1'):
                grid_start += [float(line.split()[2])]
                grid_delta += [float(line.split()[4])]
                grid_points += [int(line.split()[6])]
            if line.startswith('#grid2'):
                grid_start += [float(line.split()[2])]
                grid_delta += [float(line.split()[4])]
                grid_points += [int(line.split()[6])]
            if line.startswith('#grid3'):
                grid_start += [float(line.split()[2])]
                grid_delta += [float(line.split()[4])]
                grid_points += [int(line.split()[6])]
            if line.startswith(' ') or line.startswith('-'):
                if line.rstrip():
                    density += [float(line.split()[-1])]
        density = np.reshape(density, (grid_points[0], grid_points[1], grid_points[2]))
    return grid_start, grid_delta, grid_points, density


def transformation(axis1, axis2, density, radius=9.666, axis=2):
    """
    Function that transforms the density into the form factor
    Parameters
    ----------
    axis1 : float, array or list
    Grid along this axis
    axis2 : float, array or list
    Grid along this axis
    density : float, array or list
    Electron density or the density difference
    radius : float
    Maximum value of q
    axis : int
    Axis to which the pulse is parallel

    Returns
    -------
    Values for the form factor, q and phi
    """
    q_phi = np.linspace(start=0, stop=2 * np.pi, num=len(axis2))
    q_rad = np.linspace(start=0, stop=radius, num=len(axis1))
    density = density - (np.sum(density)/(len(axis1) * len(axis2) * np.shape(density)[axis]))
    density = np.sum(density, axis=axis)
    f0 = np.zeros_like(density, dtype=complex)
    for i in range(len(axis1)):
        for k in range(len(axis2)):
            temp = 0
            for m in range(len(axis1)):
                for n in range(len(axis2)):
                    temp += density[m, n] * np.exp((np.sin(q_phi[k]) * q_rad[i] * axis2[n] + np.cos(q_phi[k]) * q_rad[i] * axis1[m])*1j)
            f0[i, k] = temp
    return f0, q_rad, q_phi

for time in tsteps:
    print('Current Time: ', time)
    filename = 'el_dens_{}.dtx'.format(time)
    if os.path.isfile(filename):
        grid_start, grid_delta, grid_points, density = read_density(filename)
        x = np.linspace(start=grid_start[0], stop=grid_start[0] + grid_delta[0] * grid_points[0], num=grid_points[0])  # , endpoint=False)
        y = np.linspace(start=grid_start[1], stop=grid_start[1] + grid_delta[1] * grid_points[1], num=grid_points[1])  # , endpoint=False)
        z = np.linspace(start=grid_start[2], stop=grid_start[2] + grid_delta[2] * grid_points[2], num=grid_points[2])  # , endpoint=False)
        if plane == 'xy':
            f0, q_rad, q_phi = transformation(x, y, density, axis=2)
        if plane == 'yz':
            f0, q_rad, q_phi = transformation(y, z, density, axis=0)
        if plane == 'xz':
            f0, q_rad, q_phi = transformation(x, z, density, axis=1)
        name = str(int(float(time) / 0.5)).zfill(4)
        plot_polar_signal(q_rad, q_phi, np.absolute(f0) ** 2 / np.amax(np.absolute(f0) ** 2), levels, vmin=0, outname='ued_pyr_t1_{}_absolute_13A_{}'.format(plane, name))
        plot_polar_signal(q_rad, q_phi, np.real(f0) / np.amax(np.absolute(np.real(f0))), levels, vmin=-1.0, vmax=1.0, outname='ued_pyr_t1_{}_real_13A_{}'.format(plane, name))
        plot_polar_signal(q_rad, q_phi, np.imag(f0) / np.amax(np.absolute(np.imag(f0))), levels, vmin=-1.0, vmax=1.0, outname='ued_pyr_t1_{}_imag_13A_{}'.format(plane, name))
        plot_polar_signal(q_rad, q_phi, np.angle(f0), levels, vmin=-np.pi, vmax=np.pi, outname='ued_pyr_t1_{}_phase_13A_{}'.format(plane, name), color_code='hsv')
    else: continue
