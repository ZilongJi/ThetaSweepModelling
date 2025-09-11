import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_theta_modulation(ax, time_steps, theta_phase, theta_modulation, add_lines=True, atol=1e-2):
    """Plot theta modulation trace into a given axis."""
    ax.plot(time_steps, theta_modulation, color="black", lw=1)
    ax.set_yticks([])

    if add_lines:
        zero_phase_index = np.where(np.isclose(theta_phase, 0, atol=atol))[0]
        for i in zero_phase_index:
            ax.axvline(x=time_steps[i], color="grey", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_ylabel(r"$\theta$ mdulation")
    sns.despine(ax=ax)

    return ax

def plot_population_activity(ax, time_steps, theta_phase, net_activity, direction, add_lines=True, atol=1e-2):
    """Plot HD network population activity + direction trace."""
    im = ax.imshow(
        net_activity.T * 100,
        aspect="auto",
        extent=[time_steps[0], time_steps[-1], -np.pi, np.pi],
        cmap="jet",
        origin="lower",
    )
    
    # find the jump points where the difference between two adjacent points is greater than pi
    jumps = np.where(np.abs(np.diff(direction)) > np.pi)[0]
    # set the jump points to NaN for plotting
    direction_plot = direction.copy()
    direction_plot[jumps + 1] = np.nan    
    ax.plot(time_steps, direction_plot, color="white", lw=3)
    
    if add_lines:
        zero_phase_index = np.where(np.isclose(theta_phase, 0, atol=atol))[0]
        for i in zero_phase_index:
            ax.axvline(x=time_steps[i], color="grey", linestyle="--", linewidth=1, alpha=0.5)    

    ax.set_yticks([-np.pi, np.pi])
    ax.set_yticklabels([0, 360])
    ax.set_ylabel("Direction(°)")

    sns.despine(ax=ax)
    return ax

def plot_angular_velocity(ax, time_steps, velocity):
    """Plot angular speed trace into a given axis."""

    ax.fill_between(time_steps, velocity, 0, color="lightgrey", linewidth=1, alpha=0.9)
    
    vmax = 1.2 * np.max(np.abs(velocity))
    ax.set_ylim([-vmax, vmax])
    
    ax.set_ylabel("Ang. Vel.\n(rad/s)")
    ax.set_xlabel("Time (s)")
    
    sns.despine(ax=ax)
    return ax

def plot_phase_coding(ax, cell_activity, direction, theta_phase, cell_index=None, cell_num=None):
    """
    Plot phase coding of a cell, with optional recentering to its preferred direction.
    """
    # recenter if cell_index and cell_num provided
    if cell_index is not None and cell_num is not None:
        pref_dirs = np.linspace(-np.pi, np.pi, cell_num, endpoint=False)
        pref_dir = pref_dirs[cell_index]
        direction = (direction - pref_dir + np.pi) % (2*np.pi) - np.pi

    spike_direction, spike_phase = [], []
    for i in range(len(direction)):
        r = cell_activity[i]
        spikes = np.random.poisson(r, 1)
        if spikes > 0:
            spike_direction.append(direction[i])
            spike_phase.append(theta_phase[i])

    spike_direction = np.array(spike_direction)
    spike_phase = np.array(spike_phase)

    ax.scatter(spike_direction, spike_phase, s=5, color='black', alpha=0.8)
    ax.scatter(spike_direction, spike_phase + 2*np.pi, s=5, color='black', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Preferred Dir')

    ax.set_xlabel('Direction (recentered)', fontsize=12)
    ax.set_ylabel('Theta phase', fontsize=12)
    ax.set_xlim(-np.pi*2/3, np.pi*2/3)
    ax.set_xticks([-np.pi*2/3, np.pi*2/3])
    ax.set_xticklabels([0, 1])
    ax.set_ylim(-np.pi, 3*np.pi)
    ax.set_yticks([-np.pi, np.pi, 3*np.pi])
    ax.set_yticklabels([r'$0$', r'$2\pi$', r'$4\pi$'])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Single-cell theta phase coding', fontsize=14)
    return ax