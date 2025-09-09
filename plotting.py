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

    ax.set_ylabel(r"$\theta$", rotation=0)
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
    ax.plot(time_steps, direction_plot, color="#F18D00", lw=2)
    
    if add_lines:
        zero_phase_index = np.where(np.isclose(theta_phase, 0, atol=atol))[0]
        for i in zero_phase_index:
            ax.axvline(x=time_steps[i], color="grey", linestyle="--", linewidth=1, alpha=0.5)    

    ax.set_yticks([-np.pi, np.pi])
    ax.set_yticklabels([0, 360])
    ax.set_ylabel("Direction(°)")

    sns.despine(ax=ax)
    return ax

def plot_angular_speed(ax, time_steps, speed):
    """Plot angular speed trace into a given axis."""


    ax.fill_between(time_steps, speed, color="lightgrey", linewidth=1, alpha=0.9)
    ax.set_ylim([0, 1.2*np.max(speed)])
    ax.set_xlim([0, time_steps[-1]])
    ax.set_xticks([0, time_steps[-1]])
    ax.set_xticklabels([0,  int(time_steps[-1])/1000])

    ax.set_ylabel("Ang. Speed\n(rad/s)")
    ax.set_xlabel("Time (s)")

    sns.despine(ax=ax)
    return ax