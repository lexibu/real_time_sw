#!/home/vetinari/.cache/pypoetry/virtualenvs/codes-fO0b3aYA-py3.10/bin/python

import sys


def progress_bar(progress, total):
    """
    Function to display a progress bar in the terminal

    Parameters
    ----------
    progress : int
        The current progress value
    total : int
        The total number of steps to be completed

    Returns
    -------
    None
    """
    bar_length = 50  # Length of the progress bar
    block = int(round(bar_length * progress / total))
    filled_color = "\033[42m"  # Green background
    reset_color = "\033[0m"  # Reset color
    progress_display = (
        filled_color + " " * block + reset_color + "-" * (bar_length - block)
    )
    text = f"\rNext update in : [{progress_display}] {total - progress} seconds"
    sys.stdout.write(text)
    sys.stdout.flush()


def update_progress_bar(sc, current_step, total_steps):
    """
    Function to update the progress bar in the terminal at regular intervals using the sched module
    in Python standard library to schedule the next update of the progress bar at regular intervals
    of time until the progress is complete and then print

    Parameters
    ----------
    sc : sched.scheduler
        The scheduler object
    current_step : int
        The current step in the progress
    total_steps : int
        The total number of steps to be completed

    Returns
    -------
    None
    """
    progress_bar(current_step, total_steps)
    if current_step < total_steps:
        # Schedule the next update
        sc.enter(
            52 / total_steps,
            1,
            update_progress_bar,
            (sc, current_step + 1, total_steps),
        )
    # else:
    #     # Print a new line when the progress is complete
    #     print("")
