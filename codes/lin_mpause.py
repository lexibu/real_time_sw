import sched
import time
import sys

# Create a scheduler
s = sched.scheduler(time.time, time.sleep)

# Total duration in seconds
total_time = 5

# Number of steps (from 0 to 100)
steps = 100

# Time per step
time_per_step = total_time / steps


def progress_bar(progress, total):
    bar_length = 50  # Length of the progress bar
    block = int(round(bar_length * progress / total))
    filled_color = "\033[42m"  # Green background
    reset_color = "\033[0m"  # Reset color
    progress_display = (
        filled_color + " " * block + reset_color + "-" * (bar_length - block)
    )
    text = f"\rProgress: [{progress_display}] {progress}/{total}"
    sys.stdout.write(text)
    sys.stdout.flush()


def update_progress_bar(sc, current_step, total_steps):
    progress_bar(current_step, total_steps)
    if current_step < total_steps:
        # Schedule the next update
        sc.enter(
            time_per_step, 1, update_progress_bar, (sc, current_step + 1, total_steps)
        )
    # else:
    #     # Print a new line when the progress is complete
    #     print("")


# Your existing scheduled task (example function)
def plot_figures_ace_1day(sc):
    # Your plotting code here
    print("\nPlotting figures...")  # Example action
    s.enter(0, 1, update_progress_bar, (sc, 0, steps))

    # Reschedule the task if needed (e.g., every 60 seconds)
    s.enter(5, 2, plot_figures_ace_1day, (sc,))


# Initial call to start the progress bar
# s.enter(0, 1, update_progress_bar, (s, 0, steps))

# Schedule your existing task
s.enter(0, 1, plot_figures_ace_1day, (s,))

# Run the scheduler
s.run()

"""

import time
import sys

# Total duration in seconds
total_time = 10

# Number of steps (from 0 to 100)
steps = 100

# Time per step
time_per_step = total_time / steps


def progress_bar(progress, total):
    bar_length = 50  # Length of the progress bar
    block = int(round(bar_length * progress / total))
    filled_color = "\033[42m"  # Green background
    reset_color = "\033[0m"  # Reset color
    progress_display = (
        filled_color + " " * block + reset_color + "-" * (bar_length - block)
    )
    text = f"\rProgress: [{progress_display}] {progress}/{total}"
    sys.stdout.write(text)
    sys.stdout.flush()


for i in range(steps + 1):
    progress_bar(i, steps)
    time.sleep(time_per_step)

print()  # Move to the next line after the progress bar is complete
"""
