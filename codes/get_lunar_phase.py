import datetime
import glob
import os
import shutil


def get_hour(date=datetime.datetime.now()):
    """For a given date, find the hours elapsed since the start of the year"""
    start_of_year = datetime.datetime(date.year, 1, 1, tzinfo=datetime.timezone.utc)
    hours_elapsed = (date - start_of_year).total_seconds() / 3600
    return int(hours_elapsed)


def rename_file(initial_dir=None, final_dir=None, current_time=None):
    """Find the name of the file in the initial directory corresponding to the hours elapsed, rename
    it as current_file and copy it to the final directory"""

    # Get the hours elapsed since the start of the year
    hours_elapsed = get_hour(current_time)
    hours_elapsed = str(hours_elapsed).zfill(5)

    new_file_name = "moon_phase_current_time.png"
    file_name = glob.glob(os.path.join(initial_dir, f"*{hours_elapsed}*"))[0]

    final_dir = os.path.expanduser(final_dir)
    shutil.copy(file_name, os.path.join(final_dir, new_file_name))


if __name__ == "__main__":
    current_time = datetime.datetime.now(datetime.timezone.utc)

    initial_dir = "../data/frames"
    final_dir = "~/Dropbox/rt_sw/"
    final_dir = os.path.expanduser(final_dir)
    current_file = rename_file(initial_dir, final_dir, current_time)
    # Get the right frame corresponding to the hours elapsed from the frames_folder
    hours_elapsed = get_hour(current_time)

    print(
        f"Hours elapsed since the start of the year for {current_time}: {hours_elapsed}"
    )
    # Example output: Hours elapsed since the start of the year: 1211.25
