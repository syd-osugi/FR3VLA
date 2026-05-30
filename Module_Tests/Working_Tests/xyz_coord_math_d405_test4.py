###########################
# D405 XYZ coordinate math test.
#
# This mirrors xyz_coord_math_d435_test3.py, but uses the close-range D405 camera
# config. The D405 has a different resolution and working distance from the
# D435, so it gets its own output folder and marked debug image.
###########################

from pathlib import Path

from _working_test_utils import add_working_to_path

add_working_to_path()

import config as cfg
from _xyz_coord_math_common import run_xyz_coord_math_test


def main():
    run_xyz_coord_math_test(
        camera_name="D405",
        serial_number=cfg.D405_SERIAL,
        resolution=cfg.D405_RESOLUTION,
        fps=cfg.CAMERA_FPS,
        script_name=Path(__file__).stem,
    )


if __name__ == "__main__":
    main()
