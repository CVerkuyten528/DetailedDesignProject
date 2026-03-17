import math
from pathlib import Path

import numpy as np

# Load high-resolution calibration once and reuse for all simulations.
_MODULE_DIR = Path(__file__).resolve().parent
_FOV_DIR = _MODULE_DIR / "FOV_Files"

X_angles_highres_deg = np.loadtxt(_FOV_DIR / "horizontal_angles_highres_Scipy_Linear.txt", delimiter="\t")
Y_angles_highres_deg = np.loadtxt(_FOV_DIR / "vertical_angles_highres_Scipy_Linear.txt", delimiter="\t")
X_angles_highres = np.radians(X_angles_highres_deg).astype(np.float32, copy=False)
Y_angles_highres = np.radians(Y_angles_highres_deg).astype(np.float32, copy=False)


def spherical_separation(a1, b1, a2, b2):
    """
    Compute the spherical angular separation between two directions given by:
    (a1, b1) and (a2, b2), using the spherical law of cosines.
    All angles are in radians.
    """
    return math.acos(math.sin(b1) * math.sin(b2) + math.cos(b1) * math.cos(b2) * math.cos(a2 - a1))


def rotate_point(x, y, angle, origin=(0, 0)):
    """
    Rotate a point (x, y) around a given origin by an angle in radians.

    Parameters:
        x (float): X-coordinate of the point to rotate.
        y (float): Y-coordinate of the point to rotate.
        angle (float): Rotation angle in radians.
        origin (tuple of float, optional): The origin point (ox, oy) to rotate around. Defaults to (0, 0).

    Returns:
        The rotated point's coordinates x,y
    """
    ox, oy = origin

    x_translated = x - ox
    y_translated = y - oy

    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    x_rotated = x_translated * cos_theta - y_translated * sin_theta
    y_rotated = x_translated * sin_theta + y_translated * cos_theta

    x_new = x_rotated + ox
    y_new = y_rotated + oy

    return x_new, y_new


def simulate_earth_highres(pitch, roll, yaw, altitude_km=550.0, R_e=6378.14, earth_temp=45.0, space_temp=25.0):
    """
    Generate a 384x512 thermal image simulating Earth.
    The simulation uses upscaled fisheye calibration arrays so that each pixel has its
    own (non-uniform) angular direction.

    Parameters:
      pitch, roll, yaw : camera tilt offsets (in degrees)
      altitude_km : satellite altitude [km]
      R_e : Earth's radius [km]
      earth_temp : temperature assigned to Earth [deg C]
      space_temp : temperature for space [deg C]

    Returns:
      data: a 384x512 array of simulated thermal data.
    """
    X_angles = X_angles_highres.copy()
    Y_angles = Y_angles_highres.copy()

    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)

    earth_half_angle = math.asin(R_e / (R_e + altitude_km))

    high_res_height, high_res_width = 384, 512
    data = np.full((high_res_height, high_res_width), space_temp, dtype=float)

    Y_angles += roll

    origin = (
        X_angles[int(high_res_height / 2), int(high_res_width / 2)],
        Y_angles[int(high_res_height / 2), int(high_res_width / 2)],
    )

    X_angles, Y_angles = rotate_point(X_angles, Y_angles, pitch, origin)

    alpha_loc = X_angles + yaw
    beta_loc = Y_angles

    cos_dist_to_center = np.cos(beta_loc) * np.cos(alpha_loc)
    inside_earth = cos_dist_to_center > math.cos(earth_half_angle)
    data = np.where(inside_earth, earth_temp, space_temp)

    return data


def downsample_image(image, new_shape):
    """
    Downsample a 2D image to new_shape by averaging non-overlapping blocks.
    Assumes that the original shape dimensions are divisible by the new shape dimensions.
    """
    old_rows, old_cols = image.shape
    new_rows, new_cols = new_shape
    factor_row = old_rows // new_rows
    factor_col = old_cols // new_cols
    downsampled = image.reshape(new_rows, factor_row, new_cols, factor_col).mean(axis=(1, 3), dtype=np.float32)
    return downsampled


def downsample_stack(images, new_shape):
    """
    Downsample a stack of 2D images with shape (N, H, W) to
    (N, new_rows, new_cols) by block averaging.
    """
    n, old_rows, old_cols = images.shape
    new_rows, new_cols = new_shape
    factor_row = old_rows // new_rows
    factor_col = old_cols // new_cols
    return images.reshape(n, new_rows, factor_row, new_cols, factor_col).mean(axis=(2, 4), dtype=np.float32)


def simulate_earth_batch(
    pitch,
    roll_values,
    yaw,
    altitude_km=550.0,
    R_e=6378.14,
    earth_temp=45.0,
    space_temp=25.0,
    plot=False,
    verbose=False,
):
    """
    Vectorized simulator for one pitch and multiple rolls.

    Returns:
      data_stack: shape (N_roll, 24, 32)
    """
    rolls = np.asarray(roll_values, dtype=np.float32).reshape(-1)
    if rolls.size == 0:
        return np.empty((0, 24, 32), dtype=np.float32)

    if verbose:
        print(
            "\n",
            (
                f"Simulating Earth center with camera tilt: pitch = {pitch} deg, "
                f"roll range = [{rolls.min():.1f}, {rolls.max():.1f}] deg, yaw = {yaw} deg"
            ),
        )

    pitch_rad = np.float32(np.radians(pitch))
    yaw_rad = np.float32(np.radians(yaw))
    roll_rad = np.radians(rolls).astype(np.float32, copy=False).reshape(-1, 1, 1)

    earth_half_angle = np.float32(math.asin(R_e / (R_e + altitude_km)))
    earth_cos = np.float32(math.cos(float(earth_half_angle)))

    high_res_height, high_res_width = X_angles_highres.shape
    center_row = int(high_res_height / 2)
    center_col = int(high_res_width / 2)
    x_center = X_angles_highres[center_row, center_col]
    y_center = Y_angles_highres[center_row, center_col]

    x_rel = X_angles_highres - x_center
    y_rel = Y_angles_highres - y_center

    cos_pitch = np.float32(math.cos(float(pitch_rad)))
    sin_pitch = np.float32(math.sin(float(pitch_rad)))

    # Batched equivalent of:
    #   Y += roll
    #   rotate_point(X, Y, pitch, origin=(x_center, y_center + roll))
    # Matching the old implementation keeps X independent of roll and shifts Y by roll.
    X_angles_rot = x_rel * cos_pitch - y_rel * sin_pitch + x_center
    Y_angles_rot = x_rel * sin_pitch + y_rel * cos_pitch + y_center + roll_rad

    alpha_loc = X_angles_rot + yaw_rad
    beta_loc = Y_angles_rot
    inside_earth = (np.cos(beta_loc) * np.cos(alpha_loc)) > earth_cos

    earth_fraction = downsample_stack(inside_earth.astype(np.float32, copy=False), (24, 32))
    data_stack = np.float32(space_temp) + np.float32(earth_temp - space_temp) * earth_fraction

    if plot:
        import matplotlib.pyplot as plt

        high_res_data = np.where(inside_earth[0], earth_temp, space_temp)
        low_res_data = data_stack[0]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("High Resolution (384x512)")
        plt.imshow(high_res_data, cmap="hot", interpolation="nearest")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Downsampled to 24x32")
        plt.imshow(low_res_data, cmap="hot", interpolation="nearest")
        plt.colorbar()

        plt.show()

    return data_stack


def simulate_earth(
    pitch,
    roll,
    yaw,
    altitude_km=550.0,
    R_e=6378.14,
    earth_temp=45.0,
    space_temp=25.0,
    plot=False,
    verbose=True,
):
    # Print is now optional so batch generation can run fast.
    if verbose:
        print("\n", f"Simulating Earth center with camera tilt: pitch = {pitch} deg, roll = {roll} deg, yaw = {yaw} deg")

    data_stack = simulate_earth_batch(
        pitch=pitch,
        roll_values=[roll],
        yaw=yaw,
        altitude_km=altitude_km,
        R_e=R_e,
        earth_temp=earth_temp,
        space_temp=space_temp,
        plot=plot,
        verbose=False,
    )
    return data_stack[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pitch = 35
    roll = 100
    yaw = 0

    data = simulate_earth(pitch, roll, yaw, plot=True)
