from attr import asdict
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

__all__ = ['streamplot']


def streamplot(xy, uv, nn, wd, dp, start_point):

    x, y = xy
    u, v = uv
    
    grid = Grid(x, y)
    dmap = DomainMap(grid)

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)

    integrate = get_integrator(u, v, dmap, nn, wd, dp)
    sp = np.asanyarray(start_point, dtype=float).copy()

    xs = sp[0]
    ys = sp[1]
    # Check if start_points are outside the data boundaries
    if not (grid.x_origin <= xs <= grid.x_origin + grid.width and
            grid.y_origin <= ys <= grid.y_origin + grid.height):
        raise ValueError("Starting point ({}, {}) outside of data "
                         "boundaries".format(xs, ys))

    xs -= grid.x_origin
    ys -= grid.y_origin

    xg, yg = dmap.data2grid(xs, ys)

    xg = np.clip(xg, 0, grid.nx - 1)
    yg = np.clip(yg, 0, grid.ny - 1)

    trajectory = integrate(xg, yg)

    return np.divide(trajectory, [dmap.x_data2grid,  dmap.y_data2grid])

# Reflection definition
# ========================

def reflect(nn, wd, xp, yp, dp, up, vp):

    nx, ny = nn
    
    nx = interpgrid(nx, xp, yp)
    ny = interpgrid(ny, xp, yp)
    wall_distance = interpgrid(wd, xp, yp)
    
    if (dp/2) < wall_distance:
        if (up * nx + vp * ny) < 0:
            
            up = -(2*vp*nx*ny-up*(ny**2-nx**2))
            vp = -(2*up*nx*ny+vp*(ny**2-nx**2))
         
    return up, vp


# Coordinate definitions
# ========================

class DomainMap:
    def __init__(self, grid):
        self.grid = grid
        
        self.x_data2grid = 1. / grid.dx
        self.y_data2grid = 1. / grid.dy

    def data2grid(self, xd, yd):
        return xd * self.x_data2grid, yd * self.y_data2grid

    def grid2data(self, xg, yg):
        return xg / self.x_data2grid, yg / self.y_data2grid


class Grid:
    def __init__(self, x, y):

        if np.ndim(x) == 1:
            pass
        elif np.ndim(x) == 2:
            x_row = x[0]
            if not np.allclose(x_row, x):
                raise ValueError("The rows of 'x' must be equal")
            x = x_row
        else:
            raise ValueError("'x' can have at maximum 2 dimensions")

        if np.ndim(y) == 1:
            pass
        elif np.ndim(y) == 2:
            yt = np.transpose(y)  # Also works for nested lists.
            y_col = yt[0]
            if not np.allclose(y_col, yt):
                raise ValueError("The columns of 'y' must be equal")
            y = y_col
        else:
            raise ValueError("'y' can have at maximum 2 dimensions")

        if not (np.diff(x) > 0).all():
            raise ValueError("'x' must be strictly increasing")
        if not (np.diff(y) > 0).all():
            raise ValueError("'y' must be strictly increasing")

        self.nx = len(x)
        self.ny = len(y)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.x_origin = x[0]
        self.y_origin = y[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]

        if not np.allclose(np.diff(x), self.width / (self.nx - 1)):
            raise ValueError("'x' values must be equally spaced")
        if not np.allclose(np.diff(y), self.height / (self.ny - 1)):
            raise ValueError("'y' values must be equally spaced")

    @property
    def shape(self):
        return self.ny, self.nx

    def within_grid(self, xi, yi):
        """Return whether (*xi*, *yi*) is a valid index of the grid."""
        # Note that xi/yi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since *xi* can be `self.nx - 1 < xi < self.nx`
        return 0 <= xi <= self.nx - 1 and 0 <= yi <= self.ny - 1


class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass


# Integrator definitions
# =======================

def get_integrator(u, v, dmap, nn, wd, dp):

    # rescale velocity onto grid-coordinates for integrations.
    u, v = dmap.data2grid(u, v)

    # speed (path length) will be in axes-coordinates
    u_ax = u / (dmap.grid.nx - 1)
    v_ax = v / (dmap.grid.ny - 1)
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2)

    def forward_time(xi, yi):
        if not dmap.grid.within_grid(xi, yi):
            raise OutOfBounds
        ds_dt = interpgrid(speed, xi, yi)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui = interpgrid(u, xi, yi)
        vi = interpgrid(v, xi, yi)

        ui, vi = reflect(nn, wd, xi, yi, dp, ui, vi)
        return ui * dt_ds, vi * dt_ds

    def integrate(x0, y0):
        stotal, xy_traj = 0., []

        s, xyt = _integrate_rk12(x0, y0, dmap, forward_time)
        stotal += s
        xy_traj += xyt[1:]

        return np.broadcast_arrays(xy_traj, np.empty((1, 2)))[0]

    return integrate




class OutOfBounds(IndexError):
    pass


def _integrate_rk12(x0, y0, dmap, f):
    maxerror = 0.003
    maxds = 0.001

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    xyf_traj = []

    while True:
        try:
            if dmap.grid.within_grid(xi, yi):
                xyf_traj.append((xi, yi))
            else:
                raise OutOfBounds

            # Compute the two intermediate gradients.
            # f should raise OutOfBounds if the locations given are
            # outside the grid.
            k1x, k1y = f(xi, yi)
            k2x, k2y = f(xi + ds * k1x, yi + ds * k1y)

        except OutOfBounds:
            # Out of the domain during this step.
            # Take an Euler step to the boundary to improve neatness
            # unless the trajectory is currently empty.
            if xyf_traj:
                ds, xyf_traj = _euler_step(xyf_traj, dmap, f)
                stotal += ds
            break
        except TerminateTrajectory:
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)

        ny, nx = dmap.grid.shape
        # Error is normalized to the axes coordinates
        error = np.hypot((dx2 - dx1) / (nx - 1), (dy2 - dy1) / (ny - 1))

        # Only save step if within error tolerance
        if error < maxerror:
            xi += dx2
            yi += dy2
            stotal += ds

        # recalculate stepsize based on step error
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)

    return stotal, xyf_traj


def _euler_step(xyf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    ny, nx = dmap.grid.shape
    xi, yi = xyf_traj[-1]
    cx, cy = f(xi, yi)
    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx
    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy
    ds = min(dsx, dsy)
    xyf_traj.append((xi + cx * ds, yi + cy * ds))
    return ds, xyf_traj


# Utility functions
# ========================

def interpgrid(a, xi, yi):
    """Fast 2D, linear interpolation on an integer grid"""

    Ny, Nx = np.shape(a)
    if isinstance(xi, np.ndarray):
        x = xi.astype(int)
        y = yi.astype(int)
        # Check that xn, yn don't exceed max index
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
    else:
        x = int(xi)
        y = int(yi)
        # conditional is faster than clipping for integers
        if x == (Nx - 1):
            xn = x
        else:
            xn = x + 1
        if y == (Ny - 1):
            yn = y
        else:
            yn = y + 1

    a00 = a[y, x]
    a01 = a[y, xn]
    a10 = a[yn, x]
    a11 = a[yn, xn]
    xt = xi - x
    yt = yi - y
    a0 = a00 * (1 - xt) + a01 * xt
    a1 = a10 * (1 - xt) + a11 * xt
    ai = a0 * (1 - yt) + a1 * yt

    if not isinstance(xi, np.ndarray):
        if np.ma.is_masked(ai):
            raise TerminateTrajectory

    return ai
