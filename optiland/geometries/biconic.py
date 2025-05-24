"""Biconic Geometry

The Biconic geometry represents a surface defined by different conic profiles
along the x and y axes. The surface equation is:

z = (cx * x^2 + cy * y^2) / (1 + sqrt(1 - (1 + kx) * cx^2 * x^2 -
                                      (1 + ky) * cy^2 * y^2))

where
- cx = 1 / Rx (curvature in x)
- cy = 1 / Ry (curvature in y)
- Rx is the radius of curvature along the x-axis
- Ry is the radius of curvature along the y-axis
- kx is the conic constant along the x-axis
- ky is the conic constant along the y-axis

Biconic surfaces are used when different optical powers are needed in the
x and y directions, common in systems for correcting astigmatism or for
anamorphic optics.

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class Biconic(NewtonRaphsonGeometry):
    """Represents a biconic geometry defined as:

    z = (cx * x^2 + cy * y^2) / (1 + sqrt(1 - (1 + kx) * cx^2 * x^2 -
                                          (1 + ky) * cy^2 * y^2))

    where:
    - cx = 1 / Rx
    - cy = 1 / Ry
    - Rx is the radius of curvature along the x-axis.
    - Ry is the radius of curvature along the y-axis.
    - kx is the conic constant relevant to the x-direction.
    - ky is the conic constant relevant to the y-direction.

    The x-radius (Rx) and x-conic constant (kx) are passed to the base
    NewtonRaphsonGeometry class for iterative ray intersection calculations.

    Args:
        coordinate_system (CoordinateSystem): The coordinate system for the geometry.
        radius_x (float): The radius of curvature along the x-axis (Rx).
            Must be non-zero.
        radius_y (float): The radius of curvature along the y-axis (Ry).
            Must be non-zero.
        conic_x (float, optional): The conic constant for the x-axis (kx).
            Defaults to 0.0.
        conic_y (float, optional): The conic constant for the y-axis (ky).
            Defaults to 0.0.
        tol (float, optional): Tolerance for Newton-Raphson ray intersection.
            Defaults to 1e-10.
        max_iter (int, optional): Maximum iterations for Newton-Raphson.
            Defaults to 100.

    Raises:
        ValueError: If radius_x or radius_y is zero.
    """

    def __init__(
        self,
        coordinate_system,
        radius_x,
        radius_y,
        conic_x=0.0,
        conic_y=0.0,
        coefficients=None,
        tol=1e-10,
        max_iter=100,
    ):
        if radius_x == 0:
            raise ValueError("Radius of curvature radius_x must be non-zero.")
        if radius_y == 0:
            raise ValueError("Radius of curvature radius_y must be non-zero.")

        super().__init__(coordinate_system, radius_x, conic_x, tol, max_iter)

        self.radius_x = radius_x
        self.radius_y = radius_y
        self.conic_x = conic_x
        self.conic_y = conic_y

        self.cx = 1.0 / self.radius_x
        self.cy = 1.0 / self.radius_y

        if coefficients is None:
            self.c = []
        else:
            self.c = coefficients

        self.is_symmetric = False  # Biconic surfaces are not rotationally symmetric
        self.order = 2  # Base surface involves second-order terms in x and y

    def __str__(self):
        return "Biconic"

    def sag(self, x=0, y=0):
        """Calculates the sag of the biconic surface at given coordinates.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            float or be.ndarray: The sag value(s) at the given coordinates.
        """
        x2 = x**2
        y2 = y**2

        # Biconic part
        term_x_num = self.cx * x2
        term_y_num = self.cy * y2
        numerator = term_x_num + term_y_num

        sqrt_arg = (
            1.0
            - (1.0 + self.conic_x) * self.cx**2 * x2
            - (1.0 + self.conic_y) * self.cy**2 * y2
        )
        denominator = 1.0 + be.sqrt(sqrt_arg)

        return numerator / denominator

    def _surface_normal(self, x, y):
        """Calculates the surface normal of the biconic at (x, y).

        The normal is calculated based on the gradient of sag(x,y) - z = 0.
        Normal vector components are (dfdx, dfdy, -1) normalized, where
        dfdx = dz/dx and dfdy = dz/dy.

        Args:
            x (be.ndarray): The x-coordinate(s).
            y (be.ndarray): The y-coordinate(s).

        Returns:
            tuple: Normalized surface normal components (nx, ny, nz).
        """
        x2 = x**2
        y2 = y**2

        num_N = self.cx * x2 + self.cy * y2  # Numerator N

        sqrt_arg_P_term = (1.0 + self.conic_x) * self.cx**2 * x2 + (
            1.0 + self.conic_y
        ) * self.cy**2 * y2

        sqrt_val = be.sqrt(1.0 - sqrt_arg_P_term)

        den_D = 1.0 + sqrt_val  # Denominator D

        dDdx_num = -(1.0 + self.conic_x) * self.cx**2 * x
        dDdx = dDdx_num / sqrt_val

        dDdy_num = -(1.0 + self.conic_y) * self.cy**2 * y
        dDdy = dDdy_num / sqrt_val

        dNdx = 2.0 * self.cx * x
        dNdy = 2.0 * self.cy * y

        dfdx = (dNdx * den_D - num_N * dDdx) / (den_D**2)
        dfdy = (dNdy * den_D - num_N * dDdy) / (den_D**2)

        mag = be.sqrt(dfdx**2 + dfdy**2 + 1.0)

        nx = dfdx / mag
        ny = dfdy / mag
        nz = -1.0 / mag

        return nx, ny, nz

    def to_dict(self):
        """Converts the Biconic geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the Biconic geometry.
        """
        data = (
            super().to_dict()
        )  # Gets cs, tol, max_iter and radius_x (as 'radius'), conic_x (as 'conic')
        data["radius_x"] = self.radius_x
        data["radius_y"] = self.radius_y
        data["conic_x"] = self.conic_x
        data["conic_y"] = self.conic_y
        return data

    @classmethod
    def from_dict(cls, data):
        """Creates a Biconic geometry from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the biconic.

        Returns:
            Biconic: The Biconic geometry object.

        Raises:
            ValueError: If required keys ('cs', 'radius_x', 'radius_y')
                        are missing from the dictionary.
        """
        required_keys = {"cs", "radius_x", "radius_y"}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys for Biconic: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])
        radius_x = data["radius_x"]
        radius_y = data["radius_y"]

        conic_x = data.get("conic_x", 0.0)
        conic_y = data.get("conic_y", 0.0)

        tol = data.get("tol", 1e-10)
        max_iter = data.get("max_iter", 100)

        return cls(
            cs,
            radius_x,
            radius_y,
            conic_x,
            conic_y,
            tol,
            max_iter,
        )
