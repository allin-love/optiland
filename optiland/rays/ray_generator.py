"""Ray Generator

This module defines the RayGenerator class, which is used to generate rays
for tracing through an optical system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.rays.polarized_rays import PolarizedRays
from optiland.rays.real_rays import RealRays


class RayGenerator:
    """Generator class for creating rays through an optical system.

    Args:
        optic (Optic): A configured optical system object.
    """

    def __init__(self, optic):
        self.optic = optic

    def generate_rays(self, Hx, Hy, Px, Py, wavelength):
        """
        Generate rays for tracing based on the provided field and pupil parameters.

        The method computes the ray origins and directions based on whether the
        system is telecentric in object space. It also applies necessary error
        checks when the optic configuration does not support certain parameters.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            wavelength (float): Wavelength of the rays.

        Returns:
            RealRays or PolarizedRays: An object containing the generated rays.

        Raises:
            ValueError: If the optic configuration is inconsistent (e.g.,
                using an "angle" field type in a telecentric object space).
        """
        vx, vy = 1 - be.array(self.optic.fields.get_vig_factor(Hx, Hy))
        x0, y0, z0 = self._get_ray_origins(Hx, Hy, Px, Py, vx, vy)

        # Determine terminal ray coordinates based on object space configuration.
        if self.optic.obj_space_telecentric:
            self._validate_telecentric_conditions()
            sin_value = self.optic.aperture.value
            # Compute effective z-coordinate.
            z_terminal = be.sqrt(1 - sin_value**2) / sin_value + z0
            z1 = be.full_like(Px, z_terminal)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()
            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            z1 = be.full_like(Px, EPL)

        # Calculate direction cosines.
        L, M, N = self._compute_direction_cosines(x0, y0, z0, x1, y1, z1)

        # Ensure origins are in array form.
        x0 = be.full_like(x1, x0)
        y0 = be.full_like(x1, y0)
        z0 = be.full_like(x1, z0)

        intensity = be.ones_like(x1)
        wavelength_arr = be.full_like(x1, wavelength)

        if self.optic.polarization == "ignore":
            if self.optic.surface_group.uses_polarization:
                raise ValueError(
                    "Polarization must be set when surfaces have "
                    "polarization-dependent coatings."
                )
            return RealRays(x0, y0, z0, L, M, N, intensity, wavelength_arr)
        return PolarizedRays(x0, y0, z0, L, M, N, intensity, wavelength_arr)

    def _validate_telecentric_conditions(self):
        """
        Validate conditions that must hold for a telecentric object space configuration.

        Raises:
            ValueError: If the field type or aperture type is inconsistent with
                a telecentric object space.
        """
        if self.optic.field_type == "angle":
            raise ValueError(
                'Field type cannot be "angle" for telecentric object space.'
            )
        if self.optic.aperture.ap_type in {"EPD", "imageFNO"}:
            raise ValueError(
                f'Aperture type "{self.optic.aperture.ap_type}" cannot '
                f"be used for telecentric object space."
            )

    def _compute_direction_cosines(self, x0, y0, z0, x1, y1, z1):
        """
        Compute the unit direction cosines for rays given origin and terminal points.

        Args:
            x0, y0, z0: Coordinates of the ray origins.
            x1, y1, z1: Coordinates of the terminal points.

        Returns:
            tuple: (L, M, N) representing the direction cosines.
        """
        delta_x = x1 - x0
        delta_y = y1 - y0
        delta_z = z1 - z0
        mag = be.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        L = delta_x / mag
        M = delta_y / mag
        N = delta_z / mag
        return L, M, N

    def _get_ray_origins(self, Hx, Hy, Px, Py, vx, vy):
        """
        Calculate the initial positions for rays originating at the object.

        The origin is computed differently based on whether the object surface is
        at infinity.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: Arrays (x0, y0, z0) representing the ray origin coordinates.

        Raises:
            ValueError: If the field type is "object_height" for an object at infinity,
                or if a telecentric configuration is used with an object at infinity.
        """
        obj = self.optic.object_surface
        max_field = self.optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy

        if obj.is_infinite:
            if self.optic.field_type == "object_height":
                raise ValueError(
                    'Field type cannot be "object_height" for an object at infinity.'
                )
            if self.optic.obj_space_telecentric:
                raise ValueError(
                    "Object space cannot be telecentric for an object at infinity."
                )

            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()
            offset = self._get_starting_z_offset()

            # x, y, z positions of ray starting points
            x = -be.tan(be.radians(field_x)) * (offset + EPL)
            y = -be.tan(be.radians(field_y)) * (offset + EPL)
            z = self.optic.surface_group.positions[1] - offset

            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            z0 = be.full_like(Px, z)
        else:
            if self.optic.field_type == "object_height":
                x_base = field_x
                y_base = field_y
                # The sag function accounts for the surface curvature.
                z_val = obj.geometry.sag(x_base, y_base) + obj.geometry.cs.z
            elif self.optic.field_type == "angle":
                EPL = self.optic.paraxial.EPL()
                z_val = self.optic.surface_group.positions[0]
                x_base = -be.tan(be.radians(field_x)) * (EPL - z_val)
                y_base = -be.tan(be.radians(field_y)) * (EPL - z_val)
            else:
                raise ValueError(f"Unsupported field type: {self.optic.field_type}")

            x0 = be.full_like(Px, x_base)
            y0 = be.full_like(Px, y_base)
            z0 = be.full_like(Px, z_val)

        return x0, y0, z0

    def _get_starting_z_offset(self):
        """
        Calculate the starting ray z-coordinate offset for systems with an
        object at infinity. The offset is defined relative to the first surface
        of the optic and is chosen to be equivalent to the entrance pupil diameter.

        Returns:
            float: The z-coordinate offset relative to the first surface.
        """
        z = self.optic.surface_group.positions[1:-1]
        offset = self.optic.paraxial.EPD()
        return offset - be.min(z)
