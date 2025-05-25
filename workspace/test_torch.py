import optiland.backend as be
from optiland.optic import Optic
from optiland.materials import Material
from optiland.analysis import SpotDiagram


# set backend
be.set_backend("torch")
be.set_device(
    "cpu"
)  # we will stick to cpu for now. "cuda" is also possible for GPU calculations.
print(be.get_precision())  # check the precision

# enable the gradient mode: to tell autograd to begin recording operations on a Tensor tensor
be.grad_mode.enable()


class SingletConfigurable(Optic):
    """A configurable singlet lens."""

    def __init__(self, r1, r2, t2, material_name):
        super().__init__()
        ideal_material = Material(material_name)

        self.add_surface(index=0, radius=be.inf, thickness=be.inf)
        self.add_surface(
            index=1, thickness=7.0, radius=r1, is_stop=True, material=ideal_material
        )
        self.add_surface(index=2, radius=r2, thickness=t2)
        self.add_surface(index=3)

        self.set_aperture(aperture_type="EPD", value=25)

        self.set_field_type(field_type="angle")
        self.add_field(y=0.0)

        self.add_wavelength(value=0.55, is_primary=True)

lens = SingletConfigurable(r1=70.0, r2=-70.0, t2=70.0, material_name="BK7")
lens.draw(num_rays=10)
lens.info()