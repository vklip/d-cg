from . import _analyze_trj
from . import _tools
from . import _fit_tools
import os

try:
    import tomlkit

    dirname = os.path.dirname(__file__)
    defaults = tomlkit.load(open(dirname + "/default.toml", "r"))


except ModuleNotFoundError:
    import warnings

    warnings.warn(
        "Module 'tomlkit' is not installed. "
        + "Won't be able to read default and specific configurations for "
        + "top-level scripts."
    )

except FileNotFoundError:
    import warnings

    warnings.warn("Defaults config file 'default.toml' not found.")
# One could provide a way to change location of default file / reload new file as default
