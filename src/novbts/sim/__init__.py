"""Isaac/VBTS grasp simulation utilities.

This package intentionally does not import Isaac Sim or Isaac Lab at package
level. Host-side modules (FNO export, contact maps, tactile rendering) stay
unit-testable without the Isaac container; container-only drivers import Isaac
inside their ``main`` paths.
"""
