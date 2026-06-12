"""novbts — Neural-operator surrogate for a vision-based tactile sensor (VBTS).

An FNO learns marker displacement fields as a fast surrogate replacing an
expensive contact solver, for downstream RL.  Ground truth comes from PhysX
Deformable-Body FEM (Isaac Sim); Hertz-Mindlin provides an analytic validator.
"""
__version__ = "0.1.0"
