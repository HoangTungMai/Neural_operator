"""Phase 5 -- marker-dot VBTS sensor model.

A differentiable camera + dot renderer that turns the gel-surface marker
displacement field (FEM GT or FNO prediction) into the actual sensor output a
dotted-gel VBTS produces: a camera image of marker dots, and the 2D marker flow.
Because the renderer is built in torch it is differentiable, so render . FNO is
end-to-end differentiable (inverse/control from the real sensor image, not the
raw displacement field).
"""
