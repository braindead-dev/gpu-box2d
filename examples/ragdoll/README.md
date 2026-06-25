# Ragdoll example

A jointed chain (a simple ragdoll limb or rope) driven through the gpu-box2d Python binding. It builds a chain of box segments connected by revolute joints, pins it at the top to a static anchor, and lets it swing under gravity. The segments stay connected at their joints while the chain settles.

This is a worked instance of the joints and polygons in use together, outside any game: it uses only box bodies, revolute joints, the step, and the numpy read-back.

## Run

Build the Python binding first (see [../../bindings/README.md](../../bindings/README.md)), then:

```
python examples/ragdoll/ragdoll.py
```

It prints the chain tip height, the link lengths before and after settling, and a PASS line when the chain swings down, the joints hold the links, and all worlds agree.
