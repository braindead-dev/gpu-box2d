# Box stack example

A tower of boxes settling, driven through the gpu-box2d Python binding. It builds a stack of box bodies in every world of a batch, steps the batch, and checks that the tower settles into an ordered, evenly spaced column, with every world bit-identical.

This is a worked instance of driving the engine for a general scene with no game logic: it uses only the static boundary, box bodies, the step, and the numpy state read-back.

## Run

Build the Python binding first (see [../../bindings/README.md](../../bindings/README.md)), then:

```
python examples/box_stack/box_stack.py
```

It prints the settled box heights and a PASS line when the tower is ordered and even and all worlds agree.
