Contributing Guidelines
=======================

* Assume half our audience is non-DSP ML folks, when writing
documentation.

* `sample_rate` and `control_rate` should be parameters, not globals.

* Enforce preconditions on parameter values.

* `SynthModule` should have a `npyforward` method. Parameters intrinsic
to the module should be in `__init__`. Outputs from other modules
should go in `npyforward`.

* Let's standardize on naming conventions for things like `signal`
or `audio` or whatever.

## Code Style

* Classes should be written in functional ways, without side effects.

* Please use [Google doc-string
style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

* Please add type annotations when possible, so people know if something
is `float` or `nd.array` or whatever.

* Prefer module names that are singular (`util`) not plural (`utils`)

* Avoid short cryptic names.
