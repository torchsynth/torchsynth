Contributing Guidelines
=======================

* Assume half our audience is non-DSP ML folks, when writing
documentation.

* `sample_rate` and `control_rate` should be parameters, not globals.

* `SynthModule` should have a `__call__` method. Parameters intrinsic
to the module should be in `__init__`. Outputs from other modules
should go in `__call__`.

* Let's standardize on naming conventions for things like `signal`
or `audio` or whatever.

## Code Style

* Classes should be written in functional ways, without side effects.

* Prefer module names that are singular (`util`) not plural (`utils`)

* Avoid short cryptic names.

* @turian will occasionally run `black` and `isort` and `flake8`
to enforce code style conventions. You are welcome to do this
yourself.  We may try to add a github action that PRs will get
annotated complaints that we can ignore if we like.
