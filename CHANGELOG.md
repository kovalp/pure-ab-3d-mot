# 0.1.0

  - Added unit tests
  - Rename the main class from `AB3DMOT` to `Ab3DMot`.
  - Added `pyproject.toml` manageable by `uv` package manager.
  - Simplify `Ab3DMot.prediction()`.
  - Rename the module `kalman_filter` to `target`.
  - Rename the class `KalmanFilter` to `Target`.
  - Move the attributes of the class `Filter` to `Target`.
  - Added typehints in several methods and functions.
  - Convert some static methods to pure functions.
  - Added `scipy-stubs` dependency for `python > 3.9`.
  - Introduced `MetricKind` enumerable.
  - Added a tolerance `1e-4` to the `inside` internal function.
  - Moved the functions related to IOU to a separate module.
  - Tested the `Target` class.
  - Tested the Mahalanobis association metric.
