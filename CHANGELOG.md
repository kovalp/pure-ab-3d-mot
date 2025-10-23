# 0.3.0

  - Smaller value of the spatial tolerance (1e-14) to better serve in evaluation.


# 0.2.0

  - Example 1-757 of nuScenes annotations in unit tests.
  - Publish in PyPI registry.

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
  - Cease storing the corners of the bounding box in the objects `Box3D`.
  - Test the rest of the association metrics.
  - Test the rest of the `data_association` function.
  - Leave only `roty` from `kitti_oxts`
  - Achieve 100% coverage in unit tests.
  - Run reformat and isort  (`ruff format` and `ruff check --fix`).


