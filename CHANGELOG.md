# 2.1.0

  - Ensure integer ClavIA ids, not numpy.int64 for example.
  - Use `association-quality-clavia==0.2.0` which is more debug friendly.

# 2.0.0

  - Implement ClavIA using the `association-quality-clavia` package.

# 1.0.0

  - First version including the classification via instrumented association (ClavIA).

# 0.5.0

  - Various improvements in the `Ab3dMot.update` method.
  - Typehints in the `Box3D.bbox2array`.
  - Rename the `Box3D.bbox2array_raw` to `Box3D.bbox2array_kitti`.
  - Refactoring the `process_dets` function.
  - Add submodule `str_const`.
  - Rename the argument `dets` of `Ab3dMot.birth` to `det_boxes`.
  - Formatting the docstring in `Ab3dMot.track` according to Google standard.
  - Added unit tests for internal methods `Ab3dMot.birth` and `Ab3dMot.update`.

# 0.4.0

  - Typehints in the arguments of the `Ab3dMot.birth` method.
  - Edit the docstring of the `data_association` function. 

# 0.3.0

  - Smaller value of the spatial tolerance `1e-14` previously `1e-4` to better serve in evaluation.


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


