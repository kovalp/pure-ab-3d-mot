# Pure AB3DMOT

The gist of the AB3DMOT (a base-line of 3D multiple-object tracking).

This version produces the same results as original AB3DMOT tracker 
without center-of-motion (COM) correction. Although the results are 
the same, the implementation features several refactorings which 
improve speed, readability and maintainability of the code. 

[Original repo](https://github.com/xinshuoweng/AB3DMOT)

The original package is refactored with to ultimately demonstrate the binary
classification of the associations via instrumentation of the tracker (ClavIA).

See the [changelog](https://github.com/kovalp/pure-ab-3d-mot/blob/main/CHANGELOG.md) for a detailed list of refactoring edits.

## Usage

The package contains a bare minimum for the tracker to function. The interface for
using the tracker in evaluation will be done in a separate repository.

## Install

```shell

pip install pure-ab-3d-mot
```
