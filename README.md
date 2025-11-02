# Pure AB3DMOT

The gist of the AB3DMOT (a base-line of 3D multiple-object tracking).

[Original repo](https://github.com/xinshuoweng/AB3DMOT)

This version produces the same results as original AB3DMOT tracker 
without center-of-motion (COM) correction. Although the results are 
the same, the implementation features several refactorings which 
improve speed, readability and maintainability of the code.

This version of the package demonstrates a binary classification of the association outcomes via
instrumentation of the tracker (ClavIA). The instrumentation consists of adding
an *annotation ID* `ann_id` to the `Box3D` and `Target` classes. Apart from the annotation ID,
an *update ID* `upd_id` is added to the `Target` class. 

The tracker class `Ab3dMot` is modified to process the  instrumentation fields.
The instrumentation serves the observing (inquiry) purposes. It does not change the original
objects beyond that purpose. The function of the instrumentation is checked in this repository and
used in a separate evaluation code (package `eval-ab-3d-mot`).

## Usage

The package contains a bare minimum for the tracker to function. The interface for
using the tracker in evaluation is done in separate repos (see `eval-ab-3d-mot`).

## Install

```shell

pip install pure-ab-3d-mot
```
