# Pure AB3DMOT

The gist of the AB3DMOT (a base-line of 3D multiple-object tracking).

This version produces the same results as original AB3DMOT tracker 
without center-of-motion (COM) correction. Although the results are 
the same, the implementation features several refactorings which 
improve speed, readability and maintainability of the code.

[Original repo](https://github.com/xinshuoweng/AB3DMOT)

The original package is refactored to ultimately demonstrate the binary
classification of the associations via instrumentation of the tracker (ClavIA).

See the [changelog](https://github.com/kovalp/pure-ab-3d-mot/blob/main/CHANGELOG.md) for a detailed list of refactoring edits.

This version contains the instrumentation of the tracker association procedure.
The instrumentation consists in adding *annotation IDs* `ann_id` in `Box3D`
and `Target` classes as well as adding an *update ID* `upd_id` in the 
`Target` class. The tracker class `Ab3dMot` is modified to process the 
instrumentation fields. The instrumentation is constructed for observing (inquiry)
purposes. It does not change the original objects beyond that purpose.

## Usage

The package contains a bare minimum for the tracker to function. The interface for
using the tracker in evaluation will be done in separate repos (see `eval-ab-3d-mot`).

## Install

```shell

pip install pure-ab-3d-mot
```
