# Third-party notices

Mixdex is licensed under BSD-3-Clause; see [LICENSE](LICENSE).
Scientific method references are listed in [docs/references.md](https://github.com/AlSky361/Mixdex/blob/main/docs/references.md).
Citing a paper does not replace the license notices for incorporated source code.

## pybind11

The Python extension is compiled using pybind11 headers. Preserve the full
BSD-3-Clause notice in [LICENSES/pybind11.txt](LICENSES/pybind11.txt) when
redistributing the extension. The notice is reproduced from pybind11 3.1.0:
[upstream license](https://github.com/pybind/pybind11/blob/v3.1.0/LICENSE).
Its copyright notice identifies Wenzel Jakob (2016). C++-only builds do not
require pybind11. Builders selecting another version must check that version's
notices, including any additional notices for incorporated components.

## Separately installed dependencies

NumPy is a Python runtime dependency; scikit-learn is used as a test reference.
They retain their own licenses and notices. They are not vendored in the Mixdex
source tree. CMake, scikit-build-core and the compiler are build tools.

OpenMP is optional and disabled in the default Python package build. A builder
redistributing an OpenMP runtime, C++ runtime or other bundled binary dependency
must also supply the notices applicable to that particular binary distribution.
The default wheel does not bundle a separate OpenMP runtime.

## Related scientific software

The hdbscan package by Leland McInnes, John Healy and Steve Astels, and the
mlpack project are related implementations. Their papers and repository links
appear in docs/references.md. Those links acknowledge software and methods;
they do not identify their authors as copyright holders of all Mixdex files.
Any copied or translated source fragment requires its original notices in
addition to these references. A bibliography alone is not provenance evidence.
