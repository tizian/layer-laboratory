<img src="https://github.com/tizian/layer-laboratory/raw/master/docs/images/layer-teaser.jpg" alt="Layer Laboratory teaser">

# The Layer Laboratory: A Calculus for Additive and Subtractive Composition of Anisotropic Surface Reflectance

Source code of the paper ["The Layer Laboratory: A Calculus for Additive and Subtractive Composition of Anisotropic Surface Reflectance"](http://rgl.epfl.ch/publications/Zeltner2018Layer) by [Tizian Zeltner](https://tizianzeltner.com/) and [Wenzel Jakob](http://rgl.epfl.ch/people/wjakob) from SIGGRAPH 2018.

The implementation is based on the Mitsuba 2 Renderer, see the lower part of the README.

## Compilation

The normal compilation instructions for Mitsuba 2 apply. See the ["Getting started"](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/intro.html) sections in the documentation. For this project, only the *scalar_rgb* variant is tested. In addition, there are a few extra dependencies:

* [UMFPACK](http://faculty.cse.tamu.edu/davis/suitesparse.html) is **required** for layer addition/subtraction and can be installed via a packet manager, e.g. on macOS with:
```
brew install suite-sparse
```
* The [sparseqr](https://pypi.org/project/sparseqr/) Python package is needed for the layer subtraction experiments. It can be installed with
```
pip install sparseqr
```
* The [libvips](https://github.com/libvips/libvips) image processing library is needed for the dE00 error images in Figures 11 and 12. It can be installed from a package manager, e.g. on macOS with:
```
brew install vips
```

## Usage

After compiling everything, and [configuring the environment variables](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/compiling.html#running-mitsuba), Mitsuba's Python API can be used to create layered materials. Here is a small example that adds a rough dielectric coating onto an anisotropic rough conductor:
```Python
import mitsuba; mitsuba.set_variant('scalar_rgb')
import mitsuba.layer

# Bottom layer settings
eta_bot = 0+1j   # Metal, without Fresnel effect
alpha_u_bot = 0.1
alpha_v_bot = 0.2

# Top layer settings
eta_top = 1.5       # Dielectric
alpha_top = 0.08

# Determine necessary Fourier discretization
n1, ms1, md1 = mitsuba.layer.microfacet_parameter_heuristic(alpha_top, alpha_top, eta_top)
n2, ms2, md2 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_bot, alpha_v_bot, eta_bot)
n = max(n1, n2)
ms = max(ms1, ms2)
md = md2
mu, w = mitsuba.core.quad.gauss_lobatto(n)

# Create bottom layer
layer_bot = mitsuba.layer.Layer(mu, w, ms, md)
layer_bot.set_microfacet(eta_bot, alpha_u_bot, alpha_v_bot)
layer_bot.clear_backside()

# Create top layer
layer_top = mitsuba.layer.Layer(mu, w, ms, md)
layer_top.set_microfacet(eta_top, alpha_top, alpha_top)

# Layer addition.
# 'epsilon' is used as a threshold for dropping near-zero Fourier coefficients.
# Larger values use less memory during the addition process but reduce accuracy.
layer_bot.add_to_top(layer_top, epsilon=1e-9)

# Save as .bsdf file
storage = mitsuba.layer.BSDFStorage.from_layer("layered.bsdf", layer_bot, 1e-4)
```

The created `.bsdf` file can then be loaded in a normal Mitsuba 2 scene with the following XML tag:
```xml
<bsdf type="fourier" id="my_layered_material">
    <string name="filename" value="layered.bsdf"/>
</bsdf>
```

## Results

The directory `results` contains a set of folders for the different figures in the paper, e.g. `results/Figure_<N>_<Name>`. They contain Python scripts (to generate plots or compute BSDFs) as well as Mitsuba 2 scenes for rendered results.

* All of these scripts need to be run *from the respective subfolder* to ensure that files are written to existing directories.
* Most scripts assume that Mitsuba was added to the path either manually or by running `source setpath.sh`. See the ["Running Mitsuba"](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/compiling.html#running-mitsuba) section in the documentation.

Here is a list of available results:

### `results/Figure_2_AnisotropicFrequencies/`
* Run Python script `generate.py` which will generate pdfs for the individual subplots.

### `results/Figure_3_IncreasingMSParameter/`
* Run Python script `precompute.py` to generate `.bsdf` files.
* Run `mitsuba <scene.xml>` to render the individual scenes shown in the figure.

###  `results/Figure_4_InverseTransmission/`
* Run Python script `generate.py` to generate the plot.

### `results/Figure_5_FresnelRegularization/`
* Run Python script `generate.py` to generate the 4 plots.

### `results/Figure_6_AnisotropicSpheres/`
* Run Python script `precompute.py` to generate `.bsdf` files. As we are generating BSDFs with strong anisotropy in both layers, the precomputation here is especially expensive. It runs for a very long time and requires up to 64 GB of memory.
* Run `mitsuba <scene.xml>` to render the individual scenes shown in the figure.

See also the [interactive viewer](http://rgl.s3.eu-central-1.amazonaws.com/media/papers/Zeltner2018Layer_1.zip) to further explore the design space of two-layer BSDF combinations. The data for it was generated with the same scripts but with a larger set of combinations.

### `results/Figure_7_LayeredExamples/`

* Run Python script `precompute.py` to generate `.bsdf` files. The same considerations as above (Figure 6) apply.
* Run `mitsuba <scene.xml>` to render the individual scenes shown in the figure.

### `results/Figure_9_11_12_Validation/`

#### 9 (a) & (b): Analytic experiments

* Run Python script `results/Figure_9_11_12_Validation/analytic_addition.py` that first computes MC references for the added layers and then produces the corresponding plots from Figure 9a.

* Run Python script `results/Figure_9_11_12_Validation/analytic_subtraction.py`. This first performs the layer subtraction (implemented in Python) and then produces the corresponding plots from Figure 9b.

#### Measurement data

* Before running the next few experiments, please download the [measured data](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Zeltner2018Layer_4.zip) separately and place the `measurements` directory under `results/Figure_9_11_12_Validation/`.
* The data already includes the raw [gonio-photometric](http://rgl.epfl.ch/pages/lab/pgII) measurements for different incident directions and the converted Fourier representation as numpy arrays. This conversion process can be reproduced by running the `measurements_to_scattering_matrices.py` Python script.
* There is also an interactive Jupyter notebook (`Visualize measurements.ipynb`) that loads and visualizes the measurement files for varying incident directions, including the conversion into the Fourier representation.

#### 9 (c) & (e): Measured addition experiments

* Run Python script `results/Figure_9_11_12_Validation/measured_addition.py` that performs the additions and then generates the plots from Figure 9c,e.

#### 9 (d) & (f): Measured subtraction experiments

* Run Python script `results/Figure_9_11_12_Validation/measured_subtraction.py` that performs the subtractions and then generates the plots from Figure 9d,f.

#### Renderings

* Run Python script `results/Figure_9_11_12_Validation/render.py` that will render the Dragon scene for all steps during the layer addition and subtraction experiments.

#### 11 & 12: Experiment errors

* Run Python script `results/Figure_9_11_12_Validation/error_plots.py` that will generate the error plot in these two figures.
* Run Python script `results/Figure_9_11_12_Validation/error_images.py` that will generate the dE error images. This step requires that the corresponding scenes were previously rendered (see previous subsection) with a sufficiently high sample count (~2000 spp).

#### Supplemental material

See also the [supplemental material](http://rgl.s3.eu-central-1.amazonaws.com/media/papers/Zeltner2018Layer.zip) for this paper which includes a larger set of BSDF plots for these experiments, computed for different incident elevation angles.


### Unit tests
The code also includes a few unit tests for individual components of the system to ensure for example that our Fourier BSDF representation can correctly encode existing analytic BSDFs.
They can be found in `src/liblayer/tests/...` and can be run with [pytest](https://docs.pytest.org/en/latest/).

---

<img src="https://github.com/mitsuba-renderer/mitsuba2/raw/master/docs/images/logo_plain.png" width="120" height="120" alt="Mitsuba logo">

# Mitsuba Renderer 2
<!--
| Documentation   | Linux             | Windows             |
|      :---:      |       :---:       |        :---:        |
| [![docs][1]][2] | [![rgl-ci][3]][4] | [![appveyor][5]][6] |


[1]: https://readthedocs.org/projects/mitsuba2/badge/?version=master
[2]: https://mitsuba2.readthedocs.io/en/latest/src/getting_started/intro.html
[3]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:Mitsuba2_Build)/statusIcon.svg
[4]: https://rgl-ci.epfl.ch/viewType.html?buildTypeId=Mitsuba2_Build&guest=1
[5]: https://ci.appveyor.com/api/projects/status/eb84mmtvnt8ko8bh/branch/master?svg=true
[6]: https://ci.appveyor.com/project/wjakob/mitsuba2/branch/master
-->
| Documentation   |
|      :---:      |
| [![docs][1]][2] |


[1]: https://readthedocs.org/projects/mitsuba2/badge/?version=latest
[2]: https://mitsuba2.readthedocs.io/en/latest/src/getting_started/intro.html

Mitsuba 2 is a research-oriented rendering system written in portable C++17. It
consists of a small set of core libraries and a wide variety of plugins that
implement functionality ranging from materials and light sources to complete
rendering algorithms. Mitsuba 2 strives to retain scene compatibility with its
predecessor [Mitsuba 0.6](https://github.com/mitsuba-renderer/mitsuba).
However, in most other respects, it is a completely new system following a
different set of goals.

The most significant change of Mitsuba 2 is that it is a *retargetable*
renderer: this means that the underlying implementations and data structures
are specified in a generic fashion that can be transformed to accomplish a
number of different tasks. For example:

1. In the simplest case, Mitsuba 2 is an ordinary CPU-based RGB renderer that
   processes one ray at a time similar to its predecessor [Mitsuba
   0.6](https://github.com/mitsuba-renderer/mitsuba).

2. Alternatively, Mitsuba 2 can be transformed into a differentiable renderer
   that runs on NVIDIA RTX GPUs. A differentiable rendering algorithm is able
   to compute derivatives of the entire simulation with respect to input
   parameters such as camera pose, geometry, BSDFs, textures, and volumes. In
   conjunction with gradient-based optimization, this opens door to challenging
   inverse problems including computational material design and scene reconstruction.

3. Another type of transformation turns Mitsuba 2 into a vectorized CPU
   renderer that leverages Single Instruction/Multiple Data (SIMD) instruction
   sets such as AVX512 on modern CPUs to efficiently sample many light paths in
   parallel.

4. Yet another type of transformation rewrites physical aspects of the
   simulation: Mitsuba can be used as a monochromatic renderer, RGB-based
   renderer, or spectral renderer. Each variant can optionally account for the
   effects of polarization if desired.

In addition to the above transformations, there are
several other noteworthy changes:

1. Mitsuba 2 provides very fine-grained Python bindings to essentially every
   function using [pybind11](https://github.com/pybind/pybind11). This makes it
   possible to import the renderer into a Jupyter notebook and develop new
   algorithms interactively while visualizing their behavior using plots.

2. The renderer includes a large automated test suite written in Python, and
   its development relies on several continuous integration servers that
   compile and test new commits on different operating systems using various
   compilation settings (e.g. debug/release builds, single/double precision,
   etc). Manually checking that external contributions don't break existing
   functionality had become a severe bottleneck in the previous Mitsuba 0.6
   codebase, hence the goal of this infrastructure is to avoid such manual
   checks and streamline interactions with the community (Pull Requests, etc.)
   in the future.

3. An all-new cross-platform user interface is currently being developed using
   the [NanoGUI](https://github.com/mitsuba-renderer/nanogui) library. *Note
   that this is not yet complete.*

## Compiling and using Mitsuba 2

Please see the [documentation](http://mitsuba2.readthedocs.org/en/latest) for
details on how to compile, use, and extend Mitsuba 2.

## About

This project was created by [Wenzel Jakob](http://rgl.epfl.ch/people/wjakob).
Significant features and/or improvements to the code were contributed by
[Merlin Nimier-David](https://merlin.nimierdavid.fr/),
[Guillaume Loubet](https://maverick.inria.fr/Membres/Guillaume.Loubet/),
[Sébastien Speierer](https://github.com/Speierers),
[Delio Vicini](https://dvicini.github.io/),
and [Tizian Zeltner](https://tizianzeltner.com/).
