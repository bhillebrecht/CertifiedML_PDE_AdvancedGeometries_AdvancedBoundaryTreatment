[![arXiv][arxiv-shield]][arxiv-url]
<!--[![DOI][doi-shield]][doi-url]-->

[arxiv-shield]:https://img.shields.io/badge/arXiv-2508.07994-b31b1b.svg?style=for-the-badge
[arxiv-url]:http://arxiv.org/abs/2508.07994
[project-url]:https://github.com/bhillebrecht/CertifiedML_PDE_AdvancedGeometries_AdvancedBoundaryTreatment


# Prediction error certification for PINNs: Theory, computation, and application to Stokes flow

This repository contains the numerical examples for the paper [Prediction error certification for PINNs: Theory, computation, and application to Stokes flow][arxiv-url] [1].

In particular, the repository contains implementations to estimate the prediction error for a physics-informed neural network (PINN). Numerical examples include the heat equation and a 2D Stokes flow around a cylinder. To obtain information on the true prediction error, the second example is also discretized via the finite element method using the fenics library.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#citing">Citing</a></li>
    <li><a href="#setup">Setup</a></li>
    <li><a href="#heat-equation">Heat equation</a></li>
    <li><a href="#stokes-equation">Stokes equation</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Citing
If you use this project for academic work, please consider citing our
[publication][arxiv-url]:

    Birgit Hillebrecht and Benjamin Unger
    Prediction error certification for PINNs: Theory, computation, and application to Stokes flow
    ArXiv e-print 2508.07994 (https://doi.org/10.48550/arXiv.2508.07994)

## Setup

The implementations are based on python 3.13.2. The concrete packages are listed in environment files in the folder _envs. There, you can find two environment files.

- environment_certified_pinn.yml: This is the environment used to certify a PINN prediction
- environment_fenics.yml: This environment is required for the numerical discretization of the 2D Stokes flow.

To activate a specific environment via conda use
> conda env create -f _envs/environment_certified_pinn.yml

or
> conda env create -f _envs/environment_fenics.yml

## Heat equation

The heat equation example does not require the usage of FEM discretizations; hence, only the PINN environment is used:

> conda activate certified_pinn_env

The following steps have to be performed to recreate the example.
### Step 1. Create the initial condition

First, we have to create the initial value for the PDE. In this particular example, we use the $\sin$ function.
> python base_framework/heat_equation_soft_bcs/input_data/generate_input_data.py

This generates the file initial_data.csv (in the folder base_framework/heat_equation_soft_bcs/input_data/), which is then used for the PINN training in the next step.

Moreover, we have to construct testing data via the call
> python base_framework/heat_equation_soft_bcs/input_data/test_data/generate_test_data.py

which are stored in the folder base_framework/heat_equation_soft_bcs/input_data/test_data and are of the form
> test_data_t0.50000.csv

### Step 2. Train the PINN
To train a PINN for the heat equation, we call the script
> python base_framework/certified_pinn.py -t user -u base_framework/heat_equation_soft_bcs/heat_equation.py train -i base_framework/heat_equation_soft_bcs/input_data/initial_data.csv

Two remarks are in order:
- The neural network architecture is stored in base_framework/heat_equation_soft_bcs/config_NN.json
- The training configuration is stored in base_framework/heat_equation_soft_bcs/config_training.json

### Step 3. Prepare the error certification
To prepare the certification, we have to determine the growth bounds for the PINN error estimator and norms of the operators. This is done via
> python base_framework/certified_pinn.py -t user -u base_framework/heat_equation_soft_bcs/heat_equation.py extract -i base_framework/heat_equation_soft_bcs/input_data/initial_data.csv

The results are stored in the files
> output_data/bc_err_max_norm.csv

> tanh_kpis.json

### Step 4. Evaluate the error estimator
To evaluate the error estimator, we have to call
> python base_framework/certified_pinn.py -t user -u base_framework/heat_equation_soft_bcs/heat_equation.py run -i base_framework/heat_equation_soft_bcs/input_data/test_data/\<filename\> -ae domain -eps 0.1

where filename is one of the training files generated in step 1. To generate the plots in the paper, this call has to be done for each single time point, i.e., for each csv-file in the folder test_data.

### Step 5: Generate summary
A summary of the results from step 4 can be generated via
> python base_framework/heat_equation_soft_bcs/output_data/run_tanh_values/join_error_json.py

which produces the file
> output_data/run_tanh_values/run_tanh_test_data_t_error_over_time.csv

TODO: How to create the plot from the paper out of this file



The different parametrizations of the training weighting of the boundary error and its temporal derivative can be found in different folders listed in the following table.

| $\lambda_{bc}$ | Foldername |
|----------|------------|
| 0.1      | base_framework/heat_equation_soft_bcs_param4/heat_equation.py |
| 1.0      | base_framework/heat_equation_soft_bcs_param2/heat_equation.py |
| 3.14     | base_framework/heat_equation_soft_bcs_param3/heat_equation.py |
| 10.0     | base_framework/heat_equation_soft_bcs/heat_equation.py |

The examples can be run by adapting the folder paths in the previous instructions accordingly.

## Stokes equation
For the heat equation example, all growth-bound parameters can be estimated analytically. For the 2D Stokes problem, this is challenging, such that we rely on an FEM approximation. In particular, we will now rely on both environments. 

### Step 1: Create meshes and determine eigenmodes

This step is fully executed in the FEM environment, so get started with

> conda activate fenics_env

Then, there are three main scripts that need to be used

1) in [generate_data.py](fem_discretization_framework/input_data/flow_around_cylinder/generate_data.py) the mesh for the computation of the harmonic eigenmodes is generated. The domain is slightly extended as described in the preprint to enable the eigenfunctions to capture non-zero boundary conditions. The script can be converted to a python notebook using jupytext

    > jupytext --to ipynb fem_discretization_framework/input_data/flow_around_cylinder/generate_data.py

3) in [compute_reference_solution_from_dolfinx_tutorial.py](fem_discretization_framework/input_data/flow_around_cylinder/compute_reference_solution_from_dolfinx_tutorial.py), the DolfinX tutorial was adapted for our problem. The file can be converted into a python notebook using jupytext

    > jupytext --to ipynb fem_discretization_framework/input_data/flow_around_cylinder/compute_reference_solution_from_dolfinx_tutorial.py

    running the script will generate additional data which is necessary for a later visualization in accordance with the FEM visualization.

3) in [flow_around_cylinder_compute_harmonic_features.py](fem_discretization_framework/src/flow_around_cylinder/flow_around_cylinder_compute_harmonic_features.py), the generated meshes from [generate_data.py](fem_discretization_framework/input_data/flow_around_cylinder/generate_data.py) are used to determine the eigenmodes, store the eigenmodes and evaluate the eigenmodes suitably for the NN to train the equation and the boundary conditions. The implementation was adapted from [2].

4) The script [characteristic_params.py](fem_discretization_framework/src/flow_around_cylinder/characteristic_params.py) can be converted to a python notebook using jupytext

    > jupytext --to py fem_discretization_framework/src/flow_around_cylinder/characteristic_params.py

    Executing this script generates a csv table [characteristic_parameters_stokes.csv](fem_discretization_framework/output_data/flow_around_cylinder/characteristic_parameters_stokes.csv) containing the computed values for omega and the two operator norms.

### Step 2: Train and evaluate PINN

For this step, switch the environment to

> conda activate certified_pinn_env

Now, we can work on the PINNs

1. The data from step 3.1 needs to be entered in [flow_around_cylinder.py](flow_around_cylinder/flow_around_cylinder.py): $\omega^\ast$ needs to be entered as return value in get_Lf() and the two operator norms in get_boundary_error_factors().

2. The network can be trained using

    > python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py train -i fem_discretization_framework/output_data/flow_around_cylinder/mode_data_modes.npy

    in case some training or network parameters shall be changed, this can be done in [config_training.json](flow_around_cylinder/config_training.json) and [config_nn.json](flow_around_cylinder/config_nn.json).

3. All additional information needed for error estimation can then be retrieved using
    > python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py extract -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.0.csv       

    the input file is the data with the reference solution at time t=0.0 (which is uniformly 0 over the whole domain).

4. For each point in time, the prediction can now be evaluated independently. Exemplarily for t = 3.5 the command is

    > python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.5.csv -ae domain  -eps=0.05


### Step 3: Visualize results

Finally, we visualize the results based on the meshes we instantiated in step 1, for which we first concatenate the results calling

> python flow_around_cylinder/output_data/run_tanh_values/join_error_json.py

to then view the results in a python notebook which is created from a python file using jupytext

> jupytext --to ipynb flow_around_cylinder/output_data/run_tanh_values/plot_results.py

## References

[1] Birgit Hillebrecht and Benjamin Unger, "Prediction error certification for PINNs: Theory, computation, and application to Stokes flow", arXiv preprint 2508.07994 [available](https://doi.org/10.48550/arXiv.2508.07994).

[2] Mariella Kast, Jan S. Hesthaven. "Positional embeddings for solving PDEs with evolutional deep neural networks", Journal of Computational Physics, Volume 508, 2024, 112986, ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2024.112986.


## Contact

In case of questions, problems and ideas please contact Birgit Hillebrecht via [email](mailto:birgit.hillebrecht@kit.edu).