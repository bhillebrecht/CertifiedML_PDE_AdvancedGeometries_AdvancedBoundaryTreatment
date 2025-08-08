# Prediction error certification for PINNs: Theory, computation, and application to Stokes flow

This repository contains implementations of Finite Element Method (FEM) meshes and solvers and Physics-Informed Neural Networks (PINNs) for solving PDEs. It implements the numerical experiments from [1], when using this code, please kindly consider citing [1].

## Setup

The implementations are based on python 3.13.2 and the concrete packages used are listed in the environment files, i.e. 
> conda env create -f _envs/environment_fenics.yml

for all code working actively on FEM discretizations (based on Dolfinx 0.9.0, i.e. all in the folder fem_discretization_framework), and 

> conda env create -f _envs/environment_certified_pinn.yml

for all PINN code based on tensorflow.

## Run Heat equation example

The heat equation example does not require the usage of FEM discretizations, hence, only the PINN environment is used:

> conda activate certified_pinn_env

Training can be executed (replace the foldername by the suitable other foldernames if necessary) using 
> python base_framework/certified_pinn.py -t user -u base_framework/heat_equation_soft_bcs/heat_equation.py train -i base_framework/heat_equation_soft_bcs/input_data/initial_data.csv

the certification preparation via 
> python base_framework/certified_pinn.py -t user -u base_framework/heat_equation_soft_bcs/heat_equation.py extract -i base_framework/heat_equation_soft_bcs/input_data/initial_data.csv

and the evaluation via 
>python base_framework/certified_pinn.py -t user -u base_framework/heat_equation_soft_bcs/heat_equation.py run -i base_framework/heat_equation_soft_bcs/input_data/test_data/\<filename\> -ae domain -eps 0.1

The data is then summarized and plotted using 
> python base_framework/heat_equation_soft_bcs/output_data/run_tanh_values/join_error_json.py

The script generates two files in the folder where the script is located
1) all_errors_plotted_0_05.pdf : contains the plot of the error contributions, the total error and the reference error 
2) run_tanh_test_data_t_error_over_time.csv : contains all error contributions, total errors and reference errors in one csv file, the data stored in this file is the basis for the plot in all_errors_plotted_0_05.pdf.

The different parametrizations of the training weighting of the boundary error and its temporal derivative can be found in different folders listed in the following table.

| $\lambda_{bc}$ | Foldername |
|----------|------------|
| 0.1      | base_framework/heat_equation_soft_bcs_param4/heat_equation.py |
| 1.0      | base_framework/heat_equation_soft_bcs_param2/heat_equation.py |
| 3.14     | base_framework/heat_equation_soft_bcs_param3/heat_equation.py |
| 10.0     | base_framework/heat_equation_soft_bcs/heat_equation.py |

The examples can be run by adapting the folderpaths in the previous instructions accordingly.

## Run Stokes equation example

### Step 1: Create meshes and determine eigenmodes

This step is fully executed in the FEM environment, so get started with 

> conda activate fenics_env

Then, there are three main scripts which need to be used 

1) in [generate_data.py](fem_discretization_framework/input_data/flow_around_cylinder/generate_data.py) the meshes are generated. For the larger mesh for the eigenfunctions, set the offset to 0.5 in ln 50, i.e. 
    > x_buffer = 0.5
    
    before running the script. The script can be converted to a python notebook using jupytext

    > jupytext --to ipynb fem_discretization_framework/input_data/flow_around_cylinder/generate_data.py

3) in [compute_reference_solution_from_dolfinx_tutorial.py](fem_discretization_framework/input_data/flow_around_cylinder/compute_reference_solution_from_dolfinx_tutorial.py), the DolfinX tutorial was adapted for our problem. The file can be converted into a python notebook using jupytext 

    > jupytext --to ipynb fem_discretization_framework/input_data/flow_around_cylinder/compute_reference_solution_from_dolfinx_tutorial.py

    running the script will generate additional data which is necessary for a later visualization in accordance with the FEM visualization.

3) in [flow_around_cylinder_compute_harmonic_features.py](fem_discretization_framework/src/flow_around_cylinder/flow_around_cylinder_compute_harmonic_features.py), the generated meshes from [generate_data.py](fem_discretization_framework/input_data/flow_around_cylinder/generate_data.py) are used to determine the eigenmodes, store the eigenmodes and evaluate the eigenmodes suitably for the NN to train the equation and the boundary conditions. The implementation was adapted from [2].

4) The script [characteristic_params.py](fem_discretization_framework/src/flow_around_cylinder/characteristic_params.py) can be converted to a python notebook using jupytext

    > jupytext --to py fem_discretization_framework/src/flow_around_cylinder/characteristic_params.py

    Executing this script generates a csv table [characteristic_parameters_stokes.csv](fem_discretization_framework/output_data/flow_around_cylinder/characteristic_parameters_stokes.csv) containing the computed values for omega and the two operator norms.

    Please note, that the number of refinements is limited in ln. 101 to 10 (i.e. n_refine = 10), in the publication 42 refinements were used (i.e. until 100_000 were in the refined mesh).

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

Finally, we visualize the results based on the meshes we instantiated in step 1, i.e. switch to 

> conda activate new_fenics

We now first concatenate the results calling

> python flow_around_cylinder/output_data/run_tanh_values/join_error_json.py

to then view the results in a python notebook which is created from a python file using jupytext

> jupytext --to ipynb flow_around_cylinder/output_data/run_tanh_values/plot_results.py

## References

[1] Birgit Hillebrecht and Benjamin Unger, "Prediction error certification for PINNs: Theory, computation, and application to Stokes flow", arXiv preprint available.

[2] Mariella Kast, Jan S. Hesthaven. "Positional embeddings for solving PDEs with evolutional deep neural networks", Journal of Computational Physics, Volume 508, 2024, 112986, ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2024.112986.
