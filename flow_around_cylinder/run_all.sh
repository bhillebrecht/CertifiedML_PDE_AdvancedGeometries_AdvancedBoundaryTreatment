#!/bin/sh

#conda activate certified_pinn_env
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py extract -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.0.csv       

python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.0.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.1.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.2.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.3.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.4.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.5.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.6.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.7.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.8.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_0.9.csv -ae domain -eps=0.05

python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.0.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.1.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.2.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.3.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.4.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.5.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.6.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.7.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.8.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_1.9.csv -ae domain -eps=0.05

python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.0.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.1.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.2.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.3.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.4.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.5.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.6.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.7.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.8.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_2.9.csv -ae domain -eps=0.05

python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.0.csv -ae domain -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.1.csv -ae domain  -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.2.csv -ae domain  -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.3.csv -ae domain  -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.4.csv -ae domain  -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.5.csv -ae domain  -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.6.csv -ae domain  -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.7.csv -ae domain  -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.8.csv -ae domain  -eps=0.05
python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_3.9.csv -ae domain  -eps=0.05

python base_framework/certified_pinn.py -t user -u flow_around_cylinder/flow_around_cylinder.py run -i fem_discretization_framework/input_data/flow_around_cylinder/reference_solution/input_for_evaluation_t_4.0.csv -ae domain -eps=0.05

python flow_around_cylinder/output_data/run_tanh_values/join_error_json.py

