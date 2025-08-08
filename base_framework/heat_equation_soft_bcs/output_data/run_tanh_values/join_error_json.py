###################################################################################################
# Copyright (c) 2022 Birgit Hillebrecht
#
# To cite this code in publications, please use
#       B. Hillebrecht and B. Unger : "Certified machine learning: Rigorous a posteriori error bounds for PDE defined PINNs", arxiV preprint available
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###################################################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt 

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from helpers.csv_helpers import export_csv
from helpers.nn_parametrization import get_param_as_float, has_param
from helpers.plotting import new_fig, save_fig

if __name__ == "__main__":

    all_files = os.listdir(os.path.dirname(__file__))
    prefix =  "run_tanh_test_data_t"
    postfix =  "_error.json"

    # count available time points
    counter = 0
    for file in all_files:
        if file[-4:] == "json":    
            counter = counter+1

    # initialize data set
    data = np.zeros((counter, 13), dtype="float32")
    index = 0

    # extract data and fill it to data array
    for file in all_files:
        if file[-4:] == "json":
            head, tail = os.path.split(file)
            time = tail.replace(prefix, "")
            time = time.replace(postfix, "")
            
            data[index, 0] = float(time)
            data[index, 1] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_init")
            data[index, 2] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_PI")
            data[index, 3] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_bc")
            data[index, 4] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_bc_int_ub")
            data[index, 5] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_bc_int_ubdot")
            data[index, 6] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_bc_sum_ubt")
            data[index, 7] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_bc_sum_ub0")
            data[index, 8] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "N_SP")
            data[index, 9] = np.sum(data[index, 1:7])

            data[index, 10] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_ref")
            
            data[index, 11] = data[index, 5]/data[index, 6]
            data[index, 12] = 1.0/data[index, 11]
            index = index+1

    sortedidx = np.argsort(data[:,0])
    data = data[sortedidx,:]

    export_csv(data, os.path.join(os.path.dirname(__file__), prefix + "_error_over_time.csv"), columnheaders=np.array(["t", "E_init", "E_PI", "E_bc", "E_bc_int_ub", "E_bc_int_ubdot", "E_bc_sum_ubt", "E_bc_sum_ub0", "N_SP", "E_tot",  "E_ref", "bc_intubdot_by_sumubt", "bc_sububt_by_intubdot"]), rowheaders=np.linspace(1, data.shape[0], data.shape[0]))

    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    ax.set( ylabel=r'$E(t)$')
    ax.set( xlabel="$t$")
    ax.set (yscale="log")

    ax.grid(True, which='both')
    ax.set(xlim=[0,0.5])

    ax.plot(data[:,0:1], data[:,1:2], linewidth=2, linestyle="-", label="$E_\mathrm{init}$")
    ax.plot(data[:,0:1], data[:,2:3], linewidth=2, linestyle="--", label="$E_\mathrm{evo}$")
    ax.plot(data[:,0:1], data[:,4:5], linewidth=2, linestyle=":", label="$E_{\mathrm{bc}, \int, 1}$")
    ax.plot(data[:,0:1], data[:,5:6], linewidth=2, linestyle="--", label="$E_{\mathrm{bc}, \int, 2}$")
    ax.plot(data[:,0:1], data[:,6:7], linewidth=2, linestyle=":", label="$E_{\mathrm{bc}, t}$")
    ax.plot(data[:,0:1], data[:,7:8], linewidth=2, linestyle="--", label="$E_{\mathrm{bc}, 0}$")
    ax.plot(data[:,0:1], data[:,9:10], linewidth=2, linestyle=":", label="$E_{tot}$")
    ax.plot(data[:,0:1], data[:,10:11], linewidth=2, linestyle="-", label="$E_{ref}$")


    ax.legend(loc='best')
    fig.tight_layout()

    plt.show()
    save_fig(fig, "all_errors_plotted_0_05", os.path.dirname(__file__))