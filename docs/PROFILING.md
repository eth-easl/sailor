Sailor is based on profiling information for the model and cluster.
* We profile a model's memory footprint and forward/backward pass under different scenarios. For each GPU type, we need 1 machine with >=1 GPUs to profile multiple tensor parallelism degrees. For LLMs, we dont profile all transformer layers since they are identical. This enables us to profile large models on a single machine.
* We profile a cluster's network bandwidth across machines using Torch NCCL-based send/recv for varying message sizes. We then fit a function to get a relation between network bandwidth and message size.

## 1. Network profiling

When adding a new GPU or cluster, the first step is to profile the network bandwidth across machines.
Sailor, as well as the rest baselines need this information. We will profile network bandwidth across GPUs on the same machine, and across machines, and update the proper files. For the remaining of this subsection, we will assume you are profiling GPU *X*, and the machine has 4 GPUs of type *X*.

### 1. Profile bandwidth + update profiling files

1. Run profiling script

Go to 'sailor/profiling/networking' directory.

For a pair of 2 GPUs:

```bash
bash run.sh 2 0 127.0.0.1 <free_port>
bash run.sh 2 1 127.0.0.1 <free_port>
```
This will generate a data2.csv file.

For a pair of 4 GPUs:

```bash
bash run.sh 4 0 127.0.0.1 <free_port>
bash run.sh 4 2 127.0.0.1 <free_port>
```

This will generate a data4.csv file.

2. Fit function to get network coefficients (used by Sailor), and max network bandwidth (used by baselines)

For data2.csv, do:

```bash
python fit_func.py data_2.csv
```
This will output a list of coefficients, and the max network bandwidth found.
Do the same for the other generated files as well.

3. Add generated info to the appropriate files

Put the coeffecients and max network bandwidth found in the [sailor/providers/intra_node_bandwidths.json](sailor/providers/intra_node_bandwidths.json) and [sailor/providers/network_coeffs_all_homo.json](sailor/providers/network_coeffs_all_homo.json) (under the "intra" keyword) for all gpu counts (look at the examples already provided).


### 2. Profile bandwidth across machines + update profiling files

1. Run profiling script

Do the same as before, but:
* Change '127.0.0.1' to an IP address/hostname discoverable across machines
* Also run for world_size=8 (assuming again 4 GPUs per node)

2. Fit function to get network coefficients (used by Sailor), and max network bandwidth (used by baselines)

Same as before

3. Add generated info to the appropriate files

Same as before, but now update:
1. the [sailor/providers/network_coeffs_all_homo.json](sailor/providers/network_coeffs_all_homo.json) under "inter"
2. the [sailor/providers/multizone_bandwidths.json](sailor/providers/multizone_bandwidths.json)

## 2. Model Profiling

### 1. Profile the model

The script *profile.sh* profiles a given model under different combinations of tensor parallelism and microbatch size. We use a big enough *largest* microbatch size, and OOM failures are expected and do not affect profiling. Of course, you can adapt this accordingly. Make sure to provide the *model_name* and *prof_dir*.


```bash
cd /root/sailor/third_party/Megatron-DeepSpeed
bash profile.sh <model_name> <prof_dir>
```
A file will be generated in the form of *profile_model_gpu_tmp_mbs.json*, where *tmp* is the Tensor Model parallel degree, and *mbs* is the microbatch size.

### 2. Generate results for Sailor

After constructing the files for the different combinations of *tmp* and *mbs*, combine them by running:

```bash

cd /root/sailor/sailor/profiling
python gather_profs.py --model-name <model> --gpu-type <gpu_type> --profile-dir <prof_dir> --sailor-parent-dir <path_to_sailor_dir>

```

This will:
1. Generate a file of the form *model/gpu/profile.json* under *sailor/Planner/sailor_planner/profiles/*, containing timing information.
2. Append memory-specific info for the model in the file *sailor/Planner/llm_info.json*.
3. Append timing-specific info for this combination of model and gpu in the file *sailor/Planner/simulations/profiles_tmp.json*, which is the file used for simulations

### 3. Generate results for baselines

If you are adding a new GPU type, make sure that network profiling is done, as mentioned above.
Also, make sure to update the GPU_MEMORY_GB dictionary in [sailor/Planner/simulations/constants.py](sailor/Planner/simulations/constants.py).
All baselines require some profiling information that we can extract from the data we already collected. Some baselines require some extra information which we collect:

### For Varuna

1. Collect the extra info:

```bash
cd /root/sailor/third_party/Megatron-DeepSpeed
bash profile_varuna.sh <model_name> <prof_dir>
```

2. Generate profiles:

```bash

cd /root/sailor/sailor/profiling/baselines
bash generate_varuna.sh
```

Adapt the GPU type, and number of GPUs per VM accordingly in the file

### For Galvatron

1. Collect the extra info:

```bash
cd /root/sailor/third_party/Megatron-DeepSpeed
bash profile_galvatron.sh <model_name> <prof_dir>
Copy the <prof_dir>/memory_profiles.json under a directory sailor/Planner/baselines/Galvatron/profiles/<model_name>/<gpu_type>
```

### For the rest baselines

Generate profiles:

```bash

cd /root/sailor/sailor/profiling/baselines
bash generate.sh
```

Adapt the name of the baseline, GPU type, and number of GPUs per machine accordingly in the file.
