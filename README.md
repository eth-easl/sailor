# SAILOR

Sailor is a framework that automates large-scale training over heterogeneous and geo-distributed resources. It is based on our SOSP'25 paper: 'Sailor: Automating Distributed Training over Dynamic,
Heterogeneous, and Geo-distributed Clusters'

It has 3 major components:
* A *Simulator* to estimate iteration time and memory footprint
* A *Planner* to decide on resource allocation and parallelization strategies on a pool of resources
* A *Training framework* supporting heterogeneous training, based on Megatron-DeepSpeed.

## Project Structure
The project is structured as follows:

```
> tree .
├── sailor
|   ├── Planner
|   ├──   ├──  baselines           # Code for the various Planner baselines (organized by baseline name)
|   ├──   ├──  sailor_planner      # Code for the Sailor Planner
|   ├──   ├──  simulations         # Code for the Sailor Simulator
|   ├── profiling                  # Profiling code (for models and network)
|   ├── providers                  # Network bandwidth profiles, and data exchange costs (only GCP)
|   ├── models                     # Definition of models used for profiling
|   ├── Worker                     # Basic worker and checkpointing logic
|   ├── Controller                 # Code for the sailor controller (local, GKE-based, etc)
├── ae_scripts # Scripts that automate experiments and plotting (used for the SOSP'25 artifact evaluation)
├── deepspeed # Necessary modification for DeepSpeed (needed by our framework)
├── third_party/Megatron-DeepSpeed # Copy of the Megatron-DeepSpeed with modifications for the Sailor framework

```

## Environments used in the paper

* Software: We use nvcr.io/nvidia/pytorch:24.10-py3 as the base image for our container
* Hardware: We run experiments in 3 different types of clusters:
     * The [Alps Clariden](https://docs.cscs.ch/clusters/clariden/#logging-into-clariden) cluster, containing Grace-Hopper GPU nodes.
     * A cluster from the MIT university containing 8-Titan-RTX, 8-RTX-2080, and 8-RTX-3090.
     * Google Cloud, where we used A100-40 GPUs, and V100-16 GPUs (with n1-standard VMs)
Our simulator validation and plan generation experiments do not require a GPU

## Instructions for Artifact evaluation

For artifact evaluation, please go to the X branch.
Instructions for a simple functional use and reproducing key experiments from the paper are in [ArtifactEvaluation.md](ArtifactEvaluation.md)

## SAILOR image creation

You can build the SAILOR image with:

```sh
git clone https://github.com/eth-easl/sailor.git
cd sailor
docker buildx build -t <image_name> .
```

To build an image on the Alps cluster, follow the instructions in [create_image_alps.md](create_image_alps.md)


## Citation

If you use Sailor, please cite our paper:

```bibtex
@inproceedings{10.1145/3731569.3764839,
     author = {Strati, Foteini and Zhang, Zhendong and Manos, George and P\'{e}riz, Ixeia S\'{a}nchez and Hu, Qinghao and Chen, Tiancheng and Buzcu, Berk and Han, Song and Delgado, Pamela and Klimovic, Ana},
     title = {Sailor: Automating Distributed Training over Dynamic, Heterogeneous, and Geo-distributed Clusters},
     year = {2025},
     isbn = {9798400718700},
     publisher = {Association for Computing Machinery},
     address = {New York, NY, USA},
     url = {https://doi.org/10.1145/3731569.3764839},
     doi = {10.1145/3731569.3764839},
     booktitle = {Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles},
     pages = {204–220},
     numpages = {17},
     keywords = {distributed training},
     location = {Lotte Hotel World, Seoul, Republic of Korea},
     series = {SOSP '25}
}
```
