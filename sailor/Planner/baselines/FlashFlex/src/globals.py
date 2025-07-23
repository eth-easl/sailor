import numpy as np
from sailor.Planner.baselines.FlashFlex.src.initialize import *
from typing import List
from sailor.Planner.baselines.FlashFlex.src.utils import *
from sailor.Planner.baselines.FlashFlex.src.device import Device
from sailor.Planner.baselines.FlashFlex.src.config import Config

np.random.seed(40404)

def update_configs(configs, machine_config_dict):
    machines = create_machines_list(machine_config_dict)

    devices: List[Device]  = create_devices(machines)

    device_machine_map = create_device_machine_map(devices)
    tensor_cores, comm_bws, comm_bws_dict = create_specs(devices, configs.inter_bw)

    reverse_comm_bws = (1 / np.array(comm_bws) * 1e10).tolist()

    configs.specs = [tensor_cores, comm_bws, comm_bws_dict, reverse_comm_bws]
    configs.device_machine_map = device_machine_map
    configs.devices = devices

    print("Scheduling Input Log============================================================")
    print("Machines:", machines)
    print("Total GPUs:", len(devices))
    print(f"Inter Bandwidth: {configs.inter_bw} GB/s", )


    print("=" * 80)

    return devices
