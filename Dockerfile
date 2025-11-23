FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN apt-get update
RUN apt-get install -y build-essential --no-install-recommends
RUN apt-get install -y libtbb-dev

# sailor dependencies
RUN pip install kubernetes==21.7.0
RUN pip install grpcio==1.66.0
RUN pip install grpcio-tools==1.66.0
RUN pip install google-cloud-storage==2.10.0
RUN pip install google-cloud-container==2.31.0
RUN pip install protobuf==3.19.5
RUN pip install cupy-cuda12x
RUN pip install wandb
RUN pip install datasets==3.5.0
RUN pip install transformers==4.51.0
RUN pip install skypilot[gcp]
RUN pip install google-api-python-client
RUN pip install einops
RUN pip install deepspeed==0.16.5
RUN pip install nltk

WORKDIR /root

RUN apt-get update
RUN apt-get -y install python3.10-dev iproute2 libjsoncpp-dev python3-pybind11

# Some pydantic related change
RUN pip uninstall -y pydantic pydantic-core annotated-types
RUN rm -rf /usr/local/lib/python3.10/dist-packages/pydantic*
RUN rm -rf /usr/local/lib/python3.10/dist-packages/annotated_types*
RUN pip install pydantic

COPY . ./sailor

WORKDIR /root/sailor/
RUN bash install_baselines.sh

# deepspeed

COPY deepspeed/pipe_engine.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/pipe/engine.py
COPY deepspeed/engine.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/engine.py
COPY deepspeed/config.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/config.py
COPY deepspeed/__init__.py /usr/local/lib/python3.10/dist-packages/deepspeed/__init__.py
COPY deepspeed/groups.py /usr/local/lib/python3.10/dist-packages/deepspeed/utils/
COPY deepspeed/p2p.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/pipe/p2p.py
COPY deepspeed/topology.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/pipe/topology.py
COPY deepspeed/utils.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/
COPY deepspeed/module.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/pipe/module.py
COPY deepspeed/comm.py /usr/local/lib/python3.10/dist-packages/deepspeed/comm/comm.py
COPY deepspeed/torch.py /usr/local/lib/python3.10/dist-packages/deepspeed/comm/torch.py


WORKDIR /root/sailor/third_party/Megatron-DeepSpeed
RUN python -c 'from datasets import load_dataset; ds = load_dataset("stas/oscar-en-10k", split="train", keep_in_memory=False); ds.to_json(f"data/oscar-en-10k.jsonl", orient="records", lines=True, force_ascii=False)'
RUN python tools/preprocess_data.py --input data/oscar-en-10k.jsonl --output-prefix data/meg-gpt2-oscar-en-10k --dataset-impl mmap --tokenizer-type GPT2BPETokenizer --merge-file data/gpt2-merges.txt --vocab-file data/gpt2-vocab.json --append-eod --workers 4

# for FlashFlex
RUN pip install pyomo evaluate accelerate pymetis==2023.1.1

WORKDIR /root/sailor
RUN python3 -m pip install -e .
