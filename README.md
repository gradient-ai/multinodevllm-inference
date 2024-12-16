# multinodevllm-inference
A step-by-step tutorial to setup multinode inference on two baremetal H100x8.

## Introduction 
This document aims to provide guidelines for running Llama-3.1-405B-Instruct on its original FP16 on H100. As H100x8 only comes with 640GB GPU Memory, but the model requires [more than that](https://www.substratus.ai/blog/llama-3-1-405b-gpu-requirements). Therefore, a multinode inference is necessary to run the model in H100x8 machines. This tutorial aims to provide step-by-step instruction on how to run the model in vllm under 2 H100x8 nodes in DigitalOcean Baremetal product.

## Prereq (IMPORTANT! Please read)
- Install miniconda if it's not alredy installed. Following Instruction from [Miniconda official website](https://docs.anaconda.com/miniconda/install/)
- Install Docker if it's not already installed. Following Instructions from [Official Docker Website](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).
- Nvidia H100 with Nvidia drivers pre-installed. The nodes should be multinode ready with NVLink and GPU Fabric enabled. The Baremetals would have equipped with these by default. For Droplets please refer to the [multinode docs](https://docs.digitalocean.com/products/droplets/how-to/gpu/configure-multi-node/).
- Install nvidia-container-toolkit if it's not already installed. Following instructions as per [Nvidia official website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt). 
- If you have just installed any drivers, docker, or nvidia-container-toolkit, please reboot the server `sudo reboot` before continuing to the next section.

## Steps to run the Cluster and serve the model 
- Clone the vllm repo into your current directory `git clone https://github.com/vllm-project/vllm.git`
- Pick a head node, and identify the IP of the head node to be used for GPU Fabric Operations. Run `ip -br a` to identify the IP to be used. For Baremetals, it would be the private IP of the node (look for **bond0** that comes with **10.x.x.x**). For Droplets, it would be the fabric IP (use the IP address of **eth2**)
- Run `bash vllm/examples/run_cluster.sh vllm/vllm-openai head_ip --head /root/meta-llama -e NCCL_SOCKET_IFNAME=bond0 -e GLOO_SOCKET_IFNAME=bond0 --privileged -e NCCL_IB_HCA=mlx5` on **head** node, and `bash vllm/examples/run_cluster.sh vllm/vllm-openai head_ip --worker /root/meta-llama -e NCCL_SOCKET_IFNAME=bond0 -e GLOO_SOCKET_IFNAME=bond0 --privileged -e NCCL_IB_HCA=mlx5` on worker node. Replace **head_ip** with the real head IP address. For **Droplet Users**, replace **bond0** with **eth2**.
- To validate if the previous step is ran successfully, you should see `Ray runtime started.` printed out in the terminal for both nodes.
- Choose any node, run `docker exec -it node /bin/bash` and then run `ray status`, you should observe two nodes in the **Active:** printed out in the terminal, and right number of GPUs (For 2 H100X8, it should be 16).
- Inside the container, login to huggingface `huggingface-cli login`. Then serve the model by running `vllm serve meta-llama/Llama-3.1-405B-Instruct --enforce-eager --tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray --dtype float16 --gpu-memory-utilization 0.99  --max-model-len 8192 --max-num-batched-tokens 65536 --enable-chunked-prefill=False`. Make sure your Huggingface account has access to the **meta-llama/Llama-3.1-405B-Instruct** model.
- If it's first time running the model, it will take a while to download the model (there are 191 of size 3-4GB files to download). If it's running successfully, you should see the message `Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)` in the terminal.

## Steps to benchmark the model
- Go to the node where the **vllm serve** command is ran in the previous section, Clone the vllm git repo `git clone https://github.com/vllm-project/vllm.git` into your current directory, and run `python3 vllm/benchmarks/benchmark_serving.py --backend vllm --port 8000 --model meta-llama/Llama-3.1-405B-Instruct --dataset-name random --random-input-len 128 --random-output-len 128 --max-concurrency 16 --num-prompts 1000 --percentile-metrics ttft,tpot,itl,e2el`, adjust random-input-len, random-output-len, max-concurrency, and num-prompt as necessary. For max throughput, set max-concurrency to 256 (but this will lead to increase in TTFT). For more accuracy, increase num-prompts (but this will lead to longer test period).

## Disclaimer
- Please note that this should be used for tutorial purpose only. For running the model in production, additional testing and validation is required from your end. 
- The steps are **only tested on our H100x8 Baremetal machines**.
- We have not employed any other optimizations such as Speculative Decoding in this tutorial. If you would like to see other optimizations, please raise an Enhancement under GitHub Issues and we may consider it for the future.
- If there are any issues you have encountered while following the tutorial, please raise a bug in GitHub Issue.

## References
- This doc extensively references the [vllm multinode training official docs](https://docs.vllm.ai/en/latest/serving/distributed_serving.html).
