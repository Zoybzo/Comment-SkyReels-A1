#!/bin/bash

# 检查是否传入了 CUDA 设备号
if [ -z "$1" ]; then
  # 如果没有传入，默认使用设备号 0
  CUDA_DEVICE=0
else
  # 如果传入了设备号，使用传入的值
  CUDA_DEVICE=$1
fi

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# 运行 Python 脚本
CUDA_LAUNCH_BLOCKING=1 python inference.py
