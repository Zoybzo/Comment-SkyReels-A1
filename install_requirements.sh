#!/bin/bash

# 检查是否提供了 requirements.txt 文件路径
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_requirements.txt>"
  exit 1
fi

REQUIREMENTS_FILE="$1"

# 检查文件是否存在
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: File $REQUIREMENTS_FILE not found!"
  exit 1
fi

# 逐行读取 requirements.txt 并安装
while IFS= read -r package; do
  # 跳过空行和注释行（以 # 开头）
  if [[ -z "$package" || "$package" =~ ^# ]]; then
    continue
  fi

  echo "Installing package: $package"
  pip install "$package"

  # 检查安装是否成功
  if [ $? -eq 0 ]; then
    echo "Successfully installed: $package"
  else
    echo "Failed to install: $package"
  fi

  echo "----------------------------------------"
done < "$REQUIREMENTS_FILE"

echo "All packages processed."
