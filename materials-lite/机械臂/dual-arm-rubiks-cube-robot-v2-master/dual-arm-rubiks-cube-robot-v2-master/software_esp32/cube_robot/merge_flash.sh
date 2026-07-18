#!/bin/bash

# 获取当前时间戳（格式：年月日）
CURRENT_TIME=$(date +%Y%m%d)

# 设置build目录路径
BUILD_DIR="build"
OUTPUT_FILE="merged-flash-${CURRENT_TIME}.bin"
FLASH_ARGS_FILE="${BUILD_DIR}/flash_args"

echo "进入 ${BUILD_DIR} 目录"
cd "${BUILD_DIR}"
echo "开始合并固件: ${OUTPUT_FILE}"

# 执行合并命令
esptool.py --chip ESP32-S3 merge_bin -o "${OUTPUT_FILE}" @flash_args
echo "可用如下指令烧写: esptool.py write_flash 0x0 ${BUILD_DIR}/${OUTPUT_FILE}"
# 返回上级目录
cd - > /dev/null