# 定义编译器和选项
CC = gcc
CFLAGS = -Wall -O3 -g -Wincompatible-pointer-types -fPIC

# 定义源文件、目标目录和可执行文件
SOURCES = solve.c table.c optimizer.c main.c color_detect.c
BUILD_DIR = build
EXECUTABLE = demo

# 根据平台设置库扩展名
ifeq ($(OS),Windows_NT)
    LIBRARY = libcube.dll
    EXECUTABLE := $(EXECUTABLE).exe
    LDFLAGS_SHARED = -shared
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        LIBRARY = libcube.so
        LDFLAGS_SHARED = -shared
    endif
    ifeq ($(UNAME_S),Darwin)
        LIBRARY = libcube.dylib
        LDFLAGS_SHARED = -dynamiclib
    endif
endif

# 生成目标文件列表
OBJECTS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SOURCES))

# 默认规则：创建build目录并构建可执行文件+动态库
all: $(BUILD_DIR) $(EXECUTABLE) $(LIBRARY)

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 链接规则：生成可执行文件
$(EXECUTABLE): $(OBJECTS)
	$(CC) $^ -o $@

# 链接规则：生成动态库
$(LIBRARY): $(OBJECTS)
	$(CC) $(LDFLAGS_SHARED) $^ -o $@

# 包含自动生成的依赖关系
-include $(OBJECTS:.o=.d)

# 编译规则 + 生成依赖关系
$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -MMD -c $< -o $@

# 清理规则
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE) $(LIBRARY)

.PHONY: all clean