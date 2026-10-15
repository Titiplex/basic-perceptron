.DEFAULT_GOAL := build

BUILD_DIR := build-ninja
INSTALL_DIR := $(CURDIR)/install

ifeq ($(OS),Windows_NT)
    EXE := $(BUILD_DIR)/TP1.exe
else
    EXE := $(BUILD_DIR)/TP1
endif

.PHONY: configure build run clean

configure:
	cmake -S . -B "$(BUILD_DIR)" -G Ninja -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build "$(BUILD_DIR)" --parallel

run: build
	"./$(EXE)"

clean:
	cmake --build "$(BUILD_DIR)" --target clean