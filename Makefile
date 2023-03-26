UTEST=OFF
BUILD_EXAMPLES=OFF
BUILD_TYPE=Release
CMAKE_ARGS:=$(CMAKE_ARGS)

default:
	@export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/libtorch/lib
	@mkdir -p build
	@cd build && cmake .. -DBUILD_EXAMPLES=$(BUILD_EXAMPLES) \
                              -DBUILD_TEST=$(UTEST) \
                              -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
                              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                              $(CMAKE_ARGS)
	@cd build && make

debug:
	@make default BUILD_TYPE=Debug

apps:
	@make default BUILD_EXAMPLES=ON

debug_apps:
	@make debug BUILD_EXAMPLES=ON

unittest:
	@sudo apt-get install -y --no-install-recommends libcurl4-openssl-dev
	@export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/libtorch/lib && make default UTEST=ON && ./build/tests/image_registration_unit_tests

clean:
	@rm -rf build*
