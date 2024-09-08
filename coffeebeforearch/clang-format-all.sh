#!/bin/bash

# shellcheck disable=SC2038
find src -iname '*.h' -o -iname '*.cpp' -o -iname '*.cc' -o -iname '*.cu' | xargs clang-format -i
