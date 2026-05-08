// Force-included via CMake when building caspar_lib_core on Windows.
//
// 1. The POSIX `uint` typedef is exposed transitively via <sys/types.h> on
//    glibc/Linux but absent on MSVC. SymForce-generated CUDA headers
//    (memops.cuh and friends) use it. A typedef (not `#define uint
//    unsigned`) is required: a macro replacement would also rewrite
//    `uint1`/`uint2` etc. inside CUDA's own <vector_types.h> token-paste
//    expansions, breaking tuple_size specializations.
//
// 2. SymForce-generated host code (solver.cc) calls std::to_string and
//    constructs std::runtime_error without explicitly including <string> /
//    <stdexcept>. Linux libstdc++ pulls these in transitively through other
//    standard headers, but MSVC's STL keeps them isolated. Force-including
//    them here avoids patching upstream sources.
#pragma once

#ifdef _WIN32

#include <stdexcept>
#include <string>

#ifndef SYMFORCE_CASPAR_UINT_COMPAT
#define SYMFORCE_CASPAR_UINT_COMPAT
typedef unsigned int uint;
#endif

#endif  // _WIN32
