#pragma once

// Symbol visibility macros for shared library builds
// For Python extensions, we always export symbols
#if defined(_WIN32) || defined(__CYGWIN__)
    #define LABNEURA_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
    #define LABNEURA_API __attribute__((visibility("default")))
#else
    #define LABNEURA_API
#endif
