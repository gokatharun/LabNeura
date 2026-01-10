#pragma once

// Symbol visibility macros for shared library builds
#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef LABNEURA_BUILD_SHARED
        #define LABNEURA_API __declspec(dllexport)
    #else
        #define LABNEURA_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4 || defined(__clang__)
        #define LABNEURA_API __attribute__((visibility("default")))
    #else
        #define LABNEURA_API
    #endif
#endif
