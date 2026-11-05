#pragma once

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <cerrno>
#endif

namespace labneura::test {

inline void set_environment_variable(const char* name,
                                     const std::optional<std::string>& value) {
#if defined(_WIN32)
    if (_putenv_s(name, value ? value->c_str() : "") != 0) {
        throw std::runtime_error(std::string("Failed to set environment variable ") +
                                 name);
    }
#else
    const int result = value ? setenv(name, value->c_str(), 1)
                             : unsetenv(name);
    if (result != 0) {
        throw std::runtime_error(std::string("Failed to set environment variable ") +
                                 name);
    }
#endif
}

class ScopedEnvironmentVariable {
public:
    ScopedEnvironmentVariable(const char* name,
                              std::optional<std::string> value)
        : name_(name) {
        if (const char* current = std::getenv(name)) {
            previous_ = current;
        }
        set_environment_variable(name_.c_str(), value);
    }

    ~ScopedEnvironmentVariable() noexcept {
        try {
            set_environment_variable(name_.c_str(), previous_);
        } catch (...) {
            std::terminate();
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
    ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

private:
    std::string name_;
    std::optional<std::string> previous_;
};

}  // namespace labneura::test
