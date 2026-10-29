#ifndef LABNEURA_EDA_PARSED_VERILOG_CONTEXT_H
#define LABNEURA_EDA_PARSED_VERILOG_CONTEXT_H

#include "slang/ast/Compilation.h"
#include "slang/driver/Driver.h"

#include <memory>
#include <mutex>

namespace labneura::eda::detail {

struct ParsedVerilogContext {
    slang::driver::Driver driver;
    std::unique_ptr<slang::ast::Compilation> compilation;
    std::mutex mutex;
};

}  // namespace labneura::eda::detail

#endif
