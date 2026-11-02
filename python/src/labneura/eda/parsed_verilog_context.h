#ifndef LABNEURA_EDA_PARSED_VERILOG_CONTEXT_H
#define LABNEURA_EDA_PARSED_VERILOG_CONTEXT_H

#include "slang/ast/Compilation.h"
#include "slang/ast/Expression.h"
#include "slang/driver/Driver.h"
#include "slang/text/SourceManager.h"

#include <memory>
#include <mutex>
#include <string>

namespace labneura::eda::detail {

struct ParsedVerilogContext {
    slang::driver::Driver driver;
    std::unique_ptr<slang::ast::Compilation> compilation;
    std::mutex mutex;
};

std::string verilog_expression_source(
    const slang::ast::Expression& expression,
    const slang::SourceManager& source_manager,
    const std::string& scope_prefix = {});

}  // namespace labneura::eda::detail

#endif
