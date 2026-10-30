#ifndef LABNEURA_EDA_MAPPING_GENERIC_BEHAVIORAL_SYNTHESIS_H
#define LABNEURA_EDA_MAPPING_GENERIC_BEHAVIORAL_SYNTHESIS_H

#include "labneura/eda/netlist.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace labneura::eda::detail {

// Lowers a previously parsed Slang design without reparsing its source files.
Module lower_behavioral_rtl(
    const std::shared_ptr<ParsedVerilogContext>& context, const std::string& top,
    const std::map<std::string, std::string>& parameters,
    bool preserve_hierarchy = false);

}  // namespace labneura::eda::detail

#endif
