#ifndef LABNEURA_EDA_MAPPING_GENERIC_BEHAVIORAL_SYNTHESIS_H
#define LABNEURA_EDA_MAPPING_GENERIC_BEHAVIORAL_SYNTHESIS_H

#include "labneura/eda/netlist.h"

#include <map>
#include <string>
#include <vector>

namespace labneura::eda::detail {

// Internal helper of generic_map: elaborates `source_files` with slang and
// lowers behavioral Verilog-2005 RTL (always blocks, generate, memories,
// hierarchy) below `top` into a flat bit-level generic netlist.
Module lower_behavioral_rtl(const std::vector<std::string>& source_files,
                            const std::string& top,
                            const std::map<std::string, std::string>& parameters);

}  // namespace labneura::eda::detail

#endif
