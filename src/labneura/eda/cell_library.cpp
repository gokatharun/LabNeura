#include "labneura/eda/cell_library.h"
#include "labneura/eda/default_cell_library.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace labneura::eda {
namespace {

struct Token {
    std::string text;
    std::size_t start;
    std::size_t end;
};

struct Group {
    std::string name;
    std::vector<std::string> arguments;
    std::map<std::string, std::string> attributes;
    std::map<std::string, std::vector<std::string>> list_attributes;
    std::vector<Group> children;
    bool is_attribute = false;
};

std::string trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(first, last - first + 1));
}

std::string decode_value(std::string value) {
    value = trim(value);
    if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
        return value;
    }

    std::string decoded;
    decoded.reserve(value.size() - 2);
    for (std::size_t i = 1; i + 1 < value.size(); ++i) {
        if (value[i] == '\\' && i + 2 < value.size()) ++i;
        decoded.push_back(value[i]);
    }
    return decoded;
}

class LibertyParser {
public:
    explicit LibertyParser(const std::string& source) : source_(source) {
        tokenize();
    }

    std::vector<Group> parse() {
        std::vector<Group> groups;
        while (!at_end()) {
            groups.push_back(parse_statement());
        }
        if (groups.empty()) {
            throw std::runtime_error("Liberty source contains no groups");
        }
        return groups;
    }

private:
    const std::string& source_;
    std::vector<Token> tokens_;
    std::size_t position_ = 0;

    bool at_end() const noexcept { return position_ == tokens_.size(); }
    const Token& peek() const {
        if (at_end()) fail("unexpected end of Liberty source");
        return tokens_[position_];
    }
    Token take() {
        const auto token = peek();
        ++position_;
        return token;
    }
    bool accept(std::string_view text) {
        if (!at_end() && tokens_[position_].text == text) {
            ++position_;
            return true;
        }
        return false;
    }
    void require(std::string_view text) {
        if (!accept(text)) fail("expected '" + std::string(text) + "'");
    }
    [[noreturn]] void fail(const std::string& message) const {
        const auto offset =
            at_end() ? source_.size() : tokens_[position_].start;
        throw std::runtime_error("Liberty parse error at byte " +
                                 std::to_string(offset) + ": " + message);
    }

    void tokenize() {
        std::size_t i = 0;
        while (i < source_.size()) {
            const unsigned char character =
                static_cast<unsigned char>(source_[i]);
            if (std::isspace(character)) {
                ++i;
                continue;
            }
            if (source_[i] == '/' && i + 1 < source_.size() &&
                source_[i + 1] == '/') {
                i += 2;
                while (i < source_.size() && source_[i] != '\n') ++i;
                continue;
            }
            if (source_[i] == '/' && i + 1 < source_.size() &&
                source_[i + 1] == '*') {
                const auto comment_start = i;
                i += 2;
                while (i + 1 < source_.size() &&
                       !(source_[i] == '*' && source_[i + 1] == '/')) {
                    ++i;
                }
                if (i + 1 >= source_.size()) {
                    throw std::runtime_error(
                        "Liberty parse error at byte " +
                        std::to_string(comment_start) +
                        ": unterminated block comment");
                }
                i += 2;
                continue;
            }

            const auto start = i;
            if (source_[i] == '"') {
                ++i;
                bool closed = false;
                while (i < source_.size()) {
                    if (source_[i] == '\\' && i + 1 < source_.size()) {
                        i += 2;
                    } else if (source_[i++] == '"') {
                        closed = true;
                        break;
                    }
                }
                if (!closed) {
                    throw std::runtime_error(
                        "Liberty parse error at byte " +
                        std::to_string(start) + ": unterminated string");
                }
            } else if (std::string_view("(){}:;,").find(source_[i]) !=
                       std::string_view::npos) {
                ++i;
            } else {
                while (i < source_.size() &&
                       !std::isspace(static_cast<unsigned char>(source_[i])) &&
                       std::string_view("(){}:;,\"").find(source_[i]) ==
                           std::string_view::npos) {
                    if (source_[i] == '/' && i + 1 < source_.size() &&
                        (source_[i + 1] == '/' || source_[i + 1] == '*')) {
                        break;
                    }
                    ++i;
                }
                if (i == start) ++i;
            }
            tokens_.push_back(
                {source_.substr(start, i - start), start, i});
        }
    }

    std::string token_slice(std::size_t begin, std::size_t end) const {
        if (begin >= end) return {};
        return trim(std::string_view(source_).substr(
            tokens_[begin].start, tokens_[end - 1].end - tokens_[begin].start));
    }

    std::vector<std::string> parse_arguments() {
        require("(");
        std::vector<std::string> arguments;
        std::size_t item_start = position_;
        std::size_t nesting = 0;
        while (!at_end()) {
            if (tokens_[position_].text == "(") {
                ++nesting;
                ++position_;
            } else if (tokens_[position_].text == ")") {
                if (nesting == 0) {
                    if (item_start != position_) {
                        arguments.push_back(token_slice(item_start, position_));
                    } else if (!arguments.empty()) {
                        fail("empty Liberty group argument");
                    }
                    ++position_;
                    return arguments;
                }
                --nesting;
                ++position_;
            } else if (tokens_[position_].text == "," && nesting == 0) {
                if (item_start == position_) fail("empty Liberty group argument");
                arguments.push_back(token_slice(item_start, position_));
                ++position_;
                item_start = position_;
            } else {
                ++position_;
            }
        }
        fail("unterminated parenthesized argument list");
    }

    std::string parse_scalar() {
        const auto start = position_;
        std::size_t nesting = 0;
        while (!at_end()) {
            if (tokens_[position_].text == "(") {
                ++nesting;
                ++position_;
            } else if (tokens_[position_].text == ")") {
                if (nesting == 0) fail("unexpected ')' in Liberty attribute");
                --nesting;
                ++position_;
            } else if (tokens_[position_].text == ";" && nesting == 0) {
                const auto value = decode_value(token_slice(start, position_));
                ++position_;
                return value;
            } else {
                ++position_;
            }
        }
        fail("unterminated Liberty attribute");
    }

    Group parse_statement() {
        if (peek().text == "}") fail("unexpected closing brace");
        Group result;
        result.name = take().text;

        if (accept(":")) {
            result.attributes.emplace(result.name, parse_scalar());
            result.is_attribute = true;
            return result;
        }

        if (!at_end() && peek().text == "(") {
            result.arguments = parse_arguments();
        }

        if (accept("{")) {
            while (!accept("}")) {
                if (at_end()) fail("unterminated Liberty group");
                auto child = parse_statement();
                if (child.is_attribute && !child.attributes.empty()) {
                    for (auto& [key, value] : child.attributes) {
                        result.attributes[key] = std::move(value);
                    }
                } else if (child.is_attribute &&
                           !child.list_attributes.empty()) {
                    for (auto& [key, values] : child.list_attributes) {
                        result.list_attributes[key] = std::move(values);
                    }
                } else {
                    result.children.push_back(std::move(child));
                }
            }
            accept(";");
            return result;
        }

        if (accept(";")) {
            if (!result.arguments.empty()) {
                result.list_attributes.emplace(result.name,
                                               std::move(result.arguments));
                result.is_attribute = true;
            }
            return result;
        }
        fail("expected group body or ';'");
    }
};

std::string attribute(const Group& group, std::string_view key,
                      std::string default_value = {}) {
    const auto it = group.attributes.find(std::string(key));
    return it == group.attributes.end() ? std::move(default_value) : it->second;
}

std::optional<double> numeric_attribute(const Group& group,
                                        std::string_view key) {
    const auto value = group.attributes.find(std::string(key));
    if (value == group.attributes.end()) return std::nullopt;
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stod(value->second, &consumed);
        if (consumed != value->second.size() || !std::isfinite(parsed)) {
            throw std::invalid_argument("not a finite number");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid numeric Liberty attribute '" +
                                 std::string(key) + "': " + value->second);
    }
}

std::vector<double> numeric_values(const Group& group,
                                   std::string_view key) {
    std::vector<double> result;
    const auto found = group.list_attributes.find(std::string(key));
    if (found == group.list_attributes.end()) return result;

    for (const auto& argument : found->second) {
        std::stringstream stream(decode_value(argument));
        std::string item;
        while (std::getline(stream, item, ',')) {
            item = trim(item);
            if (item.empty()) continue;
            try {
                std::size_t consumed = 0;
                const auto parsed = std::stod(item, &consumed);
                if (consumed != item.size() || !std::isfinite(parsed)) {
                    throw std::invalid_argument("not a finite number");
                }
                result.push_back(parsed);
            } catch (const std::exception&) {
                throw std::runtime_error("invalid numeric value in Liberty '" +
                                         std::string(key) + "': " + item);
            }
        }
    }
    return result;
}

std::map<std::string, std::vector<double>> collect_templates(
    const Group& library) {
    std::map<std::string, std::vector<double>> templates;
    for (const auto& child : library.children) {
        if (child.name != "lu_table_template" || child.arguments.empty()) continue;
        templates[decode_value(child.arguments.front())] =
            numeric_values(child, "index_1");
        templates[decode_value(child.arguments.front()) + "#2"] =
            numeric_values(child, "index_2");
    }
    return templates;
}

std::optional<LookupTable> parse_table(
    const Group& group,
    const std::map<std::string, std::vector<double>>& templates) {
    const auto rows = numeric_values(group, "values");
    if (rows.empty()) return std::nullopt;

    std::vector<std::vector<double>> values;
    const auto& raw_rows = group.list_attributes.at("values");
    for (const auto& raw_row : raw_rows) {
        std::vector<double> row;
        std::stringstream stream(decode_value(raw_row));
        std::string item;
        while (std::getline(stream, item, ',')) {
            item = trim(item);
            if (item.empty()) continue;
            try {
                std::size_t consumed = 0;
                const auto parsed = std::stod(item, &consumed);
                if (consumed != item.size() || !std::isfinite(parsed)) {
                    throw std::invalid_argument("not a finite number");
                }
                row.push_back(parsed);
            } catch (const std::exception&) {
                throw std::runtime_error("invalid Liberty table value: " + item);
            }
        }
        values.push_back(std::move(row));
    }

    std::vector<double> index_1 = numeric_values(group, "index_1");
    std::vector<double> index_2 = numeric_values(group, "index_2");
    const std::string template_name =
        group.arguments.empty() ? "" : decode_value(group.arguments.front());
    if (index_1.empty()) {
        const auto it = templates.find(template_name);
        if (it != templates.end()) index_1 = it->second;
    }
    if (index_2.empty()) {
        const auto it = templates.find(template_name + "#2");
        if (it != templates.end()) index_2 = it->second;
    }
    if (!index_1.empty() && !index_2.empty()) {
        if (values.size() != index_1.size()) {
            throw std::runtime_error(
                "Liberty timing table row count does not match index_1");
        }
        for (const auto& row : values) {
            if (row.size() != index_2.size()) {
                throw std::runtime_error(
                    "Liberty timing table column count does not match index_2");
            }
        }
    }
    return LookupTable(template_name, std::move(index_1), std::move(index_2),
                       std::move(values));
}

TimingArc parse_timing(
    const Group& group,
    const std::map<std::string, std::vector<double>>& templates) {
    std::optional<LookupTable> cell_rise;
    std::optional<LookupTable> cell_fall;
    std::optional<LookupTable> rise_transition;
    std::optional<LookupTable> fall_transition;
    std::optional<LookupTable> rise_constraint;
    std::optional<LookupTable> fall_constraint;
    for (const auto& child : group.children) {
        if (child.name == "cell_rise") {
            cell_rise = parse_table(child, templates);
        } else if (child.name == "cell_fall") {
            cell_fall = parse_table(child, templates);
        } else if (child.name == "rise_transition") {
            rise_transition = parse_table(child, templates);
        } else if (child.name == "fall_transition") {
            fall_transition = parse_table(child, templates);
        } else if (child.name == "rise_constraint") {
            rise_constraint = parse_table(child, templates);
        } else if (child.name == "fall_constraint") {
            fall_constraint = parse_table(child, templates);
        }
    }
    return TimingArc(attribute(group, "related_pin"),
                     attribute(group, "timing_type"),
                     attribute(group, "timing_sense"),
                     std::move(cell_rise), std::move(cell_fall),
                     std::move(rise_transition), std::move(fall_transition),
                     std::move(rise_constraint), std::move(fall_constraint));
}

CellPin parse_pin(const Group& group,
                  const std::map<std::string, std::vector<double>>& templates) {
    if (group.arguments.empty()) {
        throw std::runtime_error("Liberty pin group is missing its name");
    }
    std::vector<TimingArc> timing_arcs;
    for (const auto& child : group.children) {
        if (child.name == "timing") {
            timing_arcs.push_back(parse_timing(child, templates));
        }
    }
    return CellPin(decode_value(group.arguments.front()),
                   attribute(group, "direction"),
                   attribute(group, "function"),
                   numeric_attribute(group, "capacitance"),
                   std::move(timing_arcs));
}

StandardCell parse_cell(
    const Group& group,
    const std::map<std::string, std::vector<double>>& templates) {
    if (group.arguments.empty()) {
        throw std::runtime_error("Liberty cell group is missing its name");
    }
    std::vector<CellPin> pins;
    for (const auto& child : group.children) {
        if (child.name == "pin") pins.push_back(parse_pin(child, templates));
    }
    return StandardCell(decode_value(group.arguments.front()),
                        numeric_attribute(group, "area"), std::move(pins));
}

StandardCellLibrary parse_library(const std::string& source) {
    const auto roots = LibertyParser(source).parse();
    const Group* library = nullptr;
    for (const auto& root : roots) {
        if (root.name == "library") {
            if (library) {
                throw std::runtime_error(
                    "Liberty source contains more than one library group");
            }
            library = &root;
        }
    }
    if (!library || library->arguments.empty()) {
        throw std::runtime_error(
            "Liberty source must contain a named library group");
    }

    const auto templates = collect_templates(*library);
    std::vector<StandardCell> cells;
    for (const auto& child : library->children) {
        if (child.name == "cell") cells.push_back(parse_cell(child, templates));
    }
    if (cells.empty()) {
        throw std::runtime_error("Liberty library contains no cell groups");
    }
    std::string capacitance_unit = attribute(*library, "capacitive_load_unit");
    const auto unit = library->list_attributes.find("capacitive_load_unit");
    if (unit != library->list_attributes.end()) {
        std::ostringstream formatted;
        for (std::size_t i = 0; i < unit->second.size(); ++i) {
            if (i != 0) formatted << ", ";
            formatted << decode_value(unit->second[i]);
        }
        capacitance_unit = formatted.str();
    }
    return StandardCellLibrary(decode_value(library->arguments.front()),
                               attribute(*library, "time_unit"),
                               std::move(capacitance_unit), std::move(cells));
}

}  // namespace

StandardCellLibrary parse_liberty_source(const std::string& source) {
    return parse_library(source);
}

StandardCellLibrary parse_liberty(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("could not open Liberty file: " + path);
    }
    std::ostringstream source;
    source << input.rdbuf();
    if (input.bad()) {
        throw std::runtime_error("failed reading Liberty file: " + path);
    }
    return parse_library(source.str());
}

const StandardCellLibrary& get_default_cell_library() {
    static const StandardCellLibrary library =
        parse_liberty_source(std::string(detail::bundled_liberty));
    return library;
}

}  // namespace labneura::eda
