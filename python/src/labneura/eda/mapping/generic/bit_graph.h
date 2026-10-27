#ifndef LABNEURA_EDA_BIT_GRAPH_H
#define LABNEURA_EDA_BIT_GRAPH_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace labneura::eda::synth {

using Bit = std::int32_t;
using Word = std::vector<Bit>;  // least-significant bit first

constexpr Bit kZero = 0;
constexpr Bit kOne = 1;
constexpr Bit kLatch = 2;  // value of a signal not assigned on every path

enum class Op { Const, Latch, Input, Wire, Not, And, Or, Xor, Mux, Flop, LatchCell, Tri };

struct Node {
    Op op = Op::Const;
    Bit a = -1;  // Not/And/Or/Xor: operands; Mux: a0; Wire: alias target; Flop: D
    Bit b = -1;  // Mux: a1; Flop: clock
    Bit c = -1;  // Mux: select; Flop: async reset (active high), -1 if none
    bool negedge = false;
    std::string name;
    bool reset_value = false;  // Flop: value loaded while the async reset is active
    Bit d = -1;  // Flop with both async controls: async set (active high); c is then the reset
};
// LatchCell: a = D, b = enable (transparent while high).
// Tri: a = data, b = enable; drives high-impedance while the enable is low.

// Gate-level DAG with structural hashing and constant folding.
class BitGraph {
public:
    BitGraph() {
        nodes_.push_back({Op::Const, -1, -1, -1, false, "0"});
        nodes_.push_back({Op::Const, -1, -1, -1, false, "1"});
        nodes_.push_back({Op::Latch, -1, -1, -1, false, "latch"});
    }

    const Node& node(Bit id) const { return nodes_.at(static_cast<std::size_t>(id)); }
    Node& node(Bit id) { return nodes_.at(static_cast<std::size_t>(id)); }
    std::size_t size() const { return nodes_.size(); }

    Bit input(const std::string& name) {
        nodes_.push_back({Op::Input, -1, -1, -1, false, name});
        return last();
    }
    Bit wire() {
        nodes_.push_back({Op::Wire, -1, -1, -1, false, {}});
        return last();
    }
    Bit flop(const std::string& name, bool negedge) {
        nodes_.push_back({Op::Flop, -1, -1, -1, negedge, name});
        return last();
    }

    Bit latch_cell(const std::string& name) {
        nodes_.push_back({Op::LatchCell, -1, -1, -1, false, name, false});
        return last();
    }

    Bit tri(Bit data, Bit enable) {
        nodes_.push_back({Op::Tri, data, enable, -1, false, {}});
        return last();
    }

    Bit make_not(Bit a) {
        if (a == kZero) return kOne;
        if (a == kOne) return kZero;
        if (node(a).op == Op::Not) return node(a).a;
        return hashed(Op::Not, a, -1, -1);
    }
    Bit make_and(Bit a, Bit b) {
        if (a == kZero || b == kZero) return kZero;
        if (a == kOne) return b;
        if (b == kOne) return a;
        if (a == b) return a;
        if (is_complement(a, b)) return kZero;
        if (a > b) std::swap(a, b);
        return hashed(Op::And, a, b, -1);
    }
    Bit make_or(Bit a, Bit b) {
        if (a == kOne || b == kOne) return kOne;
        if (a == kZero) return b;
        if (b == kZero) return a;
        if (a == b) return a;
        if (is_complement(a, b)) return kOne;
        if (a > b) std::swap(a, b);
        return hashed(Op::Or, a, b, -1);
    }
    Bit make_xor(Bit a, Bit b) {
        if (a == kZero) return b;
        if (b == kZero) return a;
        if (a == kOne) return make_not(b);
        if (b == kOne) return make_not(a);
        if (a == b) return kZero;
        if (is_complement(a, b)) return kOne;
        if (a > b) std::swap(a, b);
        return hashed(Op::Xor, a, b, -1);
    }
    // select ? when_true : when_false
    Bit make_mux(Bit select, Bit when_true, Bit when_false) {
        if (select == kOne) return when_true;
        if (select == kZero) return when_false;
        if (when_true == when_false) return when_true;
        if (when_false == kZero && when_true == kOne) return select;
        if (when_false == kOne && when_true == kZero) return make_not(select);
        if (when_false == kZero) return make_and(select, when_true);
        if (when_true == kZero) return make_and(make_not(select), when_false);
        if (when_true == kOne) return make_or(select, when_false);
        if (when_false == kOne) return make_or(make_not(select), when_true);
        return hashed(Op::Mux, when_false, when_true, select);
    }

private:
    std::vector<Node> nodes_;
    std::map<std::tuple<int, Bit, Bit, Bit>, Bit> table_;

    Bit last() const { return static_cast<Bit>(nodes_.size() - 1); }
    bool is_complement(Bit a, Bit b) const {
        return (node(a).op == Op::Not && node(a).a == b) ||
               (node(b).op == Op::Not && node(b).a == a);
    }
    Bit hashed(Op op, Bit a, Bit b, Bit c) {
        const auto key = std::make_tuple(static_cast<int>(op), a, b, c);
        const auto found = table_.find(key);
        if (found != table_.end()) return found->second;
        Node created;
        created.op = op;
        created.a = a;
        created.b = b;
        created.c = c;
        nodes_.push_back(std::move(created));
        table_.emplace(key, last());
        return last();
    }
};

inline Word constant_word(std::uint64_t value, std::size_t width) {
    Word word(width);
    for (std::size_t i = 0; i < width; ++i) {
        word[i] = i < 64 && ((value >> i) & 1U) ? kOne : kZero;
    }
    return word;
}

inline std::optional<std::uint64_t> as_constant(const Word& word) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < word.size(); ++i) {
        if (word[i] == kOne) {
            if (i >= 64) return std::nullopt;
            value |= std::uint64_t{1} << i;
        } else if (word[i] != kZero) {
            return std::nullopt;
        }
    }
    return value;
}

inline Word resize(const Word& word, std::size_t width, bool sign_extend) {
    Word out(word.begin(),
             word.begin() + static_cast<std::ptrdiff_t>(std::min(width, word.size())));
    const Bit fill = sign_extend && !word.empty() ? word.back() : kZero;
    out.resize(width, fill);
    return out;
}

// Word-level operators built on top of a BitGraph.
class WordOps {
public:
    explicit WordOps(BitGraph& graph) : g(graph) {}

    Word bit_not(const Word& a) {
        Word out(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) out[i] = g.make_not(a[i]);
        return out;
    }
    Word bitwise(char op, const Word& a, const Word& b) {
        Word out(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            out[i] = op == '&'   ? g.make_and(a[i], b[i])
                     : op == '|' ? g.make_or(a[i], b[i])
                                 : g.make_xor(a[i], b[i]);
        }
        return out;
    }
    Bit reduce_or(const Word& a) {
        Bit result = kZero;
        for (const Bit bit : a) result = g.make_or(result, bit);
        return result;
    }
    Bit reduce_and(const Word& a) {
        Bit result = kOne;
        for (const Bit bit : a) result = g.make_and(result, bit);
        return result;
    }
    Bit reduce_xor(const Word& a) {
        Bit result = kZero;
        for (const Bit bit : a) result = g.make_xor(result, bit);
        return result;
    }
    Word mux(Bit select, const Word& when_true, const Word& when_false) {
        Word out(when_true.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = g.make_mux(select, when_true[i], when_false[i]);
        }
        return out;
    }
    Word add(const Word& a, const Word& b, Bit carry_in = kZero) {
        Word out(a.size());
        Bit carry = carry_in;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const Bit half = g.make_xor(a[i], b[i]);
            out[i] = g.make_xor(half, carry);
            carry = g.make_or(g.make_and(a[i], b[i]), g.make_and(half, carry));
        }
        return out;
    }
    Word sub(const Word& a, const Word& b) { return add(a, bit_not(b), kOne); }
    // Unsigned restoring division; division by zero yields all-ones / dividend.
    std::pair<Word, Word> divmod_unsigned(const Word& a, const Word& b) {
        const std::size_t n = a.size();
        Word divisor = resize(b, n + 1, false);
        Word remainder(n + 1, kZero);
        Word quotient(n, kZero);
        for (std::size_t step = n; step-- > 0;) {
            Word shifted(n + 1, kZero);
            shifted[0] = a[step];
            for (std::size_t i = 1; i <= n; ++i) shifted[i] = remainder[i - 1];
            const Bit fits = g.make_not(less_than(shifted, divisor, false));
            remainder = mux(fits, sub(shifted, divisor), shifted);
            quotient[step] = fits;
        }
        return {quotient, resize(remainder, n, false)};
    }
    // Signed division truncating toward zero; remainder takes the dividend's sign.
    std::pair<Word, Word> divmod(const Word& a, const Word& b, bool is_signed) {
        if (!is_signed || a.empty()) return divmod_unsigned(a, b);
        const Bit sign_a = a.back();
        const Bit sign_b = b.empty() ? kZero : b.back();
        const Word abs_a = mux(sign_a, negate(a), a);
        const Word abs_b = mux(sign_b, negate(b), b);
        auto [q, r] = divmod_unsigned(abs_a, abs_b);
        q = mux(g.make_xor(sign_a, sign_b), negate(q), q);
        r = mux(sign_a, negate(r), r);
        return {q, r};
    }
    Word negate(const Word& a) { return sub(constant_word(0, a.size()), a); }
    Word multiply(const Word& a, const Word& b) {
        Word result = constant_word(0, a.size());
        for (std::size_t i = 0; i < b.size() && i < a.size(); ++i) {
            Word partial(a.size(), kZero);
            for (std::size_t j = 0; i + j < a.size(); ++j) {
                partial[i + j] = g.make_and(a[j], b[i]);
            }
            result = add(result, partial);
        }
        return result;
    }
    Bit equal(const Word& a, const Word& b) {
        Bit result = kOne;
        for (std::size_t i = 0; i < a.size(); ++i) {
            result = g.make_and(result, g.make_not(g.make_xor(a[i], b[i])));
        }
        return result;
    }
    // a < b
    Bit less_than(const Word& a, const Word& b, bool is_signed) {
        Bit less = kZero;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const bool top = i + 1 == a.size() && is_signed;
            const Bit ai = top ? b[i] : a[i];
            const Bit bi = top ? a[i] : b[i];
            const Bit differ = g.make_xor(ai, bi);
            less = g.make_mux(differ, bi, less);
        }
        return less;
    }
    enum class Shift { Left, LogicalRight, ArithmeticRight };
    Word shift(const Word& a, const Word& amount, Shift kind) {
        const Bit fill = kind == Shift::ArithmeticRight && !a.empty() ? a.back() : kZero;
        Word current = a;
        Bit overflow = kZero;
        for (std::size_t level = 0; level < amount.size(); ++level) {
            if (level >= 62 || (std::uint64_t{1} << level) >= a.size()) {
                overflow = g.make_or(overflow, amount[level]);
                continue;
            }
            const std::size_t step = std::size_t{1} << level;
            Word shifted(a.size());
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (kind == Shift::Left) {
                    shifted[i] = i >= step ? current[i - step] : kZero;
                } else {
                    shifted[i] = i + step < a.size() ? current[i + step] : fill;
                }
            }
            current = mux(amount[level], shifted, current);
        }
        return mux(overflow, Word(a.size(), kind == Shift::Left ? kZero : fill),
                   current);
    }
    // options[k] selected when index == k; out-of-range yields zeros.
    Word select(const std::vector<Word>& options, const Word& index,
                std::size_t width) {
        std::vector<Word> level = options;
        Bit overflow = kZero;
        std::size_t used = 0;
        while ((std::size_t{1} << used) < std::max<std::size_t>(level.size(), 1)) ++used;
        for (std::size_t bit = 0; bit < index.size(); ++bit) {
            if (bit >= used) {
                overflow = g.make_or(overflow, index[bit]);
                continue;
            }
            std::vector<Word> next;
            for (std::size_t i = 0; i < level.size(); i += 2) {
                const Word& low = level[i];
                const Word high = i + 1 < level.size()
                                      ? level[i + 1]
                                      : constant_word(0, width);
                next.push_back(mux(index[bit], high, low));
            }
            level = std::move(next);
        }
        Word result = level.empty() ? constant_word(0, width) : level.front();
        if (used > index.size()) {
            // Not every option is addressable; remaining ones are unreachable.
        }
        return mux(overflow, constant_word(0, width), result);
    }

private:
    BitGraph& g;
};

}  // namespace labneura::eda::synth

#endif
