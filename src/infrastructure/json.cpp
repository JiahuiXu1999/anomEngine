#include "infrastructure/json.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace anom::model::json {
namespace {

class ParseError final : public std::runtime_error {
public:
    ParseError(std::string message, std::size_t line, std::size_t column)
        : std::runtime_error(std::move(message)), line(line), column(column) {}

    std::size_t line;
    std::size_t column;
};

class Parser {
public:
    explicit Parser(const std::string& input) : input_(input) {}

    Value parseDocument() {
        skipWhitespace();
        Value value = parseValue();
        skipWhitespace();
        if (!eof()) fail("Unexpected trailing content");
        return value;
    }

private:
    Value parseValue() {
        if (eof()) fail("Expected a JSON value");
        switch (peek()) {
            case '{': return parseObject();
            case '[': return parseArray();
            case '"': return Value(parseString());
            case 't': consumeLiteral("true"); return Value(true);
            case 'f': consumeLiteral("false"); return Value(false);
            case 'n': consumeLiteral("null"); return Value(nullptr);
            default:
                if (peek() == '-' || std::isdigit(static_cast<unsigned char>(peek()))) {
                    return Value(parseNumber());
                }
                fail("Unexpected character while parsing a value");
        }
    }

    Value parseObject() {
        expect('{');
        Value::Object object;
        skipWhitespace();
        if (consumeIf('}')) return Value(std::move(object));
        while (true) {
            skipWhitespace();
            if (peekOrZero() != '"') fail("Object key must be a string");
            std::string key = parseString();
            skipWhitespace();
            expect(':');
            skipWhitespace();
            auto inserted = object.emplace(std::move(key), parseValue());
            if (!inserted.second) fail("Duplicate object key");
            skipWhitespace();
            if (consumeIf('}')) break;
            expect(',');
            skipWhitespace();
        }
        return Value(std::move(object));
    }

    Value parseArray() {
        expect('[');
        Value::Array array;
        skipWhitespace();
        if (consumeIf(']')) return Value(std::move(array));
        while (true) {
            array.emplace_back(parseValue());
            skipWhitespace();
            if (consumeIf(']')) break;
            expect(',');
            skipWhitespace();
        }
        return Value(std::move(array));
    }

    std::string parseString() {
        expect('"');
        std::string output;
        while (!eof()) {
            const char ch = take();
            if (ch == '"') return output;
            if (static_cast<unsigned char>(ch) < 0x20) fail("Control character in string");
            if (ch != '\\') {
                output.push_back(ch);
                continue;
            }
            if (eof()) fail("Unterminated escape sequence");
            switch (take()) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': appendUnicode(output); break;
                default: fail("Invalid string escape");
            }
        }
        fail("Unterminated string");
    }

    void appendUnicode(std::string& output) {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            if (eof()) fail("Incomplete unicode escape");
            const char ch = take();
            value <<= 4;
            if (ch >= '0' && ch <= '9') value |= static_cast<unsigned>(ch - '0');
            else if (ch >= 'a' && ch <= 'f') value |= static_cast<unsigned>(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F') value |= static_cast<unsigned>(ch - 'A' + 10);
            else fail("Invalid unicode escape");
        }
        if (value >= 0xD800 && value <= 0xDFFF) fail("UTF-16 surrogate pairs are not supported");
        if (value <= 0x7F) output.push_back(static_cast<char>(value));
        else if (value <= 0x7FF) {
            output.push_back(static_cast<char>(0xC0 | (value >> 6)));
            output.push_back(static_cast<char>(0x80 | (value & 0x3F)));
        } else {
            output.push_back(static_cast<char>(0xE0 | (value >> 12)));
            output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (value & 0x3F)));
        }
    }

    double parseNumber() {
        const std::size_t begin = offset_;
        if (consumeIf('-') && eof()) fail("Incomplete number");
        if (consumeIf('0')) {
            if (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) fail("Leading zero in number");
        } else {
            requireDigits();
        }
        if (consumeIf('.')) requireDigits();
        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            take();
            if (!eof() && (peek() == '+' || peek() == '-')) take();
            requireDigits();
        }
        const std::string token = input_.substr(begin, offset_ - begin);
        errno = 0;
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (errno == ERANGE || end != token.c_str() + token.size() || !std::isfinite(value)) {
            fail("Invalid or out-of-range number");
        }
        return value;
    }

    void requireDigits() {
        if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) fail("Expected digit");
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) take();
    }

    void consumeLiteral(const char* literal) {
        while (*literal) {
            if (eof() || take() != *literal++) fail("Invalid literal");
        }
    }

    void skipWhitespace() {
        while (!eof() && std::isspace(static_cast<unsigned char>(peek()))) take();
    }

    void expect(char expected) {
        if (eof() || take() != expected) {
            fail(std::string("Expected '") + expected + "'");
        }
    }

    bool consumeIf(char expected) {
        if (!eof() && peek() == expected) {
            take();
            return true;
        }
        return false;
    }

    char peek() const { return input_[offset_]; }
    char peekOrZero() const { return eof() ? '\0' : peek(); }
    bool eof() const noexcept { return offset_ >= input_.size(); }

    char take() {
        const char value = input_[offset_++];
        if (value == '\n') { ++line_; column_ = 1; }
        else { ++column_; }
        return value;
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw ParseError(message, line_, column_);
    }

    const std::string& input_;
    std::size_t offset_{0};
    std::size_t line_{1};
    std::size_t column_{1};
};

void appendIndent(std::ostringstream& output, int depth) {
    output << std::string(static_cast<std::size_t>(depth) * 2, ' ');
}

void appendEscaped(std::ostringstream& output, const std::string& value) {
    output << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(ch) << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(ch);
                }
                break;
        }
    }
    output << '"';
}

void appendValue(std::ostringstream& output, const Value& value, bool pretty, int depth) {
    if (value.isNull()) {
        output << "null";
    } else if (value.isBool()) {
        output << (value.asBool() ? "true" : "false");
    } else if (value.isNumber()) {
        output << std::setprecision(std::numeric_limits<double>::max_digits10)
               << value.asNumber();
    } else if (value.isString()) {
        appendEscaped(output, value.asString());
    } else if (value.isArray()) {
        const auto& array = value.asArray();
        output << '[';
        for (std::size_t index = 0; index < array.size(); ++index) {
            if (index != 0) output << ',';
            if (pretty) output << ' ';
            appendValue(output, array[index], pretty, depth + 1);
        }
        if (pretty && !array.empty()) output << ' ';
        output << ']';
    } else {
        const auto& object = value.asObject();
        output << '{';
        if (pretty && !object.empty()) output << '\n';
        std::size_t index = 0;
        for (const auto& [key, child] : object) {
            if (index++ != 0) output << (pretty ? ",\n" : ",");
            if (pretty) appendIndent(output, depth + 1);
            appendEscaped(output, key);
            output << (pretty ? ": " : ":");
            appendValue(output, child, pretty, depth + 1);
        }
        if (pretty && !object.empty()) {
            output << '\n';
            appendIndent(output, depth);
        }
        output << '}';
    }
}

}  // namespace

bool Value::isNull() const noexcept { return std::holds_alternative<std::nullptr_t>(storage_); }
bool Value::isBool() const noexcept { return std::holds_alternative<bool>(storage_); }
bool Value::isNumber() const noexcept { return std::holds_alternative<double>(storage_); }
bool Value::isString() const noexcept { return std::holds_alternative<std::string>(storage_); }
bool Value::isArray() const noexcept { return std::holds_alternative<Array>(storage_); }
bool Value::isObject() const noexcept { return std::holds_alternative<Object>(storage_); }
bool Value::asBool() const { return std::get<bool>(storage_); }
double Value::asNumber() const { return std::get<double>(storage_); }
const std::string& Value::asString() const { return std::get<std::string>(storage_); }
const Value::Array& Value::asArray() const { return std::get<Array>(storage_); }
const Value::Object& Value::asObject() const { return std::get<Object>(storage_); }
Value::Array& Value::asArray() { return std::get<Array>(storage_); }
Value::Object& Value::asObject() { return std::get<Object>(storage_); }

const Value* Value::find(const std::string& key) const {
    if (!isObject()) return nullptr;
    const auto& object = asObject();
    const auto found = object.find(key);
    return found == object.end() ? nullptr : &found->second;
}

Value* Value::find(const std::string& key) {
    if (!isObject()) return nullptr;
    auto& object = asObject();
    const auto found = object.find(key);
    return found == object.end() ? nullptr : &found->second;
}

Result<Value> parse(const std::string& text) {
    try {
        return Parser(text).parseDocument();
    } catch (const ParseError& error) {
        std::ostringstream context;
        context << "line " << error.line << ", column " << error.column;
        return Status::error(ErrorCode::InvalidManifest, error.what(), context.str());
    } catch (const std::exception& error) {
        return Status::error(ErrorCode::InvalidManifest, error.what());
    }
}

std::string serialize(const Value& value, bool pretty) {
    std::ostringstream output;
    appendValue(output, value, pretty, 0);
    if (pretty) output << '\n';
    return output.str();
}

}  // namespace anom::model::json
