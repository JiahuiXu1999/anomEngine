#pragma once

#include "model/result.h"

#include <cstddef>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace anom::model::json {

class Value {
public:
    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>;

    Value() = default;
    explicit Value(std::nullptr_t) : storage_(nullptr) {}
    explicit Value(bool value) : storage_(value) {}
    explicit Value(double value) : storage_(value) {}
    explicit Value(std::string value) : storage_(std::move(value)) {}
    explicit Value(Array value) : storage_(std::move(value)) {}
    explicit Value(Object value) : storage_(std::move(value)) {}

    [[nodiscard]] bool isNull() const noexcept;
    [[nodiscard]] bool isBool() const noexcept;
    [[nodiscard]] bool isNumber() const noexcept;
    [[nodiscard]] bool isString() const noexcept;
    [[nodiscard]] bool isArray() const noexcept;
    [[nodiscard]] bool isObject() const noexcept;

    [[nodiscard]] bool asBool() const;
    [[nodiscard]] double asNumber() const;
    [[nodiscard]] const std::string& asString() const;
    [[nodiscard]] const Array& asArray() const;
    [[nodiscard]] const Object& asObject() const;
    [[nodiscard]] Array& asArray();
    [[nodiscard]] Object& asObject();

    [[nodiscard]] const Value* find(const std::string& key) const;
    [[nodiscard]] Value* find(const std::string& key);

private:
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> storage_{nullptr};
};

Result<Value> parse(const std::string& text);
std::string serialize(const Value& value, bool pretty = true);

}  // namespace anom::model::json
