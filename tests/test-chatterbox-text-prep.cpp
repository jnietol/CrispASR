// test-chatterbox-text-prep.cpp - regression tests for Chatterbox text prep.

#include "chatterbox_text_prep.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("chatterbox text prep normalizes punctuation and whitespace", "[unit][chatterbox][text-prep]") {
    REQUIRE(chatterbox_text_prep::normalize("  hello   world  ", false) == "Hello world.");
    REQUIRE(chatterbox_text_prep::normalize("hello...   world", false) == "Hello, world.");
    REQUIRE(chatterbox_text_prep::normalize("hello\u00a0world", false) == "Hello world.");
}

TEST_CASE("chatterbox multilingual prep lowercases ascii after punctuation normalization",
          "[unit][chatterbox][text-prep]") {
    REQUIRE(chatterbox_text_prep::normalize("  Justice   Justice  ", true) == "justice justice.");
    REQUIRE(chatterbox_text_prep::normalize("Fur, justice", true) == "fur, justice.");
}
