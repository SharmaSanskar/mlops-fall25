"""
Pytest tests for text_analyzer module
"""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from text_analyzer import (
    count_words,
    count_characters,
    count_sentences,
    get_word_frequency,
    calculate_average_word_length,
    analyze_text
)


def test_count_words_simple():
    """Test word counting with simple text"""
    assert count_words("Hello world") == 2
    assert count_words("This is a test") == 4


def test_count_words_empty():
    """Test word counting with empty string"""
    assert count_words("") == 0
    assert count_words("   ") == 0


def test_count_words_single_word():
    """Test word counting with single word"""
    assert count_words("Hello") == 1


def test_count_characters_with_spaces():
    """Test character counting including spaces"""
    assert count_characters("Hello") == 5
    assert count_characters("Hello world") == 11


def test_count_characters_without_spaces():
    """Test character counting excluding spaces"""
    assert count_characters("Hello world", include_spaces=False) == 10
    assert count_characters("Test text", include_spaces=False) == 8


def test_count_sentences_simple():
    """Test sentence counting"""
    assert count_sentences("Hello.") == 1
    assert count_sentences("Hello. How are you?") == 2
    assert count_sentences("Great! Awesome! Fantastic!") == 3


def test_count_sentences_empty():
    """Test sentence counting with empty string"""
    assert count_sentences("") == 0


def test_get_word_frequency():
    """Test word frequency analysis"""
    text = "hello world hello python python python"
    freq = get_word_frequency(text, top_n=2)
    assert freq[0] == ('python', 3)
    assert freq[1] == ('hello', 2)


def test_get_word_frequency_empty():
    """Test word frequency with empty string"""
    assert get_word_frequency("") == []


def test_calculate_average_word_length():
    """Test average word length calculation"""
    assert calculate_average_word_length("hi my") == 2.0
    assert calculate_average_word_length("hello world") == 5.0


def test_calculate_average_word_length_empty():
    """Test average word length with empty string"""
    assert calculate_average_word_length("") == 0.0


def test_analyze_text_comprehensive():
    """Test comprehensive text analysis"""
    text = "Hello world. This is a test."
    result = analyze_text(text)
    
    assert result['word_count'] == 6
    assert result['sentence_count'] == 2
    assert 'top_words' in result
    assert 'average_word_length' in result


def test_type_errors():
    """Test that functions raise TypeError for non-string input"""
    with pytest.raises(TypeError):
        count_words(123)
    
    with pytest.raises(TypeError):
        count_characters(None)
    
    with pytest.raises(TypeError):
        count_sentences(['list', 'of', 'words'])


# Parametrized test example
@pytest.mark.parametrize("text,expected", [
    ("one", 1),
    ("one two", 2),
    ("one two three", 3),
    ("", 0),
])
def test_count_words_parametrized(text, expected):
    """Parametrized test for word counting"""
    assert count_words(text) == expected