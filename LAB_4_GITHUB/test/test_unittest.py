"""
Unittest tests for text_analyzer module
"""

import unittest
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


class TestTextAnalyzer(unittest.TestCase):
    """Test suite for text analyzer functions"""
    
    def test_count_words_simple(self):
        """Test word counting with simple text"""
        self.assertEqual(count_words("Hello world"), 2)
        self.assertEqual(count_words("This is a test"), 4)
    
    def test_count_words_empty(self):
        """Test word counting with empty string"""
        self.assertEqual(count_words(""), 0)
        self.assertEqual(count_words("   "), 0)
    
    def test_count_words_single_word(self):
        """Test word counting with single word"""
        self.assertEqual(count_words("Hello"), 1)
    
    def test_count_characters_with_spaces(self):
        """Test character counting including spaces"""
        self.assertEqual(count_characters("Hello"), 5)
        self.assertEqual(count_characters("Hello world"), 11)
    
    def test_count_characters_without_spaces(self):
        """Test character counting excluding spaces"""
        self.assertEqual(count_characters("Hello world", include_spaces=False), 10)
        self.assertEqual(count_characters("Test text", include_spaces=False), 8)
    
    def test_count_sentences_simple(self):
        """Test sentence counting"""
        self.assertEqual(count_sentences("Hello."), 1)
        self.assertEqual(count_sentences("Hello. How are you?"), 2)
        self.assertEqual(count_sentences("Great! Awesome! Fantastic!"), 3)
    
    def test_count_sentences_empty(self):
        """Test sentence counting with empty string"""
        self.assertEqual(count_sentences(""), 0)
    
    def test_get_word_frequency(self):
        """Test word frequency analysis"""
        text = "hello world hello python python python"
        freq = get_word_frequency(text, top_n=2)
        self.assertEqual(freq[0], ('python', 3))
        self.assertEqual(freq[1], ('hello', 2))
    
    def test_get_word_frequency_empty(self):
        """Test word frequency with empty string"""
        self.assertEqual(get_word_frequency(""), [])
    
    def test_calculate_average_word_length(self):
        """Test average word length calculation"""
        self.assertEqual(calculate_average_word_length("hi my"), 2.0)
        self.assertEqual(calculate_average_word_length("hello world"), 5.0)
    
    def test_calculate_average_word_length_empty(self):
        """Test average word length with empty string"""
        self.assertEqual(calculate_average_word_length(""), 0.0)
    
    def test_analyze_text_comprehensive(self):
        """Test comprehensive text analysis"""
        text = "Hello world. This is a test."
        result = analyze_text(text)
        
        self.assertEqual(result['word_count'], 6)
        self.assertEqual(result['sentence_count'], 2)
        self.assertIn('top_words', result)
        self.assertIn('average_word_length', result)
    
    def test_type_errors(self):
        """Test that functions raise TypeError for non-string input"""
        with self.assertRaises(TypeError):
            count_words(123)
        
        with self.assertRaises(TypeError):
            count_characters(None)
        
        with self.assertRaises(TypeError):
            count_sentences(['list', 'of', 'words'])


if __name__ == '__main__':
    unittest.main()