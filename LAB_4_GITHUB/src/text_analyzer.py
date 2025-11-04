"""
Text Analysis Tool
This module provides functions to analyze text data including word count,
character count, sentence analysis, and basic sentiment detection.
"""

import re
from collections import Counter


def count_words(text):
    """
    Count the number of words in the text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        int: Number of words in the text
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    if not text.strip():
        return 0
    
    words = text.split()
    return len(words)


def count_characters(text, include_spaces=True):
    """
    Count the number of characters in the text.
    
    Args:
        text (str): Input text to analyze
        include_spaces (bool): Whether to include spaces in the count
        
    Returns:
        int: Number of characters
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(" ", ""))


def count_sentences(text):
    """
    Count the number of sentences in the text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        int: Number of sentences
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    if not text.strip():
        return 0
    
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)


def get_word_frequency(text, top_n=5):
    """
    Get the most common words in the text.
    
    Args:
        text (str): Input text to analyze
        top_n (int): Number of top words to return
        
    Returns:
        list: List of tuples (word, frequency)
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    if not text.strip():
        return []
    
    # Convert to lowercase and split
    words = text.lower().split()
    # Remove punctuation from words
    words = [re.sub(r'[^\w\s]', '', word) for word in words]
    # Filter out empty strings
    words = [w for w in words if w]
    
    # Count frequencies
    word_counts = Counter(words)
    return word_counts.most_common(top_n)


def calculate_average_word_length(text):
    """
    Calculate the average length of words in the text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        float: Average word length
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    if not text.strip():
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    
    total_length = sum(len(word) for word in words)
    return round(total_length / len(words), 2)


def analyze_text(text):
    """
    Perform comprehensive text analysis.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Dictionary containing all analysis metrics
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    return {
        'word_count': count_words(text),
        'character_count': count_characters(text, include_spaces=True),
        'character_count_no_spaces': count_characters(text, include_spaces=False),
        'sentence_count': count_sentences(text),
        'average_word_length': calculate_average_word_length(text),
        'top_words': get_word_frequency(text, top_n=5)
    }