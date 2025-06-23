#!/usr/bin/env python3
"""
Script to clean conversation datasets by fixing common issues.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

def clean_text(text: str) -> str:
    """Clean problematic characters from text"""
    
    # Replace smart quotes with regular quotes
    text = text.replace(''', "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace(''', "'")  # LEFT SINGLE QUOTATION MARK
    text = text.replace('"', '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace('"', '"')  # RIGHT DOUBLE QUOTATION MARK
    
    # Replace special dashes with regular dash
    text = text.replace('—', '-')  # EM DASH
    text = text.replace('–', '-')  # EN DASH
    
    # Remove zero-width characters
    zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
    for char in zero_width_chars:
        text = text.replace(char, '')
    
    # Remove control characters (except newline, tab, carriage return)
    cleaned = []
    for char in text:
        code = ord(char)
        # Keep printable characters and common whitespace
        if code >= 32 or code in [9, 10, 13]:  # tab, newline, carriage return
            cleaned.append(char)
        # Skip other control characters
    
    text = ''.join(cleaned)
    
    # Fix multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix multiple newlines (keep maximum 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def clean_conversation(conversation: List[Dict]) -> List[Dict]:
    """Clean a single conversation"""
    cleaned_conv = []
    
    for message in conversation:
        if isinstance(message, dict) and 'content' in message:
            cleaned_message = message.copy()
            cleaned_message['content'] = clean_text(message['content'])
            
            # Ensure role is valid
            if 'role' in cleaned_message:
                role = cleaned_message['role'].lower()
                if role in ['user', 'assistant', 'system']:
                    cleaned_message['role'] = role
                else:
                    cleaned_message['role'] = 'assistant'  # Default to assistant
            
            cleaned_conv.append(cleaned_message)
    
    return cleaned_conv

def process_file(input_path: str, output_path: str):
    """Process the entire JSONL file"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"Processing {input_path}...")
    
    total_lines = 0
    cleaned_lines = 0
    error_lines = 0
    
    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if line_num % 1000 == 0:
                print(f"Processed {line_num} conversations...")
            
            total_lines += 1
            line = line.strip()
            
            if not line:
                continue
            
            try:
                # Parse JSON
                conversation = json.loads(line)
                
                # Clean the conversation
                if isinstance(conversation, list) and len(conversation) > 0:
                    cleaned_conv = clean_conversation(conversation)
                    
                    # Only keep conversations with at least one user and one assistant message
                    has_user = any(msg.get('role') == 'user' for msg in cleaned_conv)
                    has_assistant = any(msg.get('role') == 'assistant' for msg in cleaned_conv)
                    
                    if has_user and has_assistant and len(cleaned_conv) >= 2:
                        # Write cleaned conversation
                        json.dump(cleaned_conv, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        cleaned_lines += 1
                
            except json.JSONDecodeError as e:
                error_lines += 1
                print(f"Error on line {line_num}: {e}")
            except Exception as e:
                error_lines += 1
                print(f"Unexpected error on line {line_num}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Total lines: {total_lines:,}")
    print(f"Cleaned lines: {cleaned_lines:,}")
    print(f"Error lines: {error_lines:,}")
    print(f"Removed lines: {total_lines - cleaned_lines - error_lines:,}")
    print(f"Output saved to: {output_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python clean_conversations.py <input_file> <output_file>")
        print("Example: python clean_conversations.py conversations.jsonl conversations_cleaned.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    process_file(input_file, output_file)

if __name__ == "__main__":
    main() 