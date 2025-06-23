#!/usr/bin/env python3
"""
Script to detect weird, corrupted, or problematic characters in conversation datasets.
Useful for cleaning data before LLM fine-tuning.
"""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

# Common problematic character categories
CONTROL_CHARS = set(range(0x00, 0x20)) | {0x7F}  # ASCII control characters
ZERO_WIDTH_CHARS = {0x200B, 0x200C, 0x200D, 0xFEFF}  # Zero-width characters
REPLACEMENT_CHAR = 0xFFFD  # Unicode replacement character
PRIVATE_USE_AREAS = [
    (0xE000, 0xF8FF),    # Private Use Area
    (0xF0000, 0xFFFFF),  # Supplementary Private Use Area-A
    (0x100000, 0x10FFFF) # Supplementary Private Use Area-B
]

class ConversationAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.issues = defaultdict(list)
        self.stats = {
            'total_conversations': 0,
            'total_messages': 0,
            'json_errors': 0,
            'encoding_errors': 0,
            'conversations_with_issues': 0
        }
        self.char_frequency = Counter()
        self.suspicious_patterns = []
        
    def analyze(self):
        """Main analysis function"""
        print(f"Analyzing {self.file_path}...")
        
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} conversations...")
                
                self.analyze_line(line.strip(), line_num)
        
        self.generate_report()
    
    def analyze_line(self, line: str, line_num: int):
        """Analyze a single line from the JSONL file"""
        if not line:
            return
        
        self.stats['total_conversations'] += 1
        
        # Try to parse JSON
        try:
            conversation = json.loads(line)
        except json.JSONDecodeError as e:
            self.stats['json_errors'] += 1
            self.issues['json_errors'].append({
                'line': line_num,
                'error': str(e),
                'preview': line[:100] + '...' if len(line) > 100 else line
            })
            return
        
        # Analyze messages in conversation
        if isinstance(conversation, list):
            has_issues = False
            for msg_idx, message in enumerate(conversation):
                if isinstance(message, dict) and 'content' in message:
                    content = message.get('content', '')
                    role = message.get('role', 'unknown')
                    
                    self.stats['total_messages'] += 1
                    
                    # Check for various issues
                    issues = self.check_text_issues(content)
                    if issues:
                        has_issues = True
                        self.issues['message_issues'].append({
                            'line': line_num,
                            'message_index': msg_idx,
                            'role': role,
                            'issues': issues,
                            'preview': content[:100] + '...' if len(content) > 100 else content
                        })
            
            if has_issues:
                self.stats['conversations_with_issues'] += 1
    
    def check_text_issues(self, text: str) -> List[Dict]:
        """Check for various text issues"""
        issues = []
        
        # 1. Check for control characters
        control_chars = self.find_control_characters(text)
        if control_chars:
            issues.append({
                'type': 'control_characters',
                'details': control_chars
            })
        
        # 2. Check for zero-width characters
        zero_width = self.find_zero_width_characters(text)
        if zero_width:
            issues.append({
                'type': 'zero_width_characters',
                'details': zero_width
            })
        
        # 3. Check for replacement characters (encoding issues)
        if '\ufffd' in text:
            count = text.count('\ufffd')
            issues.append({
                'type': 'replacement_characters',
                'count': count,
                'positions': [i for i, c in enumerate(text) if c == '\ufffd'][:10]  # First 10
            })
        
        # 4. Check for private use characters
        private_use = self.find_private_use_characters(text)
        if private_use:
            issues.append({
                'type': 'private_use_characters',
                'details': private_use
            })
        
        # 5. Check for unusual Unicode blocks
        unusual_blocks = self.find_unusual_unicode_blocks(text)
        if unusual_blocks:
            issues.append({
                'type': 'unusual_unicode_blocks',
                'blocks': unusual_blocks
            })
        
        # 6. Check for suspicious patterns
        patterns = self.find_suspicious_patterns(text)
        if patterns:
            issues.append({
                'type': 'suspicious_patterns',
                'patterns': patterns
            })
        
        # Update character frequency
        self.char_frequency.update(text)
        
        return issues
    
    def find_control_characters(self, text: str) -> List[Tuple[int, str]]:
        """Find ASCII control characters (except common ones like \\n, \\r, \\t)"""
        allowed_control = {0x09, 0x0A, 0x0D}  # tab, newline, carriage return
        found = []
        
        for i, char in enumerate(text):
            code = ord(char)
            if code in CONTROL_CHARS and code not in allowed_control:
                found.append((i, f'U+{code:04X}'))
        
        return found[:10]  # Return first 10 occurrences
    
    def find_zero_width_characters(self, text: str) -> List[Tuple[int, str]]:
        """Find zero-width characters"""
        found = []
        
        for i, char in enumerate(text):
            if ord(char) in ZERO_WIDTH_CHARS:
                found.append((i, f'U+{ord(char):04X}'))
        
        return found[:10]
    
    def find_private_use_characters(self, text: str) -> List[Tuple[int, str]]:
        """Find characters in private use areas"""
        found = []
        
        for i, char in enumerate(text):
            code = ord(char)
            for start, end in PRIVATE_USE_AREAS:
                if start <= code <= end:
                    found.append((i, f'U+{code:04X}'))
                    break
        
        return found[:10]
    
    def find_unusual_unicode_blocks(self, text: str) -> Dict[str, int]:
        """Find characters from unusual Unicode blocks"""
        block_counts = Counter()
        
        for char in text:
            try:
                block = unicodedata.name(char).split()[0]
                # Skip common blocks
                if block not in ['LATIN', 'DIGIT', 'SPACE', 'PUNCTUATION']:
                    block_counts[block] += 1
            except ValueError:
                # Character has no name (could be private use, control, etc.)
                block_counts['UNNAMED'] += 1
        
        # Return blocks that appear unusually often
        return {block: count for block, count in block_counts.items() 
                if count > 5 or block == 'UNNAMED'}
    
    def find_suspicious_patterns(self, text: str) -> List[str]:
        """Find suspicious patterns that might indicate corruption"""
        patterns_found = []
        
        # Check for repeated special characters
        special_repeat = re.findall(r'([^\w\s])\1{4,}', text)
        if special_repeat:
            patterns_found.append(f'Repeated special chars: {set(special_repeat)}')
        
        # Check for unusual character sequences
        if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text):
            patterns_found.append('Contains raw control characters')
        
        # Check for mojibake patterns (common UTF-8 decode errors)
        mojibake_patterns = [
            r'Ã¢â‚¬â„¢',  # ' encoded incorrectly
            r'Ã¢â‚¬â€',   # " encoded incorrectly
            r'Ã‚Â',        # Common mojibake
            r'â€™',        # ' encoded incorrectly
            r'â€œ',        # " encoded incorrectly
        ]
        
        for pattern in mojibake_patterns:
            if pattern in text:
                patterns_found.append(f'Possible mojibake: {pattern}')
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_char_ratio > 0.3:
            patterns_found.append(f'High special character ratio: {special_char_ratio:.2%}')
        
        return patterns_found
    
    def generate_report(self):
        """Generate and print analysis report"""
        print("\n" + "="*60)
        print("CONVERSATION DATASET ANALYSIS REPORT")
        print("="*60)
        
        print("\n📊 STATISTICS:")
        print(f"  Total conversations: {self.stats['total_conversations']:,}")
        print(f"  Total messages: {self.stats['total_messages']:,}")
        print(f"  JSON parsing errors: {self.stats['json_errors']:,}")
        print(f"  Conversations with issues: {self.stats['conversations_with_issues']:,}")
        
        if self.stats['conversations_with_issues'] > 0:
            percentage = (self.stats['conversations_with_issues'] / self.stats['total_conversations']) * 100
            print(f"  Percentage with issues: {percentage:.2f}%")
        
        # Report JSON errors
        if self.issues['json_errors']:
            print(f"\n❌ JSON ERRORS ({len(self.issues['json_errors'])} found):")
            for i, error in enumerate(self.issues['json_errors'][:5]):
                print(f"  Line {error['line']}: {error['error']}")
                print(f"    Preview: {error['preview']}")
            if len(self.issues['json_errors']) > 5:
                print(f"  ... and {len(self.issues['json_errors']) - 5} more")
        
        # Report message issues
        if self.issues['message_issues']:
            print(f"\n⚠️  MESSAGE ISSUES ({len(self.issues['message_issues'])} found):")
            
            # Group by issue type
            issue_types = defaultdict(int)
            for msg_issue in self.issues['message_issues']:
                for issue in msg_issue['issues']:
                    issue_types[issue['type']] += 1
            
            print("\n  Issue type breakdown:")
            for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {issue_type}: {count}")
            
            # Show examples
            print("\n  Examples:")
            for i, msg_issue in enumerate(self.issues['message_issues'][:3]):
                print(f"\n  Example {i+1} (Line {msg_issue['line']}, Message {msg_issue['message_index']}):")
                print(f"    Role: {msg_issue['role']}")
                print(f"    Preview: {msg_issue['preview']}")
                print(f"    Issues:")
                for issue in msg_issue['issues']:
                    print(f"      - {issue['type']}: {issue}")
        
        # Report unusual characters
        print("\n🔤 UNUSUAL CHARACTERS:")
        unusual_chars = [(char, count) for char, count in self.char_frequency.most_common() 
                        if ord(char) > 127 or ord(char) < 32]
        
        if unusual_chars:
            print("  Most frequent non-ASCII or control characters:")
            for char, count in unusual_chars[:10]:
                try:
                    name = unicodedata.name(char)
                except ValueError:
                    name = "UNNAMED"
                print(f"    U+{ord(char):04X} ({repr(char)}): {count:,} occurrences - {name}")
        
        # Save detailed report
        report_path = self.file_path.parent / f"{self.file_path.stem}_analysis_report.txt"
        self.save_detailed_report(report_path)
        print(f"\n📄 Detailed report saved to: {report_path}")
    
    def save_detailed_report(self, output_path: Path):
        """Save detailed report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("DETAILED CONVERSATION DATASET ANALYSIS\n")
            f.write("="*60 + "\n\n")
            
            # Write all issues with full details
            if self.issues['message_issues']:
                f.write("MESSAGE ISSUES (Full List):\n")
                f.write("-"*40 + "\n")
                for issue in self.issues['message_issues']:
                    f.write(f"\nLine {issue['line']}, Message {issue['message_index']} ({issue['role']}):\n")
                    f.write(f"Content preview: {issue['preview']}\n")
                    f.write("Issues found:\n")
                    for iss in issue['issues']:
                        f.write(f"  - {iss}\n")
            
            # Write character frequency analysis
            f.write("\n\nCHARACTER FREQUENCY ANALYSIS:\n")
            f.write("-"*40 + "\n")
            for char, count in self.char_frequency.most_common(100):
                if ord(char) > 127 or ord(char) < 32:
                    try:
                        name = unicodedata.name(char)
                    except ValueError:
                        name = "UNNAMED"
                    f.write(f"U+{ord(char):04X} ({repr(char)}): {count:,} - {name}\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python detect_corrupted_chars.py <path_to_conversations.jsonl>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    analyzer = ConversationAnalyzer(file_path)
    analyzer.analyze()


if __name__ == "__main__":
    main() 