#!/usr/bin/env python3
"""
Count observation epochs in RINEX files to determine expected .pos records
"""

import sys
from pathlib import Path

def count_rinex_epochs(rinex_file):
    """Count observation epochs in a RINEX observation file"""
    epochs = 0
    
    with open(rinex_file, 'r', encoding='utf-8', errors='ignore') as f:
        in_header = True
        
        for line in f:
            # Check for end of header
            if 'END OF HEADER' in line:
                in_header = False
                continue
            
            # Skip header lines
            if in_header:
                continue
            
            # RINEX 3.x: epoch lines start with '>'
            # RINEX 2.x: epoch lines have format: YY MM DD HH MM SS.SSSSSSS
            if line.startswith('>'):
                epochs += 1
            elif len(line) >= 26:
                # Try to parse as RINEX 2.x epoch line
                try:
                    # Check if first chars look like date/time
                    year = line[1:3].strip()
                    month = line[4:6].strip()
                    if year.isdigit() and month.isdigit():
                        epochs += 1
                except:
                    pass
    
    return epochs

def analyze_test_data(test_dir):
    """Analyze all test data RINEX files"""
    test_dir = Path(test_dir)
    
    # Find all rover RINEX files recursively
    rover_patterns = [
        "**/gnss*.20o",
        "**/gnss*.21o", 
        "**/gnss*.22o",
        "**/*rinex*.20o",
        "**/*rinex*.21o",
        "**/*rinex*.22o"
    ]
    
    all_rover_files = []
    for pattern in rover_patterns:
        files = list(test_dir.glob(pattern))
        # Exclude base station files
        files = [f for f in files if 'sutb' not in f.name.lower() and 'base' not in f.name.lower()]
        all_rover_files.extend(files)
    
    # Remove duplicates
    all_rover_files = list(set(all_rover_files))
    all_rover_files = sorted(all_rover_files)
    
    if not all_rover_files:
        print(f"No rover files found in {test_dir}")
        print(f"Searched for patterns: {rover_patterns}")
        return
    
    print(f"Found {len(all_rover_files)} rover observation files\n")
    print(f"{'File Path':<80} {'Epochs':>10}")
    print("=" * 92)
    
    total_epochs = 0
    
    for rover_file in all_rover_files:
        epochs = count_rinex_epochs(rover_file)
        
        # Show relative path from test_dir
        try:
            rel_path = rover_file.relative_to(test_dir)
        except:
            rel_path = rover_file
        
        print(f"{str(rel_path):<80} {epochs:>10,}")
        
        total_epochs += epochs
    
    print("=" * 92)
    print(f"{'TOTAL':<80} {total_epochs:>10,}")
    
    print(f"\n📊 Summary:")
    print(f"  Rover files found:    {len(all_rover_files)}")
    print(f"  Expected epochs:      {total_epochs:,}")
    print(f"\n✓ Your .pos files should have {total_epochs:,} total records")
    
    return total_epochs

def main():
    if len(sys.argv) < 2:
        print("Usage: python count_rinex_epochs.py <test_data_directory>")
        print("\nExample:")
        print("  python count_rinex_epochs.py /Users/avantika/Documents/ppkcode/ppk_output_v3")
        print("  python count_rinex_epochs.py C:\\Users\\gauth\\Documents\\SC4000_proj\\test_data")
        sys.exit(1)
    
    test_dir = sys.argv[1]
    analyze_test_data(test_dir)

if __name__ == "__main__":
    main()

#C:\Users\gauth\Documents\SC4000_proj\test