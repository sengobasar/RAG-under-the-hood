from src.loader import clean_text

test_cases = [
    "Sb sh rv rzh vz", # Common junk
    "Hindi text •◦■◆●►▶※ English fragments Sb", # Symbols and junk
    "Hello + world - this is | a test.", # Special characters
    "Many dots..... and underscores_____", # Repeated chars
    "स्वतंत्रताSb सेनानी sh", # Mixed junk and Hindi
]

print("Starting Verification of clean_text...")
for i, test in enumerate(test_cases):
    cleaned = clean_text(test)
    print(f"Test case {i+1}:")
    print(f"  Input:  {test}")
    print(f"  Output: {cleaned}")
    print("-" * 20)

print("Verification complete.")
