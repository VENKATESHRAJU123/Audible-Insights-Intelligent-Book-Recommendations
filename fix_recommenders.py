"""
Quick Fix: Patch recommenders.py to handle missing columns
"""

import re

# Read the file
with open('src/recommenders.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to find the problematic return statement
old_pattern = r"return recommendations\[\['book_name', 'author', 'rating', 'genre', 'similarity_score'\]\]"

new_code = """# Select only columns that exist
        available_cols = ['book_name', 'similarity_score']
        for col in ['author', 'rating', 'genre', 'price']:
            if col in recommendations.columns:
                available_cols.append(col)
        
        return recommendations[available_cols]"""

# Replace
content = re.sub(old_pattern, new_code, content)

# Also fix other similar patterns
old_pattern2 = r"return recommendations\[\['book_name', 'author', 'rating', 'genre', 'cluster'\]\]"
new_code2 = """# Select only columns that exist
        available_cols = ['book_name', 'cluster']
        for col in ['author', 'rating', 'genre']:
            if col in recommendations.columns:
                available_cols.append(col)
        
        return recommendations[available_cols]"""

content = re.sub(old_pattern2, new_code2, content)

# Write back
with open('src/recommenders.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed recommenders.py!")
print("Now run: python scripts/train_models.py")
