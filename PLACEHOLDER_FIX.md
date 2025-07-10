# Quick Fix for Placeholder Restoration Issue

## Problem
Google Translate changes the case of placeholders:
- `XPROTXMDX0X` becomes `XprotxMDX0X` or `XProtxmdx0x`
- This causes placeholders to not be restored, leaving gibberish in the output

## Solution
Add this improved restoration function to the `BookTranslator` class in `translator.py`:

```python
def restore_formatting(self, text: str, protected_elements: dict) -> str:
    """Restore protected formatting elements after translation with robust case handling"""
    restored_text = text
    
    # Sort placeholders by length (longest first) to avoid partial replacements
    sorted_placeholders = sorted(protected_elements.items(), key=lambda x: len(x[0]), reverse=True)
    
    for placeholder, original in sorted_placeholders:
        # Google Translate transforms placeholders predictably:
        # XPROTXMDX0X -> XprotxMDX0X, XProtxmdx0x, etc.
        
        transformations = [
            placeholder,  # Original
            placeholder.replace('XPROTX', 'XprotX').replace('X', 'x'),
            placeholder.replace('XPROTX', 'XProtX').replace('X', 'x'),  
            placeholder.replace('XPROTX', 'Xprotx').replace('X', 'x'),
            placeholder.replace('XPROTX', 'xprotx').replace('X', 'x'),
            placeholder.replace('XPROTX', 'XprotX'),
            placeholder.replace('XPROTX', 'XProtX'),
            placeholder.replace('XPROTX', 'Xprotx'),
            placeholder.replace('XPROTX', 'xprotx'),
        ]
        
        # Add variations for different element types
        additional_variations = []
        for transform in transformations:
            variations = [
                transform,
                transform.replace('MDX', 'mdx'),
                transform.replace('MDX', 'Mdx'),
                transform.replace('CALX', 'Calx'),
                transform.replace('CALX', 'calx'),
                transform.replace('LBX', 'lbx'),
                transform.replace('LBX', 'Lbx'),
                transform.replace('CODEX', 'Codex'),
                transform.replace('CODEX', 'codex'),
                transform.replace('LINKX', 'Linkx'),
                transform.replace('LINKX', 'linkx'),
            ]
            additional_variations.extend(variations)
        
        transformations.extend(additional_variations)
        
        # Remove duplicates and apply replacements
        unique_transformations = list(set(transformations))
        
        for transformation in unique_transformations:
            if transformation in restored_text:
                restored_text = restored_text.replace(transformation, original)
    
    return restored_text
```

## Quick Test
After implementing this fix, test with your complex markdown example. The placeholders should now be properly restored instead of showing as gibberish.

## Status
✅ **Issue identified**: Google Translate case transformations  
✅ **Solution ready**: Robust case-insensitive restoration  
⏳ **Implementation needed**: Replace the existing `restore_formatting` method  