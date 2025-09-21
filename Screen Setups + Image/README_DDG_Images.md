# DuckDuckGo Images API for Python

A Python implementation inspired by the Node.js package `duckduckgo-images-api` that you referenced.

## ✅ What's Working

The **`final_ddg_image_api.py`** is your main working solution that:

- 🎯 **Downloads images directly to `image2.png`** as requested
- 🔄 **Matches the Node.js API style** with the same configuration options
- 🛡️ **Multiple fallback methods** for reliability
- 📸 **Real image search** using the `ddgs` package (proven to work!)

## 🚀 Quick Usage

```python
from final_ddg_image_api import image_search, find_and_download_image

# Simple search (like Node.js API)
results = image_search({'query': 'birds', 'moderate': True})

# Direct download to image2.png
success = find_and_download_image("medical stethoscope")
```

## 📁 Files Created

1. **`final_ddg_image_api.py`** - Main working implementation ⭐
2. **`ddg_image_example.py`** - Usage examples
3. **`duckduckgo_images_api.py`** - Comprehensive version (backup)
4. **`reliable_ddg_search.py`** - Using ddgs package (backup)
5. **`simple_ddg_image_search.py`** - Simplified approach (backup)

## 🔧 Configuration Options

Just like the Node.js package:

```python
config = {
    'query': "search term",      # Required
    'moderate': False,           # Optional: content filtering
    'iterations': 2,             # Optional: result sets to fetch
    'retries': 2                 # Optional: retries per iteration
}
```

## ✅ Test Results

- ✅ Successfully finds real images from DuckDuckGo
- ✅ Downloads images to `image2.png` as requested
- ✅ Handles medical searches, bird searches, nature searches
- ✅ Provides fallback methods if main search fails
- ✅ Generator support for large iterations

## 🎉 Ready to Use!

Your `final_ddg_image_api.py` is production-ready and working exactly as you requested!
