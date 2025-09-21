# Keyboard Maestro Medical Education Workflow

A comprehensive automation system for generating medical education content from screenshots and text. This toolkit uses OCR, GPT-4, and AI image generation to create educational materials for medical study.

## 🎯 Features

- **OCR Text Extraction**: Extract text from medical screenshots
- **GPT-Powered Analysis**: Generate High Yield medical information using GPT-4
- **AI Image Generation**: Create patient images using Pollinations.AI
- **Smart Image Search**: Find relevant medical diagrams from web sources
- **Keyboard Maestro Integration**: Seamless automation with macOS

## 📁 Project Structure

```
Keyboard Maestro/
├── Clipboard2Image/          # Clipboard to patient image generation
│   ├── gen_local.py         # Main clipboard processing script
│   └── gen_local.command    # Shell wrapper for Keyboard Maestro
├── Screen Setups + Image/    # Main workflow and image search
│   ├── enhanced_medical_workflow.py  # ⭐ Main comprehensive workflow
│   ├── final_ddg_image_api.py        # DuckDuckGo image search API
│   ├── fixed_functions.py           # Additional utility functions
│   ├── pollinations_mcp_server.py   # MCP server for Pollinations.AI
│   ├── requirements.txt             # Python dependencies
│   └── README_DDG_Images.md         # DuckDuckGo API documentation
├── generated_image.png      # AI-generated patient image
├── image1.png              # Educational diagram 1
├── image2.png              # Educational diagram 2  
├── image3.png              # High Yield summary image
└── high_yield_info.txt     # High Yield text content
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create Python virtual environment
python3 -m venv sd-venv
source sd-venv/bin/activate

# Install dependencies
pip install -r "Screen Setups + Image/requirements.txt"
```

### 2. Environment Variables

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export IMG_WIDTH=1024  # Optional: default image width
export IMG_HEIGHT=1024 # Optional: default image height
```

### 3. Usage Examples

#### A. Complete Medical Workflow (Recommended)
```bash
# Copy medical case text to clipboard, then run:
cd "Screen Setups + Image"
python3 enhanced_medical_workflow.py
```

This generates:
- `image1.png`: AI-generated medical visual
- `image2.png`: Real medical diagram from search
- `image3.png`: High Yield summary (colorful background)
- `high_yield_info.txt`: Text summary
- `generated_image.png`: AI patient image

#### B. Clipboard to Patient Image Only
```bash
cd Clipboard2Image
./gen_local.command
```

#### C. DuckDuckGo Image Search
```python
from final_ddg_image_api import find_and_download_image

# Search and download to image2.png
success = find_and_download_image("pneumonia chest X-ray")
```

## 🔧 Core Components

### 1. Enhanced Medical Workflow (`enhanced_medical_workflow.py`)
The main comprehensive script that:
- Extracts text from clipboard or Chrome selection
- Generates GPT-powered High Yield summaries
- Creates targeted search queries for medical diagrams
- Downloads real medical images
- Generates AI patient images
- Creates colorful summary images

### 2. Clipboard Image Generator (`gen_local.py`)
Focused on patient image generation:
- Reads medical case text from clipboard
- Uses GPT to create appropriate patient image prompts
- Generates realistic patient photos using Pollinations.AI
- Includes watermark removal for clean results

### 3. DuckDuckGo Image API (`final_ddg_image_api.py`)
Reliable image search implementation:
- Multiple fallback strategies for reliability
- Medical image filtering and validation
- Direct download to workspace files
- Compatible with Node.js API style

## 🎮 Keyboard Maestro Integration

### Setup Instructions:
1. Create a new Keyboard Maestro macro
2. Set trigger (e.g., hotkey or screenshot event)
3. Add "Execute Shell Script" action:
   ```bash
   cd "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/Screen Setups + Image"
   source "$HOME/sd-venv/bin/activate"
   python3 enhanced_medical_workflow.py
   ```

### Workflow:
1. Take screenshot of medical case → OCR extracts text
2. Macro triggers automatically
3. AI generates educational content
4. Results saved to workspace for immediate use

## 🔒 Security Notes

- **API Keys**: All API keys use environment variables (no hardcoded keys)
- **Local Processing**: Images and text processed locally when possible
- **Optional Cloud**: Only GPT and Pollinations.AI require internet access

## 📊 Output Files

| File | Description | Size | Purpose |
|------|-------------|------|---------|
| `image1.png` | AI medical visual | 1024×768 | Educational diagram (no text) |
| `image2.png` | Real medical image | 1024×768 | Search result from web |
| `image3.png` | High Yield summary | 1200×500 | Colorful summary for study |
| `generated_image.png` | AI patient image | 1024×1024 | Realistic patient photo |
| `high_yield_info.txt` | Text summary | - | Bottom line medical facts |

## 🛠️ Dependencies

Core Python packages:
- `openai>=1.0.0` - GPT-4 integration
- `requests` - HTTP requests for image downloads
- `Pillow>=10.0.0` - Image processing
- `duckduckgo-search>=6.2.6` - Real image search

System requirements:
- macOS (for `pbpaste`/`pbcopy` clipboard access)
- Python 3.8+
- Internet connection for AI services

## 🔄 Development Workflow

The project has been cleaned of:
- ✅ Duplicate/experimental DuckDuckGo implementations
- ✅ Debug and test scripts
- ✅ Hardcoded API keys (moved to environment variables)
- ✅ Python cache directories
- ✅ Unused backup files

Current state: **Production ready** with clean, maintainable code.

## 📚 Additional Resources

- `README_DDG_Images.md`: Detailed DuckDuckGo API documentation
- `requirements.txt`: Complete dependency list
- `pollinations_mcp_server.py`: MCP server integration for advanced workflows

## 🆘 Troubleshooting

### Common Issues:
1. **No images generated**: Check internet connection and API keys
2. **Empty clipboard**: Ensure text is copied before running
3. **Import errors**: Verify virtual environment activation and dependencies

### Debug Mode:
Set environment variable for verbose output:
```bash
export DEBUG=1
python3 enhanced_medical_workflow.py
```

---

**Note**: This system is designed for educational purposes in medical training. Ensure compliance with your institution's policies regarding AI-generated content.
