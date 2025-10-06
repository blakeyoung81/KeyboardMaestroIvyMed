#!/usr/bin/env python3
"""
Pollinations Medical Image Generator MCP Server
Provides tools for OCR text extraction and patient image generation
"""

import asyncio
import json
import sys
import os
import subprocess
import requests
import openai
from urllib.parse import quote
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import re
import textwrap
from PIL import Image, ImageDraw, ImageFont

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    pass

# Add the mcp package to path
sys.path.append(os.path.expanduser("~/.local/lib/python3.11/site-packages"))

try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types
except ImportError:
    print("MCP library not found. Installing...", file=sys.stderr)
    subprocess.run([sys.executable, "-m", "pip", "install", "mcp"])
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types

# Configuration
OUT_PATH = Path("/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/generated_image.png")
OUT1 = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/image1.png"
OUT2 = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/image2.png"
OUT3 = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/image3.png"
HIGH_YIELD_OUT = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/high_yield_info.txt"

# Backup directories
BASE_DIR = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro"
AI_PATIENTS_DIR = os.path.join(BASE_DIR, "AI Generated Patients")
AI_IMAGES_DIR = os.path.join(BASE_DIR, "AI Generated Images")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Set this environment variable
GPT_MODEL = "gpt-4o-mini"

def get_next_backup_number(backup_dir: str, prefix: str = "") -> int:
    """Find the next available backup number in the directory"""
    try:
        os.makedirs(backup_dir, exist_ok=True)
        existing_files = os.listdir(backup_dir)
        
        # Find files matching the pattern (prefix + number + extension)
        numbers = []
        for filename in existing_files:
            if filename.startswith(prefix):
                # Extract number from filename like "patient_001.jpg" or "image_042.png"
                try:
                    # Remove prefix and extension to get the number part
                    name_without_prefix = filename[len(prefix):]
                    number_str = name_without_prefix.split('.')[0].lstrip('_')
                    if number_str.isdigit():
                        numbers.append(int(number_str))
                except:
                    continue
        
        return max(numbers) + 1 if numbers else 1
    except Exception as e:
        print(f"⚠️ Error finding next backup number: {e}")
        return 1

def save_backup_copy(source_file: str, backup_dir: str, prefix: str, description: str = ""):
    """Save a backup copy of the generated image with incremental numbering"""
    try:
        if not os.path.exists(source_file):
            print(f"⚠️ Source file does not exist: {source_file}")
            return
        
        # Ensure backup directory exists
        os.makedirs(backup_dir, exist_ok=True)
        
        # Get next available number
        backup_num = get_next_backup_number(backup_dir, prefix)
        
        # Determine file extension
        _, ext = os.path.splitext(source_file)
        if not ext:
            ext = '.png'  # Default extension
        
        # Create backup filename
        backup_filename = f"{prefix}_{backup_num:03d}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy the file
        import shutil
        shutil.copy2(source_file, backup_path)
        
        print(f"💾 Backup saved: {backup_filename} {description}")
        
    except Exception as e:
        print(f"⚠️ Error saving backup: {e}")

# Create the server
server = Server("pollinations-medical-images")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="extract_ocr_text",
            description="Extract text from clipboard image using OCR",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["clipboard", "chrome"],
                        "description": "Source of text: clipboard (from OCR) or chrome (highlighted text)",
                        "default": "clipboard"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="generate_patient_image",
            description="Generate a patient image from medical case text using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "medical_text": {
                        "type": "string",
                        "description": "Medical case text to generate patient image from"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width (default: 888, standardized)",
                        "default": 888
                    },
                    "height": {
                        "type": "integer", 
                        "description": "Image height (default: 664, standardized)",
                        "default": 664
                    },
                    "use_clipboard": {
                        "type": "boolean",
                        "description": "If true, use text from clipboard instead of medical_text parameter",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="generate_image_prompt",
            description="Generate an image prompt from medical case text using GPT",
            inputSchema={
                "type": "object",
                "properties": {
                    "medical_text": {
                        "type": "string",
                        "description": "Medical case text to analyze"
                    }
                },
                "required": ["medical_text"]
            }
        ),
        types.Tool(
            name="create_image_from_prompt",
            description="Generate an image directly from a prompt using Pollinations.AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Image generation prompt"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width (default: 888, standardized)",
                        "default": 888
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height (default: 1024)", 
                        "default": 1024
                    }
                },
                "required": ["prompt"]
            }
        ),
        types.Tool(
            name="generate_high_yield_info",
            description="Generate High Yield medical information from case text using GPT",
            inputSchema={
                "type": "object",
                "properties": {
                    "medical_text": {
                        "type": "string",
                        "description": "Medical case text to analyze for high yield points"
                    },
                    "use_clipboard": {
                        "type": "boolean",
                        "description": "If true, use text from clipboard instead of medical_text parameter",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="google_images_search",
            description="Search Google Images for educational medical images and download first 2 results",
            inputSchema={
                "type": "object",
                "properties": {
                    "medical_text": {
                        "type": "string",
                        "description": "Medical case text to create search queries from"
                    },
                    "use_clipboard": {
                        "type": "boolean",
                        "description": "If true, use text from clipboard instead of medical_text parameter",
                        "default": False
                    }
                },
                "required": []
            }
        )
    ]

def get_clipboard_text() -> str:
    """Get text from clipboard."""
    try:
        result = subprocess.run(['pbpaste'], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error getting clipboard text: {e}"

def get_chrome_selection() -> str:
    """Get highlighted text from Chrome."""
    try:
        # Save current clipboard
        current_clipboard = subprocess.run(['pbpaste'], capture_output=True, text=True).stdout
        
        # Activate Chrome and copy selection
        applescript = '''
        tell application "Google Chrome" to activate
        delay 0.2
        tell application "System Events"
            key code 8 using {command down} -- Cmd+C
        end tell
        delay 0.1
        '''
        
        subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True)
        
        # Get what was copied
        new_clipboard = subprocess.run(['pbpaste'], capture_output=True, text=True).stdout.strip()
        
        # Check if clipboard changed
        if new_clipboard != current_clipboard.strip() and new_clipboard:
            # Restore original clipboard
            subprocess.run(['pbcopy'], input=current_clipboard, text=True)
            return new_clipboard
        
        # No selection, restore clipboard
        subprocess.run(['pbcopy'], input=current_clipboard, text=True)
        return ""
        
    except Exception as e:
        return f"Error getting Chrome selection: {e}"

def generate_patient_prompt(medical_text: str) -> str:
    """Generate patient image prompt using GPT."""
    if not OPENAI_API_KEY:
        return "A professional headshot of a patient in a medical setting, neutral expression, good lighting"
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        system_prompt = """You are an expert at creating image generation prompts for medical education. 
        Given a medical case or question stem, generate a detailed but appropriate prompt for creating 
        a professional patient photograph that would accompany this case.
        
        Focus on:
        - Age, gender, and general appearance that fits the case
        - Professional medical photography style
        - Appropriate clothing/setting
        - Facial expression that might reflect the condition (but keep it professional)
        - Good lighting and composition
        
        Keep it under 200 characters and avoid any graphic or inappropriate content.
        Return ONLY the image prompt, nothing else."""
        
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Medical case text: {medical_text}"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating prompt: {e}"

def generate_high_yield_info(medical_text: str) -> str:
    """Generate High Yield takeaway using GPT - concise 2-sentence summary"""
    if not OPENAI_API_KEY:
        return "❌ No OpenAI API key found for High Yield generation"
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        system_prompt = """You are a medical education expert. Create a VERY TERSE "Bottom Line" summary.
        
        Provide EXACTLY 1 short sentence (15-25 words max) that captures the most critical exam point.
        
        Focus on:
        - The key diagnosis or main concept
        - The single most important fact to remember
        
        Format: One short, punchy sentence. No extra words.
        
        Examples:
        "Low sensitivity of FOBT makes colonoscopy preferred for colon cancer screening in high-risk patients."
        "MI diagnosis requires 2 of 3: chest pain, ECG changes, elevated cardiac enzymes."
        
        Make it ultra-concise and exam-focused."""
        
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a terse Bottom Line from this medical case:\n\n{medical_text}"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"❌ Error generating High Yield info: {e}"

def generate_enhanced_search_queries(medical_text: str) -> list:
    """Generate GPT-powered Google Image search queries for relevant diagrams"""
    if not OPENAI_API_KEY:
        # Fallback queries
        summary = re.sub(r"\s+", " ", medical_text)[:100]
        return [
            f"{summary} mechanism diagram",
            f"{summary} anatomy pathology"
        ]
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        system_prompt = """You are a medical education expert creating Google Images search queries to find the MOST RELEVANT diagrams for this medical case.

Analyze the medical case and identify:
1. The primary medical condition/diagnosis
2. Key anatomical structures involved  
3. Important pathophysiological mechanisms
4. Relevant clinical findings

Create 2 SPECIFIC Google Images search queries that will find educational diagrams directly related to this case:

Query 1: Focus on the primary mechanism, pathophysiology, or disease process
Query 2: Focus on relevant anatomy, clinical findings, or diagnostic images

Requirements:
- Use precise medical terminology
- 3-6 words per query
- Focus on diagram/illustration content
- Avoid copying the entire case text
- Target educational/textbook-style images

Example:
Case about myocardial infarction → "coronary artery occlusion diagram" and "ECG ST elevation changes"

Return ONLY the two search queries, one per line, nothing else."""
        
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this medical case and create 2 targeted Google Images search queries for relevant diagrams:\n\n{medical_text}"}
            ],
            max_tokens=80,
            temperature=0.1
        )
        
        queries_text = response.choices[0].message.content.strip()
        queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
        
        # Clean up queries (remove quotes, extra formatting)
        cleaned_queries = []
        for query in queries:
            # Remove quotes and clean up
            clean_query = query.replace('"', '').replace("'", '').strip()
            if clean_query and len(clean_query) > 3:
                cleaned_queries.append(clean_query)
        
        # Ensure we have exactly 2 queries
        if len(cleaned_queries) < 2:
            summary = re.sub(r"\s+", " ", medical_text)[:60]
            fallback_queries = [f"{summary} mechanism diagram", f"{summary} anatomy pathology"]
            cleaned_queries.extend(fallback_queries)
        
        final_queries = cleaned_queries[:2]  # Take only first 2
        print(f"🎯 GPT-generated search queries: {final_queries}")
        return final_queries
        
    except Exception as e:
        print(f"⚠️ GPT query generation failed: {e}")
        summary = re.sub(r"\s+", " ", medical_text)[:60]
        return [f"{summary} mechanism diagram", f"{summary} anatomy pathology"]

def google_images_enhanced_search(query: str) -> str:
    """Enhanced Google Images search with better parsing"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    # Try multiple search strategies
    search_urls = [
        f"https://www.google.com/search?q={quote(query)}&tbm=isch&tbs=isz:m",  # Medium size images
        f"https://www.bing.com/images/search?q={quote(query)}&form=HDRSC2&first=1",  # Bing fallback
    ]
    
    for i, url in enumerate(search_urls):
        try:
            print(f"🔍 Searching: {query} (method {i+1})")
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            html = response.text
            
            # Google Images parsing
            if "google.com" in url:
                # Look for image URLs in Google's JSON data
                pattern = r'"ou":"(https?://[^"]+\.(?:jpg|jpeg|png|gif|webp))"'
                matches = re.findall(pattern, html)
                if matches:
                    # Filter out small images and return first good one
                    for match in matches:
                        if any(size in match for size in ['large', 'medium', '1024', '800']):
                            return match
                    return matches[0] if matches else ""
            
            # Bing Images parsing
            elif "bing.com" in url:
                pattern = r'"murl":"(https?://[^"]+\.(?:jpg|jpeg|png|gif|webp))"'
                matches = re.findall(pattern, html)
                if matches:
                    return matches[0]
                    
        except Exception as e:
            print(f"⚠️ Search method {i+1} failed: {e}")
            continue
    
    return ""

def download_image(url: str, out_path: str) -> bool:
    """Download image with better error handling"""
    if not url:
        return False
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
        "Referer": "https://www.google.com/",
    }
    
    try:
        print(f"⬇️  Downloading: {url[:80]}...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if it's actually an image
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'webp']):
            print(f"⚠️ Not an image: {content_type}")
            return False
        
        with open(out_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved → {out_path}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def create_high_yield_image(high_yield_text: str, output_path: str, width: int = 1200, height: int = 500):
    """Create yellow background image displaying the literal GPT text response"""
    
    # Create image with yellow background
    img = Image.new('RGB', (width, height), color='#FFFF00')  # Bright yellow
    draw = ImageDraw.Draw(img)
    
    # Try to load system fonts, fall back to default
    try:
        # Try various font paths on macOS
        font_paths = [
            "/System/Library/Fonts/SF-Pro-Display-Bold.otf",
            "/System/Library/Fonts/SF-Pro-Display-Regular.otf", 
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Times.ttc"
        ]
        
        title_font = None
        body_font = None
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    title_font = ImageFont.truetype(font_path, 32)  # Title font
                    body_font = ImageFont.truetype(font_path, 22)   # Body text
                    break
                except:
                    continue
        
        # If no system fonts found, use default
        if not title_font:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Black text on yellow background
    text_color = '#000000'  # Black
    
    # Layout settings
    margin = 50
    y_pos = 60
    line_height = 30
    
    # Title: "Bottom Line:"
    title_text = "Bottom Line:"
    
    # Get text bounding box for centering
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    
    # Draw title
    draw.text((title_x, y_pos), title_text, fill=text_color, font=title_font)
    y_pos += 80  # Space after title
    
    # Display the EXACT GPT response text (no modification)
    gpt_text = high_yield_text.strip()
    
    # Wrap the text to fit in the image width
    wrapper = textwrap.TextWrapper(
        width=85,  # Characters per line
        break_long_words=False, 
        break_on_hyphens=False,
        expand_tabs=False
    )
    wrapped_lines = wrapper.wrap(gpt_text)
    
    # Draw each line of the GPT response, centered
    for line in wrapped_lines:
        # Get text bounding box for centering this line
        line_bbox = draw.textbbox((0, 0), line, font=body_font)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        
        # Draw the line
        draw.text((line_x, y_pos), line, fill=text_color, font=body_font)
        y_pos += line_height
        
        # If we're running out of space, truncate gracefully
        if y_pos > height - 80:
            # Add ellipsis if text is cut off
            ellipsis_bbox = draw.textbbox((0, 0), "...", font=body_font)
            ellipsis_width = ellipsis_bbox[2] - ellipsis_bbox[0]
            ellipsis_x = (width - ellipsis_width) // 2
            draw.text((ellipsis_x, y_pos), "...", fill=text_color, font=body_font)
            break
    
    # No footer needed anymore
    
    # Save the image
    img.save(output_path, 'PNG', quality=95)
    print(f"✅ Bottom Line summary image saved → {output_path}")
    
    return output_path

def create_pollinations_image(prompt: str, width: int = 888, height: int = 664) -> str:
    """Generate image using Pollinations.AI."""
    try:
        # Ensure destination folder exists
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # URL encode the prompt
        encoded_prompt = quote(prompt)
        
        # Build Pollinations.AI URL
        pollinations_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        
        # Add size parameters if not default
        params = []
        # Always include dimensions to ensure consistency (standardized: 888x664)
        params.append(f"width={width}")
        params.append(f"height={height}")
            
        if params:
            pollinations_url += "?" + "&".join(params)
        
        # Make request
        response = requests.get(pollinations_url, timeout=30)
        response.raise_for_status()
        
        # Save image
        with open(OUT_PATH, 'wb') as f:
            f.write(response.content)
        
        # Save backup copy to AI Generated Patients directory
        save_backup_copy(str(OUT_PATH), AI_PATIENTS_DIR, "patient", "(MCP Server)")
        
        return f"✅ Image saved to {OUT_PATH}"
        
    except Exception as e:
        return f"❌ Error generating image: {e}"

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls."""
    
    if name == "extract_ocr_text":
        source = arguments.get("source", "clipboard")
        
        if source == "chrome":
            text = get_chrome_selection()
            if text:
                return [types.TextContent(type="text", text=f"✅ Chrome selection: {text}")]
            else:
                return [types.TextContent(type="text", text="❌ No text selected in Chrome")]
        else:
            text = get_clipboard_text()
            if text:
                return [types.TextContent(type="text", text=f"✅ Clipboard text: {text}")]
            else:
                return [types.TextContent(type="text", text="❌ No text in clipboard")]
    
    elif name == "generate_image_prompt":
        medical_text = arguments.get("medical_text", "")
        if not medical_text:
            return [types.TextContent(type="text", text="❌ No medical text provided")]
        
        prompt = generate_patient_prompt(medical_text)
        return [types.TextContent(type="text", text=f"Generated prompt: {prompt}")]
    
    elif name == "create_image_from_prompt":
        prompt = arguments.get("prompt", "")
        width = arguments.get("width", 888)
        height = arguments.get("height", 664)
        
        if not prompt:
            return [types.TextContent(type="text", text="❌ No prompt provided")]
        
        result = create_pollinations_image(prompt, width, height)
        return [types.TextContent(type="text", text=result)]
    
    elif name == "generate_patient_image":
        width = arguments.get("width", 888)
        height = arguments.get("height", 664)
        use_clipboard = arguments.get("use_clipboard", False)
        
        # Get medical text
        if use_clipboard:
            medical_text = get_clipboard_text()
            if not medical_text:
                # Try Chrome selection as fallback
                medical_text = get_chrome_selection()
                if not medical_text:
                    return [types.TextContent(type="text", text="❌ No text found in clipboard or Chrome selection")]
        else:
            medical_text = arguments.get("medical_text", "")
            if not medical_text:
                return [types.TextContent(type="text", text="❌ No medical text provided")]
        
        # Generate prompt
        patient_prompt = generate_patient_prompt(medical_text)
        
        # Generate image
        result = create_pollinations_image(patient_prompt, width, height)
        
        return [types.TextContent(type="text", text=f"Medical text: {medical_text[:100]}...\nPrompt: {patient_prompt}\nResult: {result}")]
    
    elif name == "generate_high_yield_info":
        use_clipboard = arguments.get("use_clipboard", False)
        
        # Get medical text
        if use_clipboard:
            medical_text = get_clipboard_text()
            if not medical_text:
                # Try Chrome selection as fallback
                medical_text = get_chrome_selection()
                if not medical_text:
                    return [types.TextContent(type="text", text="❌ No text found in clipboard or Chrome selection")]
        else:
            medical_text = arguments.get("medical_text", "")
            if not medical_text:
                return [types.TextContent(type="text", text="❌ No medical text provided")]
        
        # Generate High Yield information
        high_yield_content = generate_high_yield_info(medical_text)
        
        # Save to file
        try:
            with open(HIGH_YIELD_OUT, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("HIGH YIELD MEDICAL INFORMATION\n")
                f.write("=" * 60 + "\n\n")
                f.write("📋 SOURCE CASE:\n")
                f.write("-" * 30 + "\n")
                f.write(medical_text[:500] + ("..." if len(medical_text) > 500 else "") + "\n\n")
                f.write("🎯 HIGH YIELD CONTENT:\n")
                f.write("-" * 30 + "\n")
                f.write(high_yield_content + "\n\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Create image version
            create_high_yield_image(high_yield_content, OUT3)
            
            return [types.TextContent(type="text", text=f"✅ High Yield info generated and saved:\n• Text: {HIGH_YIELD_OUT}\n• Image: {OUT3}\n\n{high_yield_content}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"❌ Error saving High Yield info: {e}\n\n{high_yield_content}")]
    
    elif name == "google_images_search":
        use_clipboard = arguments.get("use_clipboard", False)
        
        # Get medical text
        if use_clipboard:
            medical_text = get_clipboard_text()
            if not medical_text:
                # Try Chrome selection as fallback
                medical_text = get_chrome_selection()
                if not medical_text:
                    return [types.TextContent(type="text", text="❌ No text found in clipboard or Chrome selection")]
        else:
            medical_text = arguments.get("medical_text", "")
            if not medical_text:
                return [types.TextContent(type="text", text="❌ No medical text provided")]
        
        # Generate search queries and download images
        queries = generate_enhanced_search_queries(medical_text)
        results = []
        
        for i, query in enumerate(queries[:2]):  # Only first 2
            url = google_images_enhanced_search(query)
            if url:
                out_path = OUT1 if i == 0 else OUT2
                if download_image(url, out_path):
                    results.append(f"✅ Image {i+1}: {out_path}")
                else:
                    results.append(f"❌ Failed to download image {i+1}")
            else:
                results.append(f"❌ No image found for query: {query}")
        
        return [types.TextContent(type="text", text=f"🔍 Google Images Search Results:\n" + "\n".join(results))]
    
    else:
        return [types.TextContent(type="text", text=f"❌ Unknown tool: {name}")]

async def main():
    # Run the server using stdio
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pollinations-medical-images",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
