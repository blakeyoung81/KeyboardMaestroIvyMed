import os, sys, re, subprocess, time, pathlib
from urllib.parse import quote
import requests
import openai
import json
from PIL import Image, ImageDraw, ImageFont
import textwrap
import hashlib
import random

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    pass

# Try to import ddgs, gracefully handle if not available
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠️ ddgs package not available - install with: pip install ddgs")

# Output paths
OUT1 = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/image1.png"
OUT2 = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/image2.png"
OUT2_V2 = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/image2 v2.png"
OUT2_V3 = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/image2 v3.png"
OUT3 = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/image3.png"
HIGH_YIELD_OUT = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/high_yield_info.txt"
PATIENT_OUT = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/generated_image.png"

# Backup directories
BASE_DIR = "/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro"
AI_PATIENTS_DIR = os.path.join(BASE_DIR, "AI Generated Patients")
AI_IMAGES_DIR = os.path.join(BASE_DIR, "AI Generated Images")

# OpenAI Configuration
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

def get_clipboard_text() -> str:
    """Get text from clipboard"""
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Could not read clipboard: {e}")
        return ""

def get_chrome_selection() -> str:
    """Try to copy highlighted text from Chrome without clobbering clipboard."""
    try:
        current = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
        applescript = '''
        tell application "Google Chrome" to activate
        delay 0.2
        tell application "System Events"
            key code 8 using {command down}
        end tell
        delay 0.1
        '''
        subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
        new_val = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout.strip()
        subprocess.run(["pbcopy"], input=current, text=True)
        if new_val and new_val.strip() != current.strip():
            return new_val
        return ""
    except Exception as e:
        print(f"⚠️ Chrome selection failed: {e}")
        return ""

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
        
        high_yield_content = response.choices[0].message.content.strip()
        print("✅ Generated High Yield Takeaway")
        return high_yield_content
        
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

def calculate_image_hash(image_path: str) -> str:
    """Calculate perceptual hash of an image for similarity comparison"""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize for consistent comparison
            img = img.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
            # Get pixel data and create hash
            pixels = list(img.getdata())
            # Simple hash based on pixel values
            hash_string = ''.join([str(1 if p > 128 else 0) for p in pixels])
            return hashlib.md5(hash_string.encode()).hexdigest()
    except Exception as e:
        print(f"⚠️ Could not calculate hash for {image_path}: {e}")
        return ""

def images_are_similar(image1_path: str, image2_path: str, threshold: float = 0.85) -> bool:
    """Check if two images are too similar using hash comparison"""
    try:
        if not os.path.exists(image1_path) or not os.path.exists(image2_path):
            return False
            
        hash1 = calculate_image_hash(image1_path)
        hash2 = calculate_image_hash(image2_path)
        
        if not hash1 or not hash2:
            return False
        
        # Calculate Hamming distance
        if len(hash1) != len(hash2):
            return False
            
        same_chars = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        similarity = same_chars / len(hash1)
        
        print(f"🔍 Image similarity: {similarity:.2f} (threshold: {threshold})")
        return similarity > threshold
        
    except Exception as e:
        print(f"⚠️ Error comparing images: {e}")
        return False

def generate_varied_visual_prompt(query: str, variation_type: int = 1) -> str:
    """Generate varied prompts for realistic medical imagery without text or labels"""
    
    # Different styles focused on visual representation, NO TEXT
    variations = [
        # Variation 1: Realistic medical photography style
        f"A high-quality realistic medical photograph showing {query}, professional medical photography, detailed anatomical view, natural lighting, clinical setting, no text, no labels, no words",
        
        # Variation 2: 3D realistic anatomical view
        f"A photorealistic 3D rendered image of {query}, detailed anatomical accuracy, medical visualization, natural colors, professional medical imaging, no text overlays, no labels, no words",
        
        # Variation 3: Clinical imaging style
        f"A realistic clinical image depicting {query}, medical imaging style, clear anatomical detail, professional medical photograph, hospital setting, no text, no labels, no words",
        
        # Variation 4: Educational visual representation
        f"A clear realistic visual representation of {query}, educational medical imagery, natural anatomical colors, detailed view, professional medical photography, no text, no labels, no words",
        
        # Variation 5: Detailed anatomical view
        f"A detailed realistic image showing {query}, anatomical accuracy, medical reference quality, natural lighting, clear visual detail, no text overlays, no labels, no words"
    ]
    
    # Select variation based on type (1-5)
    variation_index = (variation_type - 1) % len(variations)
    return variations[variation_index]

def generate_medical_visual_with_ai(query: str, out_path: str, variation_type: int = 1) -> bool:
    """Generate realistic medical imagery using AI (no text/labels, visual depiction only)"""
    try:
        print(f"🎨 Generating medical visual for: {query} (variation {variation_type})")
        
        # Create varied prompt focused on visual representation without text
        visual_prompt = generate_varied_visual_prompt(query, variation_type)
        
        # Add additional emphasis on no text/labels
        enhanced_prompt = f"{visual_prompt}, photorealistic, high quality, medical photography, absolutely no text, no labels, no words, no writing, visual only"
        
        # Generate random seed for variation
        seed = random.randint(1, 999999)
        
        # Use Pollinations.AI to generate the visual
        # STANDARDIZED DIMENSIONS: 888x664 (consistent across all AI images)
        encoded_prompt = quote(enhanced_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=888&height=664&seed={seed}&enhance=true&nologo=true"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        print(f"🌐 Generating visual imagery with AI (seed: {seed})...")
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        if response.headers.get('content-type', '').startswith('image/'):
            # Save the generated visual
            with open(out_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Medical visual saved → {out_path}")
            
            # Save backup copy to AI Generated Images directory
            save_backup_copy(out_path, AI_IMAGES_DIR, "medical_visual", f"({query})")
            
            return True
        else:
            print(f"❌ Invalid response type: {response.headers.get('content-type')}")
            return False
            
    except Exception as e:
        print(f"❌ Visual generation failed: {e}")
        return False

def generate_medical_diagram_for_search_query(query: str, out_path: str) -> bool:
    """Generate a medical diagram using AI but with search-result style prompting"""
    try:
        print(f"🔍 Creating search-result style medical image for: {query}")
        
        # Create prompts that mimic real medical textbook/educational images
        search_style_prompt = f"A high-quality medical textbook illustration of {query}, professional medical diagram style, clear educational image as it would appear in search results, anatomically accurate, clean white background"
        
        # Use different seed for variety
        seed = random.randint(100000, 999999)
        
        # Use Pollinations.AI to generate the "search result" style image
        # STANDARDIZED DIMENSIONS: 888x664 (consistent across all AI images)
        encoded_prompt = quote(search_style_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=888&height=664&seed={seed}&enhance=true&nologo=true"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        print(f"🌐 Generating search-style medical image (seed: {seed})...")
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        if response.headers.get('content-type', '').startswith('image/'):
            # Save the generated image
            with open(out_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Search-style medical image saved → {out_path}")
            return True
        else:
            print(f"❌ Invalid response type: {response.headers.get('content-type')}")
            return False
            
    except Exception as e:
        print(f"❌ Search-style image generation failed: {e}")
        return False

def simple_real_image_search(query: str) -> str:
    """Working DuckDuckGo image search using ddgs package"""
    try:
        print(f"🔍 DuckDuckGo image search: {query}")
        
        # Check if ddgs is available
        if not DDGS_AVAILABLE:
            print("❌ ddgs package not installed. Run: pip install ddgs")
            return ""
        
        # Create DDGS instance and search
        ddgs = DDGS()
        
        # Search for images
        results = list(ddgs.images(
            query=f"{query} medical diagram",
            region="us-en",
            safesearch="off",
            max_results=10
        ))
        
        if results:
            print(f"📸 Found {len(results)} images")
            
            # Filter for good quality images
            for result in results:
                image_url = result.get('image', '')
                title = result.get('title', '')
                
                # Skip unwanted image types
                skip_terms = ['logo', 'icon', 'avatar', 'banner', 'facebook', 'twitter', 'pinterest']
                if any(term in image_url.lower() or term in title.lower() for term in skip_terms):
                    continue
                
                # Must be reasonable image URL
                if (len(image_url) > 30 and 
                    image_url.startswith('http') and 
                    any(ext in image_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])):
                    
                    print(f"✅ Selected: {image_url[:60]}...")
                    return image_url
        
        print("❌ No suitable images found")
        return ""
        
    except Exception as e:
        print(f"❌ DuckDuckGo search failed: {e}")
        return ""

def robust_real_image_search(query: str) -> str:
    """Enhanced real image search using multiple strategies and sources"""
    try:
        print(f"🔍 Comprehensive real image search for: {query}")
        
        # Multiple search strategies with different query formats
        search_strategies = [
            # Strategy 1: Google Images - medical specific
            {
                "name": "Google Medical",
                "url": f"https://www.google.com/search?q={quote(f'{query} medical diagram illustration')}&tbm=isch&safe=active",
                "patterns": [
                    r'"ou":"(https?://[^"]+\.(?:jpg|jpeg|png|gif))"',
                    r'data-src="(https?://[^"]+\.(?:jpg|jpeg|png|gif))"',
                    r'"url":"(https?://[^"]+\.(?:jpg|jpeg|png|gif))"'
                ]
            },
            # Strategy 2: Google Images - educational
            {
                "name": "Google Educational", 
                "url": f"https://www.google.com/search?q={quote(f'{query} anatomy textbook')}&tbm=isch&tbs=isz:m",
                "patterns": [
                    r'"ou":"(https?://[^"]+\.(?:jpg|jpeg|png|gif))"',
                    r'imgurl=(https?://[^&]+\.(?:jpg|jpeg|png|gif))'
                ]
            },
            # Strategy 3: DuckDuckGo Images
            {
                "name": "DuckDuckGo",
                "url": f"https://duckduckgo.com/?q={quote(f'{query} medical')}&t=h_&iax=images&ia=images",
                "patterns": [
                    r'"image":"(https?://[^"]+\.(?:jpg|jpeg|png|gif))"',
                    r'data-src="(https?://[^"]+\.(?:jpg|jpeg|png|gif))"'
                ]
            },
            # Strategy 4: Bing Images
            {
                "name": "Bing",
                "url": f"https://www.bing.com/images/search?q={quote(f'{query} diagram')}&form=HDRSC2&first=1",
                "patterns": [
                    r'"murl":"(https?://[^"]+\.(?:jpg|jpeg|png|gif))"',
                    r'"imgurl":"(https?://[^"]+\.(?:jpg|jpeg|png|gif))"'
                ]
            },
            # Strategy 5: Google with simplified query
            {
                "name": "Google Simple",
                "url": f"https://www.google.com/search?q={quote(query.split()[0] + ' medical image')}&tbm=isch",
                "patterns": [
                    r'"ou":"(https?://[^"]+\.(?:jpg|jpeg|png))"'
                ]
            }
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
        
        for strategy in search_strategies:
            try:
                print(f"🌐 Trying {strategy['name']} search...")
                response = requests.get(strategy["url"], headers=headers, timeout=15)
                response.raise_for_status()
                html = response.text
                
                # Try each pattern for this strategy
                for pattern in strategy["patterns"]:
                    matches = re.findall(pattern, html)
                    if matches:
                        print(f"📋 Found {len(matches)} potential images with {strategy['name']}")
                        
                        for match in matches[:10]:  # Try first 10 matches
                            # Clean and validate URL
                            clean_url = match.replace('\\u003d', '=').replace('\\u0026', '&').replace('\\"', '"')
                            
                            # Enhanced filtering - skip unwanted image types
                            skip_terms = [
                                'logo', 'icon', 'button', 'avatar', 'thumbnail', 'favicon', 
                                'banner', 'header', 'footer', 'social', 'facebook', 'twitter',
                                'youtube', 'instagram', 'advertisement', 'ad.', '/ads/', 
                                'pixel.', 'tracking', 'analytics', 'badge'
                            ]
                            
                            if any(term in clean_url.lower() for term in skip_terms):
                                continue
                            
                            # Must be reasonable length and have proper extension
                            if len(clean_url) > 30 and any(ext in clean_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                                # Additional validation - prefer medical/educational domains
                                medical_domains = ['wiki', 'edu', 'org', 'medical', 'health', 'mayo', 'pubmed', 'ncbi']
                                is_medical_source = any(domain in clean_url.lower() for domain in medical_domains)
                                
                                if is_medical_source:
                                    print(f"🏥 Found medical source image: {clean_url[:60]}...")
                                    return clean_url
                                elif len(clean_url) > 40:  # General good quality image
                                    print(f"✅ Found quality image: {clean_url[:60]}...")
                                    return clean_url
                    
            except Exception as e:
                print(f"⚠️ {strategy['name']} search failed: {e}")
                continue
    
        print("❌ All real image search strategies failed")
        return ""
        
    except Exception as e:
        print(f"❌ Robust image search failed: {e}")
        return ""

def download_image(url: str, out_path: str, target_size: tuple = (1024, 768)) -> bool:
    """Download image with better error handling and automatic resizing to consistent dimensions"""
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
        
        # Save to temporary location first
        temp_path = out_path + ".temp"
        with open(temp_path, "wb") as f:
            f.write(response.content)
        
        # Open image with PIL and resize to consistent dimensions
        try:
            with Image.open(temp_path) as img:
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get original dimensions
                original_width, original_height = img.size
                print(f"📏 Original size: {original_width}x{original_height}")
                
                # Resize with high-quality resampling, maintaining aspect ratio
                # This will fit the image within the target size while preserving proportions
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Create a new image with the exact target size and paste the resized image centered
                final_img = Image.new('RGB', target_size, color='white')
                
                # Calculate position to center the image
                paste_x = (target_size[0] - img.width) // 2
                paste_y = (target_size[1] - img.height) // 2
                
                # Paste the resized image onto the white background
                final_img.paste(img, (paste_x, paste_y))
                
                # Save the final resized image
                final_img.save(out_path, 'PNG', quality=95, optimize=True)
                
                print(f"📐 Resized to: {target_size[0]}x{target_size[1]} → {out_path}")
                
                # Clean up temp file
                os.remove(temp_path)
                
                return True
                
        except Exception as resize_error:
            print(f"⚠️ Resize failed: {resize_error}")
            # Fallback: just save the original file
            os.rename(temp_path, out_path)
            print(f"✅ Saved original (no resize) → {out_path}")
            return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        # Clean up temp file if it exists
        temp_path = out_path + ".temp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def generate_random_background_color():
    """Generate a random, vibrant background color with good contrast potential"""
    
    # Curated list of medical-themed and vibrant colors that work well with text
    color_palette = [
        '#FFD700',  # Gold
        '#FF6B6B',  # Coral Red
        '#4ECDC4',  # Turquoise
        '#45B7D1',  # Sky Blue
        '#96CEB4',  # Mint Green
        '#FECA57',  # Orange Yellow
        '#FF9FF3',  # Pink
        '#54A0FF',  # Bright Blue
        '#5F27CD',  # Purple
        '#00D2D3',  # Cyan
        '#FF9F43',  # Orange
        '#10AC84',  # Green
        '#EE5A24',  # Red Orange
        '#0ABDE3',  # Light Blue
        '#C44569',  # Pink Purple
        '#40407A',  # Dark Blue
        '#706FD3',  # Lavender
        '#F97F51',  # Peach
        '#1DD1A1',  # Teal
        '#FECA57',  # Amber
    ]
    
    # Select random color from palette
    selected_color = random.choice(color_palette)
    print(f"🎨 Selected background color: {selected_color}")
    return selected_color

def get_contrasting_text_color(bg_color: str) -> str:
    """Determine if text should be black or white based on background color brightness"""
    
    # Remove # and convert to RGB
    hex_color = bg_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Calculate luminance (perceived brightness)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Use white text on dark backgrounds, black text on light backgrounds
    return '#FFFFFF' if luminance < 0.5 else '#000000'

def create_high_yield_image(high_yield_text: str, output_path: str):
    """Create wide, compact colorful background image for OBS overlay - optimized for single line display
    
    FIXED DIMENSIONS: 1920 x 200 pixels (locked for OBS consistency)
    """
    # Fixed dimensions - DO NOT CHANGE (optimized for OBS overlay)
    width = 1920
    height = 200
    
    # Generate random background color
    bg_color = generate_random_background_color()
    
    # Create image with random background
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load system fonts, fall back to default
    try:
        bold_font = None
        body_font = None
        
        # Try HelveticaNeue.ttc with different font indices (Bold is usually index 1)
        helvetica_neue_path = "/System/Library/Fonts/HelveticaNeue.ttc"
        if os.path.exists(helvetica_neue_path):
            try:
                # Index 1 is typically Bold in HelveticaNeue.ttc
                bold_font = ImageFont.truetype(helvetica_neue_path, 42, index=1)
                # Index 0 is typically Regular
                body_font = ImageFont.truetype(helvetica_neue_path, 38, index=0)
                print(f"✅ Loaded HelveticaNeue Bold (index 1) and Regular (index 0)")
            except Exception as e:
                print(f"⚠️ HelveticaNeue TTC index loading failed: {e}")
                # Fallback to using same font with larger size for "bold"
                try:
                    bold_font = ImageFont.truetype(helvetica_neue_path, 44)
                    body_font = ImageFont.truetype(helvetica_neue_path, 38)
                    print(f"✅ Loaded HelveticaNeue (same weight, different sizes)")
                except:
                    pass
        
        # Fallback to Helvetica.ttc if HelveticaNeue didn't work
        if not bold_font or not body_font:
            helvetica_path = "/System/Library/Fonts/Helvetica.ttc"
            if os.path.exists(helvetica_path):
                try:
                    # Index 1 is typically Bold in Helvetica.ttc
                    bold_font = ImageFont.truetype(helvetica_path, 42, index=1)
                    body_font = ImageFont.truetype(helvetica_path, 38, index=0)
                    print(f"✅ Loaded Helvetica Bold (index 1) and Regular (index 0)")
                except Exception as e:
                    print(f"⚠️ Helvetica TTC index loading failed: {e}")
                    # Fallback to same font different sizes
                    try:
                        bold_font = ImageFont.truetype(helvetica_path, 44)
                        body_font = ImageFont.truetype(helvetica_path, 38)
                        print(f"✅ Loaded Helvetica (same weight, different sizes)")
                    except:
                        pass
        
        # Final fallback to default fonts
        if not bold_font:
            bold_font = ImageFont.load_default()
        if not body_font:
            body_font = ImageFont.load_default()
            
    except Exception as e:
        print(f"❌ Font loading exception: {e}")
        bold_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Get contrasting text color based on background
    text_color = get_contrasting_text_color(bg_color)
    
    # Compact layout settings for horizontal display with proper margins
    horizontal_margin = 60  # Increased margins to prevent text cutoff
    y_center = height // 2  # Center vertically
    max_text_width = width - (2 * horizontal_margin)  # Available width for text
    
    # Display the EXACT GPT response text (no modification)
    gpt_text = high_yield_text.strip()
    
    # Calculate "Bottom Line: " width
    bold_text = "Bottom Line: "
    bold_bbox = draw.textbbox((0, 0), bold_text, font=bold_font)
    bold_width = bold_bbox[2] - bold_bbox[0]
    
    # Smart text wrapping based on ACTUAL pixel width, not character count
    # Maximum width for first line content (excluding "Bottom Line: ")
    max_first_line_content_width = max_text_width - bold_width
    
    # Try to fit everything on one line first
    full_text_bbox = draw.textbbox((0, 0), gpt_text, font=body_font)
    full_text_width = full_text_bbox[2] - full_text_bbox[0]
    
    wrapped_lines = []
    words = gpt_text.split()  # Split words for later truncation check
    
    if full_text_width <= max_first_line_content_width:
        # Fits on one line!
        wrapped_lines = [gpt_text]
    else:
        # Need to wrap - split into words and fit
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            test_bbox = draw.textbbox((0, 0), test_line, font=body_font)
            test_width = test_bbox[2] - test_bbox[0]
            
            # Check if this is the first line (needs room for "Bottom Line: ")
            if len(wrapped_lines) == 0:
                max_width = max_first_line_content_width
            else:
                max_width = max_text_width
            
            if test_width <= max_width:
                current_line.append(word)
            else:
                # Line is full, start new line
                if current_line:
                    wrapped_lines.append(' '.join(current_line))
                current_line = [word]
                
                # Limit to 2 lines
                if len(wrapped_lines) >= 2:
                    break
        
        # Add remaining words
        if current_line and len(wrapped_lines) < 2:
            wrapped_lines.append(' '.join(current_line))
    
    # If text was truncated, add ellipsis to last line
    if len(words) > len(' '.join(wrapped_lines).split()):
        if wrapped_lines:
            # Try to add ellipsis without exceeding width
            last_line = wrapped_lines[-1]
            test_line = last_line + "..."
            test_bbox = draw.textbbox((0, 0), test_line, font=body_font)
            test_width = test_bbox[2] - test_bbox[0]
            
            max_width = max_text_width if len(wrapped_lines) > 1 else max_first_line_content_width
            
            if test_width <= max_width:
                wrapped_lines[-1] = test_line
            else:
                # Remove words until ellipsis fits
                words_in_line = last_line.split()
                while len(words_in_line) > 1:
                    words_in_line = words_in_line[:-1]
                    test_line = ' '.join(words_in_line) + "..."
                    test_bbox = draw.textbbox((0, 0), test_line, font=body_font)
                    test_width = test_bbox[2] - test_bbox[0]
                    if test_width <= max_width:
                        wrapped_lines[-1] = test_line
                        break
    
    # Calculate total text height
    line_height = 50
    total_text_height = len(wrapped_lines) * line_height
    
    # Start position to center text vertically
    y_pos = y_center - (total_text_height // 2)
    
    # Draw each line with "Bottom Line:" in bold on first line only
    for i, line in enumerate(wrapped_lines):
        if i == 0:
            # First line: Draw "Bottom Line:" in bold, then the rest in regular
            regular_text = line
            regular_bbox = draw.textbbox((0, 0), regular_text, font=body_font)
            regular_width = regular_bbox[2] - regular_bbox[0]
            
            # Total width for centering (with proper margin check)
            total_width = bold_width + regular_width
            
            # Ensure we stay within margins
            if total_width > max_text_width:
                # Should not happen with our wrapping, but safety check
                start_x = horizontal_margin
            else:
                start_x = (width - total_width) // 2
            
            # Ensure minimum margin
            start_x = max(start_x, horizontal_margin)
            
            # Draw bold "Bottom Line:"
            draw.text((start_x, y_pos), bold_text, fill=text_color, font=bold_font)
            
            # Draw regular text after it
            draw.text((start_x + bold_width, y_pos), regular_text, fill=text_color, font=body_font)
        else:
            # Continuation lines: just regular text, centered with margin check
            line_bbox = draw.textbbox((0, 0), line, font=body_font)
            line_width = line_bbox[2] - line_bbox[0]
            
            # Center but respect margins
            if line_width > max_text_width:
                line_x = horizontal_margin
            else:
                line_x = (width - line_width) // 2
            
            # Ensure minimum margin
            line_x = max(line_x, horizontal_margin)
            
            draw.text((line_x, y_pos), line, fill=text_color, font=body_font)
        
        y_pos += line_height
    
    # No footer needed anymore
    
    # Save the image
    img.save(output_path, 'PNG', quality=95)
    print(f"✅ Bottom Line summary image saved → {output_path}")
    
    return output_path

def save_high_yield_info(content: str, medical_text: str):
    """Save High Yield information to file and create image version"""
    try:
        # Save text version
        with open(HIGH_YIELD_OUT, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("HIGH YIELD MEDICAL INFORMATION\n")
            f.write("=" * 60 + "\n\n")
            f.write("📋 SOURCE CASE:\n")
            f.write("-" * 30 + "\n")
            f.write(medical_text[:500] + ("..." if len(medical_text) > 500 else "") + "\n\n")
            f.write("🎯 HIGH YIELD CONTENT:\n")
            f.write("-" * 30 + "\n")
            f.write(content + "\n\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✅ High Yield info saved → {HIGH_YIELD_OUT}")
        
        # Create image version
        create_high_yield_image(content, OUT3)
        
    except Exception as e:
        print(f"❌ Error saving High Yield info: {e}")

def generate_patient_prompt(medical_text: str) -> str:
    """Generate patient image prompt using GPT"""
    if not OPENAI_API_KEY:
        return "A realistic medical patient for educational purposes"
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        system_prompt = """You are a medical education expert creating patient image prompts.

Based on the medical case text, create a prompt for generating a realistic patient image that would be educational.

Focus on:
- Patient demographics mentioned (age, gender)
- Visible physical findings if any
- General appearance appropriate for the condition
- Educational context

Keep it realistic and appropriate for medical education.

Format: Return a single paragraph prompt for image generation."""
        
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a patient image prompt from this medical case:\n\n{medical_text}"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ Error generating patient prompt: {e}")
        return "A realistic medical patient for educational purposes"

def generate_image_with_pollinations(prompt, width=888, height=664):
    """Generate patient image using Pollinations.AI
    
    STANDARDIZED DIMENSIONS: 888x664 (consistent with all AI-generated images)
    """
    try:
        print(f"🎨 Generating patient image with prompt: {prompt[:100]}...")
        
        # Pollinations.AI endpoint
        base_url = "https://image.pollinations.ai/prompt/"
        params = {
            'width': width,
            'height': height,
            'seed': -1,  # Random
            'enhance': 'true',
            'nologo': 'true'
        }
        
        # URL encode the prompt
        encoded_prompt = quote(prompt, safe='')
        url = f"{base_url}{encoded_prompt}"
        
        # Add parameters
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?{param_string}"
        
        print(f"🌐 Requesting image from Pollinations.AI...")
        
        # Make request with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(full_url, headers=headers, timeout=60)
        response.raise_for_status()
        
        if response.headers.get('content-type', '').startswith('image/'):
            # Save the image
            with open(PATIENT_OUT, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Patient image saved → {PATIENT_OUT}")
            
            # Save backup copy to AI Generated Patients directory
            save_backup_copy(PATIENT_OUT, AI_PATIENTS_DIR, "patient", f"(Generated from medical case)")
            
            return True
        else:
            print(f"❌ Invalid response type: {response.headers.get('content-type')}")
            return False
            
    except Exception as e:
        print(f"❌ Pollinations.AI generation failed: {e}")
        return False

def main():
    print("🚀 Starting Enhanced Medical Workflow...")
    
    # Step 1: Get medical text
    override = os.environ.get("IMAGE_SEARCH_TEXT", "").strip()
    text = override if override else get_clipboard_text()
    if not text:
        text = get_chrome_selection()
        if not text:
            sys.exit("❌ No text found in clipboard or Chrome selection.")
    
    # If clipboard contains a URL, prefer Chrome selection if available
    if re.match(r"^https?://", text.strip()):
        alt = get_chrome_selection()
        if alt:
            text = alt
    
    print(f"📝 Using medical text: {text[:100]}...")
    
    # Step 2: Generate High Yield information
    print("\n🎯 Generating High Yield information...")
    high_yield_content = generate_high_yield_info(text)
    save_high_yield_info(high_yield_content, text)
    
    # Step 3: Generate enhanced search queries
    print("\n🔍 Generating enhanced search queries...")
    queries = generate_enhanced_search_queries(text)
    print(f"📋 Search queries: {queries}")
    
    # Step 4: Hybrid approach - AI diagram + Real search image
    print("\n🖼️  Generating educational medical content...")
    generated_count = 0
    
    # Part A: Generate AI visual for image1.png (no text/labels, just visual depiction)
    if len(queries) > 0:
        print("🎨 Generating AI medical visual for image1.png...")
        if generate_medical_visual_with_ai(queries[0], OUT1, 1):
            generated_count += 1
            print(f"✅ AI visual generated: {queries[0]}")
        else:
            print(f"⚠️ AI generation failed for: {queries[0]}")
    
    # Part B: ONLY real search results for image2.png (no AI fallback)
    if len(queries) > 1:
        print("🔍 Searching for REAL image from search engines for image2.png...")
        image_url = simple_real_image_search(queries[1])
        if image_url and download_image(image_url, OUT2):
            generated_count += 1
            print(f"✅ Real search image downloaded: {queries[1]}")
        else:
            print(f"⚠️ Could not find real image for: {queries[1]}")
            
            # Try with the first query as fallback
            print("🔄 Trying real search with first query...")
            fallback_url = simple_real_image_search(queries[0])
            if fallback_url and download_image(fallback_url, OUT2):
                generated_count += 1
                print(f"✅ Real search image downloaded (fallback): {queries[0]}")
            else:
                print("❌ No real images found - image2.png will be skipped")
    
    # If we only have one query, try real search with broader terms
    elif len(queries) == 1:
        print("🔍 Searching for REAL image with single query for image2.png...")
        # Try broader search terms for better results
        broader_searches = [
            queries[0],
            f"{queries[0].split()[0]} medical",
            f"{queries[0].split()[0]} anatomy",
            "medical diagram illustration"
        ]
        
        found_real_image = False
        for search_term in broader_searches:
            if found_real_image:
                break
            print(f"🔍 Trying real search: {search_term}")
            image_url = simple_real_image_search(search_term)
            if image_url and download_image(image_url, OUT2):
                generated_count += 1
                print(f"✅ Real search image downloaded: {search_term}")
                found_real_image = True
                break
    
    # Additional real images: image2 v2.png and image2 v3.png
    print("\n🔍 Searching for additional real medical images...")
    
    # Generate search queries for additional images
    additional_searches = []
    if len(queries) > 0:
        # Create varied search terms based on main query
        base_term = queries[0].split()[0] if queries else "medical"
        additional_searches = [
            f"{base_term} anatomy diagram",
            f"{base_term} pathology illustration", 
            f"{base_term} medical illustration",
            f"{base_term} clinical image",
            f"medical {base_term} chart",
            "medical diagram pathophysiology",
            "anatomy medical illustration",
            "clinical pathology diagram"
        ]
    
    # Try to get image2 v2.png
    print("🔍 Searching for image2 v2.png...")
    found_v2 = False
    for search_term in additional_searches[:4]:  # Use first 4 search terms
        if found_v2:
            break
        print(f"🔍 Trying search: {search_term}")
        image_url = simple_real_image_search(search_term)
        if image_url and download_image(image_url, OUT2_V2):
            generated_count += 1
            print(f"✅ Additional real image downloaded (v2): {search_term}")
            # Save backup copy to AI Generated Images directory
            save_backup_copy(OUT2_V2, AI_IMAGES_DIR, "medical_visual", f"(v2: {search_term})")
            found_v2 = True
            break
    
    if not found_v2:
        print("⚠️ Could not find additional image for image2 v2.png")
    
    # Try to get image2 v3.png with different search terms
    print("🔍 Searching for image2 v3.png...")
    found_v3 = False
    for search_term in additional_searches[4:]:  # Use remaining search terms
        if found_v3:
            break
        print(f"🔍 Trying search: {search_term}")
        image_url = simple_real_image_search(search_term)
        if image_url and download_image(image_url, OUT2_V3):
            generated_count += 1
            print(f"✅ Additional real image downloaded (v3): {search_term}")
            # Save backup copy to AI Generated Images directory
            save_backup_copy(OUT2_V3, AI_IMAGES_DIR, "medical_visual", f"(v3: {search_term})")
            found_v3 = True
            break
    
    if not found_v3:
        print("⚠️ Could not find additional image for image2 v3.png")
    
    # Fallback strategies - only for image1 (AI generation)
    # image2 is REAL-SEARCH-ONLY, no AI fallbacks allowed
    if generated_count < 1 or (not os.path.exists(OUT1) or os.path.getsize(OUT1) == 0):
        print("🎨 Generating fallback AI visual for image1...")
        if generate_medical_visual_with_ai("medical anatomy illustration", OUT1, 1):
            print("✅ Fallback AI visual generated for image1")
            generated_count = max(generated_count, 1)
    
    # For image2, only mention that real search is required (no AI fallback)
    if not os.path.exists(OUT2) or os.path.getsize(OUT2) == 0:
        print("ℹ️  image2.png requires real search results - no AI fallback available")
        print("ℹ️  Try running again for different search results")
    
    # Step 6: Generate patient image
    print("\n👤 Generating patient image...")
    patient_prompt = generate_patient_prompt(text)
    patient_generated = generate_image_with_pollinations(patient_prompt)
    
    # Step 7: Summary
    print(f"\n✅ Enhanced Medical Workflow Complete!")
    print(f"📊 Results:")
    print(f"   • High Yield text: {HIGH_YIELD_OUT}")
    print(f"   • Bottom Line image (1920x200 - OBS optimized): {OUT3}")
    print(f"   • Educational content generated: {generated_count}/2")
    if generated_count > 0:
        print(f"     - image1.png: AI-generated medical visual (no text)")
        if generated_count > 1:
            print(f"     - image2.png: Real image from Google/DuckDuckGo search")
        elif os.path.exists(OUT2) and os.path.getsize(OUT2) > 0:
            print(f"     - image2.png: Real image from Google/DuckDuckGo search")
    print(f"   • Patient image: {'✅ Generated' if patient_generated else '❌ Failed'} → {PATIENT_OUT}")
    print(f"   • GPT-generated search queries: {len(queries)}")
    
    if generated_count == 0:
        print("⚠️  No educational content was generated. Check internet connection and try again.")
    elif generated_count == 1:
        print("⚠️  Only 1/2 images generated successfully.")
    
    print(f"\n🎯 All outputs saved to your Keyboard Maestro folder!")

if __name__ == "__main__":
    main()
