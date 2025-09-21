import os, sys, subprocess, pathlib
import openai
import requests
from urllib.parse import quote
from PIL import Image
import io

# ---- Config ----
OUT = pathlib.Path("/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/generated_image.png")
WIDTH  = int(os.environ.get("IMG_WIDTH", 1024))
HEIGHT = int(os.environ.get("IMG_HEIGHT", 1024))

# GPT Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Set this environment variable
GPT_MODEL = "gpt-4o-mini"  # fast and cheap
# ---------------

def get_text_from_clipboard():
    """Get text from clipboard"""
    try:
        result = subprocess.run(['pbpaste'], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"Error getting text from clipboard: {e}")
        return ""

def generate_patient_prompt(medical_text):
    """Use GPT to generate a patient image prompt based on medical case text"""
    if not OPENAI_API_KEY:
        print("Warning: No OpenAI API key found. Using fallback prompt.")
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
        
        prompt = response.choices[0].message.content.strip()
        print(f"Generated patient prompt: {prompt}")
        return prompt
        
    except Exception as e:
        print(f"Error generating prompt with GPT: {e}")
        # Fallback prompt
        return "A professional headshot of a patient in a medical setting, neutral expression, good lighting"

def remove_pollinations_watermark(image_bytes):
    """Clean crop to remove the pollinations.ai watermark from bottom"""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get image dimensions
        width, height = image.size
        
        # The watermark typically appears in bottom 6-8% of the image
        # Simple clean crop - just remove the bottom strip
        watermark_strip_height = int(height * 0.08)  # Remove bottom 8%
        cropped_height = height - watermark_strip_height
        
        # Clean crop - no resizing or filling
        clean_image = image.crop((0, 0, width, cropped_height))
        
        print(f"🎨 Clean crop: removed bottom {watermark_strip_height}px watermark strip")
        return clean_image
        
    except Exception as e:
        print(f"⚠️ Could not remove watermark: {e}, saving original")
        # If processing fails, return original image
        return Image.open(io.BytesIO(image_bytes))

def generate_image_with_pollinations(prompt):
    """Generate image using Pollinations.AI API with retry logic"""
    import time
    
    # Ensure destination folder exists
    OUT.parent.mkdir(parents=True, exist_ok=True)
    
    # URL encode the prompt
    encoded_prompt = quote(prompt)
    
    # Try multiple endpoints if main one fails
    endpoints = [
        f"https://image.pollinations.ai/prompt/{encoded_prompt}",
        f"https://pollinations.ai/p/{encoded_prompt}",  # Alternative endpoint
    ]
    
    # Add size parameters if specified
    params = []
    if WIDTH != 1024:
        params.append(f"width={WIDTH}")
    if HEIGHT != 1024:
        params.append(f"height={HEIGHT}")
        
    param_string = "?" + "&".join(params) if params else ""
    
    for attempt in range(3):  # Try 3 times
        for i, base_url in enumerate(endpoints):
            try:
                pollinations_url = base_url + param_string
                print(f"🌸 Generating image via Pollinations.AI (attempt {attempt+1}, endpoint {i+1})...")
                print(f"📡 Request: {pollinations_url}")
                
                # Make request with timeout
                response = requests.get(pollinations_url, timeout=45)
                
                if response.status_code == 200:
                    # Success! Process and save clean image
                    cleaned_image = remove_pollinations_watermark(response.content)
                    cleaned_image.save(OUT, 'JPEG', quality=95)
                    
                    print(f"✅ Saved clean patient image → {OUT}")
                    return
                else:
                    print(f"⚠️ HTTP {response.status_code} from endpoint {i+1}")
                    
            except Exception as e:
                print(f"⚠️ Error with endpoint {i+1}: {e}")
                continue
        
        if attempt < 2:  # Don't wait after last attempt
            print(f"🔄 Retrying in {(attempt+1)*2} seconds...")
            time.sleep((attempt+1)*2)
    
    # All attempts failed - create a simple fallback image URL
    print(f"❌ All Pollinations.AI attempts failed. Creating fallback...")
    try:
        # Simple fallback - try a very basic prompt
        simple_prompt = "professional headshot medical patient"
        fallback_url = f"https://image.pollinations.ai/prompt/{quote(simple_prompt)}"
        
        response = requests.get(fallback_url, timeout=30)
        if response.status_code == 200:
            # Process fallback image too
            cleaned_image = remove_pollinations_watermark(response.content)
            cleaned_image.save(OUT, 'JPEG', quality=95)
            print(f"✅ Saved clean fallback patient image → {OUT}")
            return
            
    except Exception as fallback_error:
        print(f"❌ Fallback also failed: {fallback_error}")
    
    print(f"❌ Could not generate image. All attempts failed.")
    raise Exception("All image generation attempts failed")

def main():
    print("Starting clipboard to patient image generation...")
    
    # Step 1: Get text from clipboard
    print("Getting text from clipboard...")
    clipboard_text = get_text_from_clipboard()
    if not clipboard_text:
        sys.exit("Clipboard is empty (no text found).")
    
    print(f"Using clipboard text: {clipboard_text[:200]}...")  # Show first 200 chars
    
    # Step 2: Generate patient image prompt using GPT
    print("Generating patient image prompt...")
    patient_prompt = generate_patient_prompt(clipboard_text)
    
    # Step 3: Generate image using Pollinations.AI
    print("Generating patient image...")
    generate_image_with_pollinations(patient_prompt)
    
    print("✅ Patient image generated successfully!")

if __name__ == "__main__":
    main()
