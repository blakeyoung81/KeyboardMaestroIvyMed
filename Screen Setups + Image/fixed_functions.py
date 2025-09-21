import requests
import re
from urllib.parse import quote

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
