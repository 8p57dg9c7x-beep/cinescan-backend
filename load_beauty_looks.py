#!/usr/bin/env python3
"""
Load beauty look data into MongoDB
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB
MONGO_URL = os.getenv('MONGO_URL', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URL)
db = client['app_database']
beauty_collection = db['beauty_looks']

# Beauty look data
beauty_looks_data = [
  {
    "title": "Zendaya's Red Carpet Glow",
    "celebrity": "Zendaya",
    "category": "glam",
    "image": "https://images.unsplash.com/photo-1487412947147-5cebf100ffc2",
    "description": "Radiant bronze glow with sculpted contour and glossy nude lips. Perfect for evening events and red carpet moments.",
    "priceRange": "$180 - $320",
    "products": [
      { "name": "Fenty Beauty Pro Filt'r Foundation", "type": "Foundation", "price": "$39" },
      { "name": "Charlotte Tilbury Hollywood Contour Wand", "type": "Contour", "price": "$38" },
      { "name": "Pat McGrath Skin Fetish Highlighter", "type": "Highlighter", "price": "$55" },
      { "name": "MAC Velvet Teddy Lipstick", "type": "Lipstick", "price": "$22" }
    ],
    "budgetDupes": [
      { "name": "Maybelline Fit Me Foundation", "type": "Foundation", "price": "$8" },
      { "name": "NYX Wonder Stick Contour", "type": "Contour", "price": "$12" },
      { "name": "e.l.f. Halo Glow Highlighter", "type": "Highlighter", "price": "$9" },
      { "name": "Revlon Super Lustrous Lipstick", "type": "Lipstick", "price": "$7" }
    ]
  },
  {
    "title": "Hailey Bieber Clean Girl Makeup",
    "celebrity": "Hailey Bieber",
    "category": "natural",
    "image": "https://images.unsplash.com/photo-1616394584738-fc6e612e71b9",
    "description": "Fresh, dewy skin with barely-there makeup. The ultimate 'no-makeup' makeup look with glowing skin.",
    "priceRange": "$120 - $240",
    "products": [
      { "name": "Rhode Peptide Glazing Fluid", "type": "Serum", "price": "$29" },
      { "name": "Glossier Perfecting Skin Tint", "type": "Tint", "price": "$26" },
      { "name": "Rare Beauty Positive Light Liquid Highlighter", "type": "Highlighter", "price": "$25" },
      { "name": "Rhode Peptide Lip Treatment", "type": "Lip Care", "price": "$16" }
    ],
    "budgetDupes": [
      { "name": "CeraVe Hydrating Serum", "type": "Serum", "price": "$14" },
      { "name": "e.l.f. BB Cream", "type": "Tint", "price": "$6" },
      { "name": "Wet n Wild MegaGlo Highlighter", "type": "Highlighter", "price": "$5" },
      { "name": "Vaseline Lip Therapy", "type": "Lip Care", "price": "$3" }
    ]
  },
  {
    "title": "Rihanna's Bold Smokey Eye",
    "celebrity": "Rihanna",
    "category": "smokey",
    "image": "https://images.unsplash.com/photo-1512496015851-a90fb38ba796",
    "description": "Dramatic smokey eye with deep plum and black tones. Bold, intense, and unforgettable.",
    "priceRange": "$150 - $280",
    "products": [
      { "name": "Urban Decay Naked Smoky Palette", "type": "Eyeshadow", "price": "$54" },
      { "name": "Stila Stay All Day Liquid Liner", "type": "Eyeliner", "price": "$24" },
      { "name": "Too Faced Better Than Sex Mascara", "type": "Mascara", "price": "$27" },
      { "name": "Fenty Beauty Stunna Lip Paint", "type": "Lipstick", "price": "$27" }
    ],
    "budgetDupes": [
      { "name": "NYX Ultimate Shadow Palette Smokey", "type": "Eyeshadow", "price": "$18" },
      { "name": "L'Oréal Infallible Gel Liner", "type": "Eyeliner", "price": "$9" },
      { "name": "Maybelline Lash Sensational Mascara", "type": "Mascara", "price": "$9" },
      { "name": "NYX Soft Matte Lip Cream", "type": "Lipstick", "price": "$7" }
    ]
  },
  {
    "title": "Ariana Grande Everyday Glam",
    "celebrity": "Ariana Grande",
    "category": "everyday",
    "image": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9",
    "description": "Signature winged liner with soft brown tones and a nude lip. Perfect for daily wear.",
    "priceRange": "$90 - $180",
    "products": [
      { "name": "NARS Radiant Creamy Concealer", "type": "Concealer", "price": "$32" },
      { "name": "Benefit Roller Lash Mascara", "type": "Mascara", "price": "$27" },
      { "name": "Stila Liquid Eyeliner", "type": "Eyeliner", "price": "$24" },
      { "name": "MAC Velvet Teddy Lipstick", "type": "Lipstick", "price": "$22" }
    ],
    "budgetDupes": [
      { "name": "Maybelline Age Rewind Concealer", "type": "Concealer", "price": "$9" },
      { "name": "Essence Lash Princess Mascara", "type": "Mascara", "price": "$5" },
      { "name": "NYX Epic Ink Liner", "type": "Eyeliner", "price": "$10" },
      { "name": "Revlon ColorStay Lipstick", "type": "Lipstick", "price": "$9" }
    ]
  },
  {
    "title": "Sydney Sweeney Soft Romantic Look",
    "celebrity": "Sydney Sweeney",
    "category": "natural",
    "image": "https://images.unsplash.com/photo-1598440947619-2c35fc9aa908",
    "description": "Soft pink blush, fluttery lashes, and rosy lips. A dreamy, romantic makeup look.",
    "priceRange": "$100 - $200",
    "products": [
      { "name": "Rare Beauty Soft Pinch Blush", "type": "Blush", "price": "$23" },
      { "name": "Charlotte Tilbury Pillow Talk Lipstick", "type": "Lipstick", "price": "$35" },
      { "name": "Benefit Hoola Bronzer", "type": "Bronzer", "price": "$32" },
      { "name": "Dior Addict Lip Glow Oil", "type": "Lip Gloss", "price": "$38" }
    ],
    "budgetDupes": [
      { "name": "e.l.f. Camo Liquid Blush", "type": "Blush", "price": "$7" },
      { "name": "Maybelline Pink Please Lipstick", "type": "Lipstick", "price": "$8" },
      { "name": "Physician's Formula Butter Bronzer", "type": "Bronzer", "price": "$12" },
      { "name": "NYX Butter Gloss", "type": "Lip Gloss", "price": "$5" }
    ]
  },
  {
    "title": "Euphoria Festival Glitter",
    "celebrity": "Alexa Demie",
    "category": "festival",
    "image": "https://images.unsplash.com/photo-1583001809411-6ca800d9c1d4",
    "description": "Bold, artistic glitter eyes inspired by Euphoria. Colorful, fun, and unapologetically creative.",
    "priceRange": "$80 - $160",
    "products": [
      { "name": "NYX Face & Body Glitter", "type": "Glitter", "price": "$8" },
      { "name": "Urban Decay Heavy Metal Liner", "type": "Eyeliner", "price": "$22" },
      { "name": "Morphe 35B Color Burst Palette", "type": "Eyeshadow", "price": "$25" },
      { "name": "Fenty Beauty Gloss Bomb", "type": "Lip Gloss", "price": "$21" }
    ],
    "budgetDupes": [
      { "name": "e.l.f. Liquid Glitter Eyeshadow", "type": "Glitter", "price": "$6" },
      { "name": "LA Girl Shockwave Liner", "type": "Eyeliner", "price": "$5" },
      { "name": "NYX Ultimate Brights Palette", "type": "Eyeshadow", "price": "$18" },
      { "name": "Maybelline Lifter Gloss", "type": "Lip Gloss", "price": "$8" }
    ]
  },
  {
    "title": "Bella Hadid Sculpted Supermodel",
    "celebrity": "Bella Hadid",
    "category": "glam",
    "image": "https://images.unsplash.com/photo-1515688594390-b649af70d282",
    "description": "Chiseled contour, sculpted cheekbones, and a matte complexion. The ultimate supermodel look.",
    "priceRange": "$160 - $300",
    "products": [
      { "name": "Anastasia Beverly Hills Contour Kit", "type": "Contour", "price": "$40" },
      { "name": "Fenty Beauty Pro Filt'r Foundation", "type": "Foundation", "price": "$39" },
      { "name": "Huda Beauty Faux Filter Concealer", "type": "Concealer", "price": "$29" },
      { "name": "MAC Ruby Woo Lipstick", "type": "Lipstick", "price": "$22" }
    ],
    "budgetDupes": [
      { "name": "NYX Wonder Stick", "type": "Contour", "price": "$12" },
      { "name": "L'Oréal Infallible 24H Foundation", "type": "Foundation", "price": "$13" },
      { "name": "Maybelline Fit Me Concealer", "type": "Concealer", "price": "$7" },
      { "name": "Wet n Wild Stoplight Red Lipstick", "type": "Lipstick", "price": "$2" }
    ]
  }
]

def load_beauty_looks():
    """Load beauty look data into MongoDB"""
    try:
        # Clear existing beauty looks
        print("Clearing existing beauty looks...")
        beauty_collection.delete_many({})
        
        # Insert new beauty looks
        print(f"Inserting {len(beauty_looks_data)} beauty looks...")
        result = beauty_collection.insert_many(beauty_looks_data)
        
        print(f"✅ Successfully loaded {len(result.inserted_ids)} beauty looks!")
        
        # Verify by category
        print("\n📊 Beauty looks by category:")
        for category in ["natural", "glam", "smokey", "everyday", "festival"]:
            count = beauty_collection.count_documents({"category": category})
            print(f"  - {category.capitalize()}: {count} looks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading beauty looks: {e}")
        return False

if __name__ == "__main__":
    load_beauty_looks()
