#!/usr/bin/env python3
"""
Load additional beauty looks into MongoDB
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

# Additional beauty look data
additional_looks_data = [
  {
    "title": "Selena Gomez Soft Glam",
    "celebrity": "Selena Gomez",
    "category": "glam",
    "image": "https://images.unsplash.com/photo-1524502397800-2eeaad7c3fe5",
    "description": "Soft, elegant glam with warm tones and a polished finish. Perfect for date nights and special occasions.",
    "priceRange": "$140 - $260",
    "products": [
      { "name": "Rare Beauty Liquid Touch Weightless Foundation", "type": "Foundation", "price": "$29" },
      { "name": "Rare Beauty Stay Vulnerable Melting Blush", "type": "Blush", "price": "$23" },
      { "name": "Urban Decay Naked3 Palette", "type": "Eyeshadow", "price": "$54" },
      { "name": "MAC Soar Lip Liner", "type": "Lip Liner", "price": "$20" }
    ],
    "budgetDupes": [
      { "name": "Maybelline Dream Radiant Foundation", "type": "Foundation", "price": "$9" },
      { "name": "e.l.f. Putty Blush", "type": "Blush", "price": "$6" },
      { "name": "NYX Ultimate Warm Neutrals Palette", "type": "Eyeshadow", "price": "$18" },
      { "name": "NYX Slim Lip Pencil", "type": "Lip Liner", "price": "$5" }
    ]
  },
  {
    "title": "Dua Lipa Bold Red Lip",
    "celebrity": "Dua Lipa",
    "category": "glam",
    "image": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11",
    "description": "Classic Hollywood glam with a bold red lip and winged liner. Timeless and powerful.",
    "priceRange": "$120 - $240",
    "products": [
      { "name": "Estée Lauder Double Wear Foundation", "type": "Foundation", "price": "$52" },
      { "name": "MAC Ruby Woo Lipstick", "type": "Lipstick", "price": "$22" },
      { "name": "Stila Stay All Day Liner", "type": "Eyeliner", "price": "$24" },
      { "name": "Laura Mercier Translucent Powder", "type": "Powder", "price": "$39" }
    ],
    "budgetDupes": [
      { "name": "Revlon ColorStay Foundation", "type": "Foundation", "price": "$13" },
      { "name": "Maybelline Color Sensational (Red)", "type": "Lipstick", "price": "$8" },
      { "name": "NYX Epic Ink Liner", "type": "Eyeliner", "price": "$10" },
      { "name": "Rimmel Stay Matte Powder", "type": "Powder", "price": "$7" }
    ]
  },
  {
    "title": "Olivia Rodrigo Grunge Glam",
    "celebrity": "Olivia Rodrigo",
    "category": "everyday",
    "image": "https://images.unsplash.com/photo-1529139574466-a303027c1d8b",
    "description": "Effortless grunge-inspired makeup with smudged liner and natural lips. Cool and edgy.",
    "priceRange": "$70 - $150",
    "products": [
      { "name": "Glossier Boy Brow", "type": "Brow Gel", "price": "$18" },
      { "name": "Urban Decay 24/7 Glide-On Pencil", "type": "Eyeliner", "price": "$24" },
      { "name": "Milk Makeup Lip + Cheek", "type": "Lip & Cheek", "price": "$24" },
      { "name": "Tower 28 BeachPlease Blush", "type": "Blush", "price": "$20" }
    ],
    "budgetDupes": [
      { "name": "e.l.f. Wow Brow Gel", "type": "Brow Gel", "price": "$4" },
      { "name": "NYX Retractable Eye Liner", "type": "Eyeliner", "price": "$9" },
      { "name": "e.l.f. Lip & Cheek Cream", "type": "Lip & Cheek", "price": "$5" },
      { "name": "Physician's Formula Happy Booster Blush", "type": "Blush", "price": "$10" }
    ]
  },
  {
    "title": "Megan Fox Sultry Smokey",
    "celebrity": "Megan Fox",
    "category": "smokey",
    "image": "https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453",
    "description": "Intense, sultry smokey eye with bronze tones and nude lips. Seductive and dramatic.",
    "priceRange": "$160 - $300",
    "products": [
      { "name": "Anastasia Beverly Hills Soft Glam Palette", "type": "Eyeshadow", "price": "$45" },
      { "name": "MAC Pro Longwear Concealer", "type": "Concealer", "price": "$26" },
      { "name": "Lancôme Lash Idôle Mascara", "type": "Mascara", "price": "$28" },
      { "name": "Charlotte Tilbury Pillow Talk Lipstick", "type": "Lipstick", "price": "$35" }
    ],
    "budgetDupes": [
      { "name": "Maybelline The Nudes Palette", "type": "Eyeshadow", "price": "$12" },
      { "name": "Maybelline Age Rewind Concealer", "type": "Concealer", "price": "$9" },
      { "name": "L'Oréal Lash Paradise Mascara", "type": "Mascara", "price": "$11" },
      { "name": "NYX Butter Lipstick (Nude)", "type": "Lipstick", "price": "$7" }
    ]
  },
  {
    "title": "Sabrina Carpenter Doe Eyes",
    "celebrity": "Sabrina Carpenter",
    "category": "natural",
    "image": "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04",
    "description": "Sweet, doe-eyed look with soft pink tones and glossy lips. Fresh and youthful.",
    "priceRange": "$90 - $180",
    "products": [
      { "name": "IT Cosmetics CC+ Cream", "type": "CC Cream", "price": "$44" },
      { "name": "Benefit Hoola Bronzer", "type": "Bronzer", "price": "$32" },
      { "name": "Glossier Cloud Paint", "type": "Blush", "price": "$18" },
      { "name": "Fenty Beauty Gloss Bomb", "type": "Lip Gloss", "price": "$21" }
    ],
    "budgetDupes": [
      { "name": "Garnier BB Cream", "type": "CC Cream", "price": "$9" },
      { "name": "Physician's Formula Butter Bronzer", "type": "Bronzer", "price": "$12" },
      { "name": "e.l.f. Camo Liquid Blush", "type": "Blush", "price": "$7" },
      { "name": "Maybelline Lifter Gloss", "type": "Lip Gloss", "price": "$8" }
    ]
  },
  {
    "title": "Florence Pugh Editorial Glam",
    "celebrity": "Florence Pugh",
    "category": "glam",
    "image": "https://images.unsplash.com/photo-1494790108377-be9c29b29330",
    "description": "Bold, editorial makeup with graphic liner and statement lips. Artistic and modern.",
    "priceRange": "$150 - $280",
    "products": [
      { "name": "Pat McGrath Mothership Palette", "type": "Eyeshadow", "price": "$128" },
      { "name": "KVD Beauty Tattoo Liner", "type": "Eyeliner", "price": "$22" },
      { "name": "MAC Retro Matte Lipstick", "type": "Lipstick", "price": "$22" },
      { "name": "Hourglass Ambient Lighting Powder", "type": "Powder", "price": "$48" }
    ],
    "budgetDupes": [
      { "name": "ColourPop Super Shock Shadows", "type": "Eyeshadow", "price": "$18" },
      { "name": "Essence Superfine Liner", "type": "Eyeliner", "price": "$4" },
      { "name": "Wet n Wild Matte Lipstick", "type": "Lipstick", "price": "$3" },
      { "name": "e.l.f. Halo Glow Powder", "type": "Powder", "price": "$9" }
    ]
  },
  {
    "title": "Taylor Swift Vintage Glam",
    "celebrity": "Taylor Swift",
    "category": "everyday",
    "image": "https://images.unsplash.com/photo-1534528741775-53994a69daeb",
    "description": "Vintage-inspired everyday glam with red lips and classic winged liner. Timeless elegance.",
    "priceRange": "$100 - $200",
    "products": [
      { "name": "NARS Sheer Glow Foundation", "type": "Foundation", "price": "$47" },
      { "name": "Benefit They're Real Mascara", "type": "Mascara", "price": "$27" },
      { "name": "MAC Ruby Woo Lipstick", "type": "Lipstick", "price": "$22" },
      { "name": "Anastasia Beverly Hills Brow Wiz", "type": "Brow Pencil", "price": "$25" }
    ],
    "budgetDupes": [
      { "name": "L'Oréal True Match Foundation", "type": "Foundation", "price": "$12" },
      { "name": "Essence Lash Princess Mascara", "type": "Mascara", "price": "$5" },
      { "name": "NYX Matte Lipstick (Red)", "type": "Lipstick", "price": "$7" },
      { "name": "NYX Micro Brow Pencil", "type": "Brow Pencil", "price": "$10" }
    ]
  },
  {
    "title": "Jenna Ortega Dark Romance",
    "celebrity": "Jenna Ortega",
    "category": "festival",
    "image": "https://images.unsplash.com/photo-1515377905703-c4788e51af15",
    "description": "Dark, romantic festival makeup with deep burgundy tones and bold lashes. Gothic and captivating.",
    "priceRange": "$130 - $250",
    "products": [
      { "name": "Urban Decay Born to Run Palette", "type": "Eyeshadow", "price": "$49" },
      { "name": "Velour Lashes Effortless", "type": "Lashes", "price": "$29" },
      { "name": "MAC Diva Lipstick", "type": "Lipstick", "price": "$22" },
      { "name": "Too Faced Born This Way Concealer", "type": "Concealer", "price": "$30" }
    ],
    "budgetDupes": [
      { "name": "Makeup Revolution Forever Flawless Palette", "type": "Eyeshadow", "price": "$10" },
      { "name": "Kiss Lashes", "type": "Lashes", "price": "$5" },
      { "name": "NYX Soft Matte Lip Cream (Dark)", "type": "Lipstick", "price": "$7" },
      { "name": "Catrice Liquid Camouflage Concealer", "type": "Concealer", "price": "$7" }
    ]
  }
]

def load_additional_looks():
    """Load additional beauty looks into MongoDB"""
    try:
        # Insert new beauty looks (don't clear existing ones)
        print(f"Adding {len(additional_looks_data)} new beauty looks...")
        result = beauty_collection.insert_many(additional_looks_data)
        
        print(f"✅ Successfully added {len(result.inserted_ids)} new beauty looks!")
        
        # Verify total count and by category
        total = beauty_collection.count_documents({})
        print(f"\n📊 Total beauty looks in database: {total}")
        
        print("\nBreakdown by category:")
        for category in ["natural", "glam", "smokey", "everyday", "festival"]:
            count = beauty_collection.count_documents({"category": category})
            print(f"  - {category.capitalize()}: {count} looks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading additional looks: {e}")
        return False

if __name__ == "__main__":
    load_additional_looks()
