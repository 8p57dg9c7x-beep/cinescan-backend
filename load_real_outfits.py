#!/usr/bin/env python3
"""
Load real outfit data into MongoDB
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB
MONGO_URL = os.getenv('MONGO_URL', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URL)
db = client['app_database']
outfits_collection = db['outfits']

# Real outfit data
outfits_data = [
  {
    "title": "Minimal Beige Streetwear",
    "category": "streetwear",
    "image": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?w=800&q=80",
    "description": "Clean neutral streetwear look with layered textures.",
    "priceRange": "$150 - $280",
    "gender": "female",
    "items": [
      { "name": "Oversized Beige Hoodie", "price": "$65" },
      { "name": "Cream Wide-Leg Sweatpants", "price": "$45" },
      { "name": "White Minimal Sneakers", "price": "$120" }
    ],
    "budgetAlternatives": [
      { "name": "Beige Hoodie (H&M)", "price": "$24" },
      { "name": "Cream Sweatpants (SHEIN)", "price": "$16" },
      { "name": "White Sneakers (Zara)", "price": "$39" }
    ]
  },
  {
    "title": "Urban Black Techwear",
    "category": "streetwear",
    "image": "https://images.unsplash.com/photo-1516826957135-700dedea698c?w=800&q=80",
    "description": "All-black techwear with futuristic lines and utility details.",
    "priceRange": "$180 - $300",
    "gender": "male",
    "items": [
      { "name": "Black Utility Jacket", "price": "$110" },
      { "name": "Cargo Tactical Pants", "price": "$75" },
      { "name": "Chunky Black Sneakers", "price": "$130" }
    ],
    "budgetAlternatives": [
      { "name": "Utility Jacket (ASOS)", "price": "$45" },
      { "name": "Cargo Pants (BoohooMAN)", "price": "$28" },
      { "name": "Chunky Sneakers (SHEIN)", "price": "$32" }
    ]
  },
  {
    "title": "Soft Girl Pastel Fit",
    "category": "minimal",
    "image": "https://images.unsplash.com/photo-1520256862855-398228c41684?w=800&q=80",
    "description": "A light pastel outfit with soft edges and a dreamy aesthetic.",
    "priceRange": "$80 - $150",
    "gender": "female",
    "items": [
      { "name": "Pastel Knit Sweater", "price": "$55" },
      { "name": "White Tennis Skirt", "price": "$35" },
      { "name": "Light Pink Sneakers", "price": "$60" }
    ],
    "budgetAlternatives": [
      { "name": "Pastel Sweater (SHEIN)", "price": "$14" },
      { "name": "Tennis Skirt (H&M)", "price": "$18" },
      { "name": "Pink Sneakers (Amazon)", "price": "$28" }
    ]
  },
  {
    "title": "Luxury Monochrome Black",
    "category": "luxury",
    "image": "https://images.unsplash.com/photo-1516574187841-cb9cc2ca948b?w=800&q=80",
    "description": "A sleek luxury-inspired fit with subtle upscale elements.",
    "priceRange": "$300 - $600",
    "gender": "male",
    "items": [
      { "name": "Black Wool Coat", "price": "$260" },
      { "name": "Tailored Black Trousers", "price": "$140" },
      { "name": "Premium Leather Loafers", "price": "$220" }
    ],
    "budgetAlternatives": [
      { "name": "Black Coat (Zara)", "price": "$89" },
      { "name": "Tailored Pants (H&M)", "price": "$35" },
      { "name": "Leather Loafers (ASOS)", "price": "$48" }
    ]
  },
  {
    "title": "Elegant Date Night Dress",
    "category": "elegant",
    "image": "https://images.unsplash.com/photo-1515378791036-0648a3ef77b2?w=800&q=80",
    "description": "A timeless satin dress paired with minimalist heels.",
    "priceRange": "$120 - $250",
    "gender": "female",
    "items": [
      { "name": "Satin Slip Dress", "price": "$120" },
      { "name": "Strappy Nude Heels", "price": "$85" },
      { "name": "Gold Minimal Necklace", "price": "$45" }
    ],
    "budgetAlternatives": [
      { "name": "Slip Dress (SHEIN)", "price": "$16" },
      { "name": "Nude Heels (Public Desire)", "price": "$29" },
      { "name": "Gold Necklace (Amazon)", "price": "$12" }
    ]
  },
  {
    "title": "Athleisure Morning Run",
    "category": "sport",
    "image": "https://images.unsplash.com/photo-1599058917212-d750089bc07b?w=800&q=80",
    "description": "A functional but stylish morning workout outfit.",
    "priceRange": "$90 - $180",
    "gender": "female",
    "items": [
      { "name": "High-Waisted Running Leggings", "price": "$60" },
      { "name": "Performance Sports Bra", "price": "$40" },
      { "name": "Lightweight Running Shoes", "price": "$80" }
    ],
    "budgetAlternatives": [
      { "name": "Running Leggings (SHEIN)", "price": "$12" },
      { "name": "Sports Bra (H&M)", "price": "$15" },
      { "name": "Running Shoes (Decathlon)", "price": "$29" }
    ]
  },
  {
    "title": "Classic Retro Street Fit",
    "category": "streetwear",
    "image": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=800&q=80",
    "description": "90s-inspired denim streetwear with clean sneakers.",
    "priceRange": "$130 - $220",
    "gender": "female",
    "items": [
      { "name": "Vintage Denim Jacket", "price": "$85" },
      { "name": "High-Waisted Blue Jeans", "price": "$60" },
      { "name": "White Retro Sneakers", "price": "$75" }
    ],
    "budgetAlternatives": [
      { "name": "Denim Jacket (Bershka)", "price": "$28" },
      { "name": "Blue Jeans (SHEIN)", "price": "$18" },
      { "name": "Retro Sneakers (Zara)", "price": "$39" }
    ]
  },
  {
    "title": "Luxury Business Casual",
    "category": "luxury",
    "image": "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=800&q=80",
    "description": "A refined, upscale business casual outfit.",
    "priceRange": "$280 - $550",
    "gender": "male",
    "items": [
      { "name": "Camel Coat", "price": "$240" },
      { "name": "Fine Knit Turtleneck", "price": "$85" },
      { "name": "Slim-Fit Wool Trousers", "price": "$150" }
    ],
    "budgetAlternatives": [
      { "name": "Camel Coat (H&M)", "price": "$69" },
      { "name": "Knit Turtleneck (SHEIN)", "price": "$12" },
      { "name": "Wool Trousers (ASOS)", "price": "$34" }
    ]
  },
  {
    "title": "Minimal All-White Summer Fit",
    "category": "minimal",
    "image": "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=800&q=80",
    "description": "A bright, fresh minimalist outfit for warm days.",
    "priceRange": "$70 - $140",
    "gender": "female",
    "items": [
      { "name": "White Linen Shirt", "price": "$45" },
      { "name": "White Flow Shorts", "price": "$35" },
      { "name": "White Flat Sandals", "price": "$55" }
    ],
    "budgetAlternatives": [
      { "name": "Linen Shirt (SHEIN)", "price": "$13" },
      { "name": "Flow Shorts (Zara)", "price": "$19" },
      { "name": "Flat Sandals (Amazon)", "price": "$17" }
    ]
  },
  {
    "title": "Sporty Street Runner",
    "category": "sport",
    "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800&q=80",
    "description": "Sport-inspired streetwear with a modern silhouette.",
    "priceRange": "$100 - $190",
    "gender": "male",
    "items": [
      { "name": "Tech Running Jacket", "price": "$90" },
      { "name": "Performance Joggers", "price": "$55" },
      { "name": "Cushioned Running Shoes", "price": "$95" }
    ],
    "budgetAlternatives": [
      { "name": "Running Jacket (Decathlon)", "price": "$29" },
      { "name": "Joggers (SHEIN)", "price": "$16" },
      { "name": "Running Shoes (H&M)", "price": "$28" }
    ]
  }
]

def load_outfits():
    """Load real outfit data into MongoDB"""
    try:
        # Clear existing outfits (optional - remove if you want to keep old data)
        print("Clearing existing outfits...")
        outfits_collection.delete_many({})
        
        # Insert new outfits
        print(f"Inserting {len(outfits_data)} real outfits...")
        result = outfits_collection.insert_many(outfits_data)
        
        print(f"✅ Successfully loaded {len(result.inserted_ids)} outfits!")
        
        # Verify by category
        print("\n📊 Outfits by category:")
        for category in ["streetwear", "minimal", "luxury", "elegant", "sport"]:
            count = outfits_collection.count_documents({"category": category})
            print(f"  - {category.capitalize()}: {count} outfits")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading outfits: {e}")
        return False

if __name__ == "__main__":
    load_outfits()
