#!/usr/bin/env python3
"""Load demo outfits with WORKING image URLs"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017/')
mongo_client = MongoClient(MONGO_URL)
db = mongo_client['cinescan']
outfits_collection = db['outfits']

# Clear existing outfits
outfits_collection.delete_many({})
print("Cleared existing outfits")

# Demo Outfits with WORKING Unsplash Images
demo_outfits = [
    # Streetwear
    {
        "title": "Urban Streetwear Essential",
        "category": "streetwear",
        "image": "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=400&h=600&fit=crop",
        "priceRange": "$250 - $400",
        "description": "Classic oversized street style perfect for everyday wear",
        "items": [
            {"name": "Oversized Graphic Hoodie", "price": "$120", "link": "https://www.ssense.com"},
            {"name": "Black Cargo Pants", "price": "$150", "link": "https://www.urbanoutfitters.com"},
            {"name": "Nike Air Force 1", "price": "$110", "link": "https://www.nike.com"}
        ],
        "budgetAlternatives": [
            {"name": "H&M Oversized Hoodie", "price": "$30", "link": "https://www2.hm.com"},
            {"name": "Uniqlo Cargo Pants", "price": "$50", "link": "https://www.uniqlo.com"},
            {"name": "Vans Old Skool", "price": "$65", "link": "https://www.vans.com"}
        ],
        "isCelebrity": False
    },
    {
        "title": "Minimalist Street Style",
        "category": "streetwear",
        "image": "https://images.unsplash.com/photo-1490367532201-b9bc1dc483f6?w=400&h=600&fit=crop",
        "priceRange": "$180 - $320",
        "description": "Clean and simple street aesthetic",
        "items": [
            {"name": "White Basic Tee", "price": "$45", "link": "https://www.ssense.com"},
            {"name": "Black Slim Jeans", "price": "$135", "link": "https://www.levi.com"},
            {"name": "Converse Chuck Taylor", "price": "$70", "link": "https://www.converse.com"}
        ],
        "budgetAlternatives": [
            {"name": "Uniqlo White Tee Pack", "price": "$15", "link": "https://www.uniqlo.com"},
            {"name": "H&M Slim Jeans", "price": "$40", "link": "https://www2.hm.com"}
        ],
        "isCelebrity": False
    },
    
    # Luxury
    {
        "title": "Designer Casual Luxury",
        "category": "luxury",
        "image": "https://images.unsplash.com/photo-1441984904996-e0b6ba687e04?w=400&h=600&fit=crop",
        "priceRange": "$800 - $1500",
        "description": "Elevated basics with designer touches",
        "items": [
            {"name": "Gucci Logo Sweatshirt", "price": "$850", "link": "https://www.gucci.com"},
            {"name": "Saint Laurent Jeans", "price": "$590", "link": "https://www.ysl.com"}
        ],
        "budgetAlternatives": [
            {"name": "Zara Designer-Inspired Sweatshirt", "price": "$60", "link": "https://www.zara.com"},
            {"name": "Mango Slim Fit Jeans", "price": "$80", "link": "https://shop.mango.com"}
        ],
        "isCelebrity": False
    },
    
    # Minimal
    {
        "title": "Pure Minimalist Look",
        "category": "minimal",
        "image": "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=600&fit=crop",
        "priceRange": "$200 - $350",
        "description": "Less is more - clean lines and neutral tones",
        "items": [
            {"name": "Everlane Cotton Shirt", "price": "$55", "link": "https://www.everlane.com"},
            {"name": "COS Minimal Pants", "price": "$129", "link": "https://www.cosstores.com"}
        ],
        "budgetAlternatives": [
            {"name": "Uniqlo Oxford Shirt", "price": "$30", "link": "https://www.uniqlo.com"},
            {"name": "H&M Relaxed Pants", "price": "$35", "link": "https://www2.hm.com"}
        ],
        "isCelebrity": False
    },
    
    # Sport
    {
        "title": "Athleisure Performance",
        "category": "sport",
        "image": "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=400&h=600&fit=crop",
        "priceRange": "$180 - $300",
        "description": "Gym to street versatility",
        "items": [
            {"name": "Nike Tech Fleece Hoodie", "price": "$110", "link": "https://www.nike.com"},
            {"name": "Lululemon Joggers", "price": "$128", "link": "https://www.lululemon.com"}
        ],
        "budgetAlternatives": [
            {"name": "Old Navy Active Hoodie", "price": "$35", "link": "https://oldnavy.gap.com"},
            {"name": "Target Joggers", "price": "$30", "link": "https://www.target.com"}
        ],
        "isCelebrity": False
    },
    
    # Elegant
    {
        "title": "Business Casual Refined",
        "category": "elegant",
        "image": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=600&fit=crop",
        "priceRange": "$350 - $650",
        "description": "Professional and polished",
        "items": [
            {"name": "Brooks Brothers Blazer", "price": "$398", "link": "https://www.brooksbrothers.com"},
            {"name": "Banana Republic Chinos", "price": "$98", "link": "https://bananarepublic.gap.com"}
        ],
        "budgetAlternatives": [
            {"name": "H&M Slim Fit Blazer", "price": "$80", "link": "https://www2.hm.com"},
            {"name": "Old Navy Chinos", "price": "$35", "link": "https://oldnavy.gap.com"}
        ],
        "isCelebrity": False
    },
    
    # Celebrity Outfits
    {
        "title": "Dress Like Drake",
        "category": "celebrity",
        "image": "https://images.unsplash.com/photo-1564859228273-274232fdb516?w=400&h=600&fit=crop",
        "priceRange": "$600 - $1200",
        "description": "OVO aesthetic - luxury streetwear meets hip-hop",
        "celebrity": "Drake",
        "items": [
            {"name": "Canada Goose Parka", "price": "$995", "link": "https://www.canadagoose.com"},
            {"name": "Amiri Jeans", "price": "$890", "link": "https://www.amiri.com"}
        ],
        "budgetAlternatives": [
            {"name": "Columbia Winter Jacket", "price": "$150", "link": "https://www.columbia.com"},
            {"name": "Zara Distressed Jeans", "price": "$60", "link": "https://www.zara.com"}
        ],
        "isCelebrity": True
    },
    {
        "title": "Dress Like Zendaya",
        "category": "celebrity",
        "image": "https://images.unsplash.com/photo-1509631179647-0177331693ae?w=400&h=600&fit=crop",
        "priceRange": "$500 - $900",
        "description": "Effortlessly chic and fashion-forward",
        "celebrity": "Zendaya",
        "items": [
            {"name": "Valentino Blazer", "price": "$2890", "link": "https://www.valentino.com"},
            {"name": "Isabel Marant Pants", "price": "$495", "link": "https://www.isabelmarant.com"}
        ],
        "budgetAlternatives": [
            {"name": "Mango Tailored Blazer", "price": "$120", "link": "https://shop.mango.com"},
            {"name": "Zara Wide Leg Pants", "price": "$50", "link": "https://www.zara.com"}
        ],
        "isCelebrity": True
    },
    {
        "title": "Dress Like Travis Scott",
        "category": "celebrity",
        "image": "https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=400&h=600&fit=crop",
        "priceRange": "$400 - $800",
        "description": "Grunge meets street - Cactus Jack vibes",
        "celebrity": "Travis Scott",
        "items": [
            {"name": "Vintage Band Tee", "price": "$85", "link": "https://www.grailed.com"},
            {"name": "Cactus Jack Hoodie", "price": "$200", "link": "https://shop.travisscott.com"}
        ],
        "budgetAlternatives": [
            {"name": "Urban Outfitters Graphic Tee", "price": "$30", "link": "https://www.urbanoutfitters.com"},
            {"name": "H&M Oversized Hoodie", "price": "$40", "link": "https://www2.hm.com"}
        ],
        "isCelebrity": True
    }
]

# Insert outfits
result = outfits_collection.insert_many(demo_outfits)
print(f"✅ Inserted {len(result.inserted_ids)} demo outfits with WORKING IMAGES")

# Show counts
categories = ['streetwear', 'luxury', 'minimal', 'sport', 'elegant', 'celebrity']
for cat in categories:
    count = outfits_collection.count_documents({"category": cat})
    print(f"  {cat.capitalize()}: {count} outfits")

celebrity_count = outfits_collection.count_documents({"isCelebrity": True})
print(f"\n🌟 Celebrity outfits: {celebrity_count}")
print("\n✅ All outfits loaded with working Unsplash images!")
