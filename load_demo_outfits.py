#!/usr/bin/env python3
"""Load demo outfits into MongoDB"""

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

# Demo Outfits with Budget Alternatives
demo_outfits = [
    # Streetwear Category
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
        "image": "https://i.pinimg.com/736x/2e/5d/89/2e5d897f8e1e0a8f9e1e0a8f9e1e0a8f.jpg",
        "priceRange": "$180 - $320",
        "description": "Clean and simple street aesthetic",
        "items": [
            {"name": "White Basic Tee", "price": "$45", "link": "https://www.ssense.com"},
            {"name": "Black Slim Jeans", "price": "$135", "link": "https://www.levi.com"},
            {"name": "Converse Chuck Taylor", "price": "$70", "link": "https://www.converse.com"}
        ],
        "budgetAlternatives": [
            {"name": "Uniqlo White Tee Pack", "price": "$15", "link": "https://www.uniqlo.com"},
            {"name": "H&M Slim Jeans", "price": "$40", "link": "https://www2.hm.com"},
            {"name": "Vans Authentic", "price": "$50", "link": "https://www.vans.com"}
        ],
        "isCelebrity": False
    },
    
    # Luxury Category
    {
        "title": "Designer Casual Luxury",
        "category": "luxury",
        "image": "https://i.pinimg.com/736x/a3/72/c4/a372c4e8f9e1e0a8f9e1e0a8f9e1e0a8.jpg",
        "priceRange": "$800 - $1500",
        "description": "Elevated basics with designer touches",
        "items": [
            {"name": "Gucci Logo Sweatshirt", "price": "$850", "link": "https://www.gucci.com"},
            {"name": "Saint Laurent Jeans", "price": "$590", "link": "https://www.ysl.com"},
            {"name": "Common Projects Sneakers", "price": "$425", "link": "https://www.mrporter.com"}
        ],
        "budgetAlternatives": [
            {"name": "Zara Designer-Inspired Sweatshirt", "price": "$60", "link": "https://www.zara.com"},
            {"name": "Mango Slim Fit Jeans", "price": "$80", "link": "https://shop.mango.com"},
            {"name": "Greats Royale Sneakers", "price": "$179", "link": "https://www.greats.com"}
        ],
        "isCelebrity": False
    },
    
    # Minimal Category
    {
        "title": "Pure Minimalist Look",
        "category": "minimal",
        "image": "https://i.pinimg.com/736x/6d/84/2f/6d842f7e8f9e1e0a8f9e1e0a8f9e1e0a.jpg",
        "priceRange": "$200 - $350",
        "description": "Less is more - clean lines and neutral tones",
        "items": [
            {"name": "Everlane Cotton Shirt", "price": "$55", "link": "https://www.everlane.com"},
            {"name": "COS Minimal Pants", "price": "$129", "link": "https://www.cosstores.com"},
            {"name": "White Leather Sneakers", "price": "$150", "link": "https://www.everlane.com"}
        ],
        "budgetAlternatives": [
            {"name": "Uniqlo Oxford Shirt", "price": "$30", "link": "https://www.uniqlo.com"},
            {"name": "H&M Relaxed Pants", "price": "$35", "link": "https://www2.hm.com"},
            {"name": "Target White Sneakers", "price": "$40", "link": "https://www.target.com"}
        ],
        "isCelebrity": False
    },
    
    # Sport Category
    {
        "title": "Athleisure Performance",
        "category": "sport",
        "image": "https://i.pinimg.com/736x/9c/a1/5e/9ca15e8f9e1e0a8f9e1e0a8f9e1e0a8f.jpg",
        "priceRange": "$180 - $300",
        "description": "Gym to street versatility",
        "items": [
            {"name": "Nike Tech Fleece Hoodie", "price": "$110", "link": "https://www.nike.com"},
            {"name": "Lululemon Joggers", "price": "$128", "link": "https://www.lululemon.com"},
            {"name": "Nike Air Max", "price": "$140", "link": "https://www.nike.com"}
        ],
        "budgetAlternatives": [
            {"name": "Old Navy Active Hoodie", "price": "$35", "link": "https://oldnavy.gap.com"},
            {"name": "Target All in Motion Joggers", "price": "$30", "link": "https://www.target.com"},
            {"name": "Reebok Classic", "price": "$75", "link": "https://www.reebok.com"}
        ],
        "isCelebrity": False
    },
    
    # Elegant Category
    {
        "title": "Business Casual Refined",
        "category": "elegant",
        "image": "https://i.pinimg.com/736x/f2/8d/91/f28d917e8f9e1e0a8f9e1e0a8f9e1e0a.jpg",
        "priceRange": "$350 - $650",
        "description": "Professional and polished",
        "items": [
            {"name": "Brooks Brothers Blazer", "price": "$398", "link": "https://www.brooksbrothers.com"},
            {"name": "Banana Republic Chinos", "price": "$98", "link": "https://bananarepublic.gap.com"},
            {"name": "Cole Haan Dress Shoes", "price": "$180", "link": "https://www.colehaan.com"}
        ],
        "budgetAlternatives": [
            {"name": "H&M Slim Fit Blazer", "price": "$80", "link": "https://www2.hm.com"},
            {"name": "Old Navy Chinos", "price": "$35", "link": "https://oldnavy.gap.com"},
            {"name": "Target Goodfellow Dress Shoes", "price": "$40", "link": "https://www.target.com"}
        ],
        "isCelebrity": False
    },
    
    # Celebrity Outfits - "Dress Like Your Icon"
    {
        "title": "Dress Like Drake",
        "category": "celebrity",
        "image": "https://i.pinimg.com/736x/c5/3a/87/c53a878f9e1e0a8f9e1e0a8f9e1e0a8f.jpg",
        "priceRange": "$600 - $1200",
        "description": "OVO aesthetic - luxury streetwear meets hip-hop",
        "celebrity": "Drake",
        "items": [
            {"name": "Canada Goose Parka", "price": "$995", "link": "https://www.canadagoose.com"},
            {"name": "Amiri Jeans", "price": "$890", "link": "https://www.amiri.com"},
            {"name": "Jordan 4 Retro", "price": "$225", "link": "https://www.nike.com"}
        ],
        "budgetAlternatives": [
            {"name": "Columbia Winter Jacket", "price": "$150", "link": "https://www.columbia.com"},
            {"name": "Zara Distressed Jeans", "price": "$60", "link": "https://www.zara.com"},
            {"name": "Nike Air Jordan 1", "price": "$170", "link": "https://www.nike.com"}
        ],
        "isCelebrity": True
    },
    {
        "title": "Dress Like Zendaya",
        "category": "celebrity",
        "image": "https://i.pinimg.com/736x/7b/1c/4f/7b1c4f8f9e1e0a8f9e1e0a8f9e1e0a8f.jpg",
        "priceRange": "$500 - $900",
        "description": "Effortlessly chic and fashion-forward",
        "celebrity": "Zendaya",
        "items": [
            {"name": "Valentino Blazer", "price": "$2890", "link": "https://www.valentino.com"},
            {"name": "Isabel Marant Pants", "price": "$495", "link": "https://www.isabelmarant.com"},
            {"name": "Jimmy Choo Heels", "price": "$795", "link": "https://www.jimmychoo.com"}
        ],
        "budgetAlternatives": [
            {"name": "Mango Tailored Blazer", "price": "$120", "link": "https://shop.mango.com"},
            {"name": "Zara Wide Leg Pants", "price": "$50", "link": "https://www.zara.com"},
            {"name": "Steve Madden Heels", "price": "$90", "link": "https://www.stevemadden.com"}
        ],
        "isCelebrity": True
    },
    {
        "title": "Dress Like Travis Scott",
        "category": "celebrity",
        "image": "https://i.pinimg.com/736x/d9/62/a3/d962a38f9e1e0a8f9e1e0a8f9e1e0a8f.jpg",
        "priceRange": "$400 - $800",
        "description": "Grunge meets street - Cactus Jack vibes",
        "celebrity": "Travis Scott",
        "items": [
            {"name": "Vintage Band Tee", "price": "$85", "link": "https://www.grailed.com"},
            {"name": "Cactus Jack Hoodie", "price": "$200", "link": "https://shop.travisscott.com"},
            {"name": "Nike SB Dunk", "price": "$180", "link": "https://www.nike.com"}
        ],
        "budgetAlternatives": [
            {"name": "Urban Outfitters Graphic Tee", "price": "$30", "link": "https://www.urbanoutfitters.com"},
            {"name": "H&M Oversized Hoodie", "price": "$40", "link": "https://www2.hm.com"},
            {"name": "Vans Sk8-Hi", "price": "$70", "link": "https://www.vans.com"}
        ],
        "isCelebrity": True
    }
]

# Insert outfits
result = outfits_collection.insert_many(demo_outfits)
print(f"✅ Inserted {len(result.inserted_ids)} demo outfits")

# Show counts by category
categories = ['streetwear', 'luxury', 'minimal', 'sport', 'elegant', 'celebrity']
for cat in categories:
    count = outfits_collection.count_documents({"category": cat})
    print(f"  {cat.capitalize()}: {count} outfits")

# Show celebrity outfits
celebrity_count = outfits_collection.count_documents({"isCelebrity": True})
print(f"\n🌟 Celebrity outfits (Dress Like Your Icon): {celebrity_count}")

print("\n✅ Demo outfits loaded successfully!")
