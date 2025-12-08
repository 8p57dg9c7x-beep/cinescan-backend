#!/usr/bin/env python3
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URL = os.getenv('MONGO_URL', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URL)
db = client['app_database']

# Update all existing outfits to have isCelebrity: false
result = db['outfits'].update_many(
    {'isCelebrity': {'$exists': False}},
    {'$set': {'isCelebrity': False}}
)
print(f'✓ Updated {result.modified_count} outfits with isCelebrity: False')

# Add some celebrity outfits
celebrity_outfits = [
    {
        "title": "Zendaya's Red Carpet Elegance",
        "category": "celebrity",
        "image": "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=800&q=80",
        "description": "Zendaya's iconic red carpet look",
        "priceRange": "$$$$",
        "gender": "Women",
        "isCelebrity": True,
        "celebrityName": "Zendaya",
        "items": [
            {"name": "Designer Gown", "price": "$5000"},
            {"name": "Statement Heels", "price": "$800"}
        ],
        "budgetAlternatives": []
    },
    {
        "title": "Timothée Chalamet's Street Style",
        "category": "celebrity",
        "image": "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=800&q=80",
        "description": "Timothée's effortless street style",
        "priceRange": "$$$",
        "gender": "Men",
        "isCelebrity": True,
        "celebrityName": "Timothée Chalamet",
        "items": [
            {"name": "Designer Jacket", "price": "$1200"},
            {"name": "Slim Jeans", "price": "$300"}
        ],
        "budgetAlternatives": []
    },
    {
        "title": "Rihanna's Bold Fashion Statement",
        "category": "celebrity",
        "image": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800&q=80",
        "description": "Rihanna's daring fashion choice",
        "priceRange": "$$$$",
        "gender": "Women",
        "isCelebrity": True,
        "celebrityName": "Rihanna",
        "items": [
            {"name": "Statement Outfit", "price": "$3000"},
            {"name": "Luxury Accessories", "price": "$1500"}
        ],
        "budgetAlternatives": []
    }
]

result = db['outfits'].insert_many(celebrity_outfits)
print(f'✓ Added {len(celebrity_outfits)} celebrity outfits')

print('\n📊 Updated counts:')
for cat in ['streetwear', 'luxury', 'minimal', 'celebrity']:
    count = db['outfits'].count_documents({'category': cat})
    print(f'  {cat}: {count} outfits')

# Test trending query
trending = list(db['outfits'].aggregate([
    {"$match": {"isCelebrity": False}},
    {"$sample": {"size": 5}}
]))
print(f'\n✓ Trending query returns {len(trending)} outfits')
