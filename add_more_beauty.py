#!/usr/bin/env python3
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URL = os.getenv('MONGO_URL', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URL)
db = client['app_database']

# Add more beauty looks
new_looks = [
    {
        "title": "Bold Red Lip Statement",
        "category": "bold",
        "image": "https://images.unsplash.com/photo-1487412947147-5cebf100ffc2?w=800&q=80",
        "description": "Classic bold red lip makeup look",
        "products": [
            {"name": "Red Lipstick", "brand": "MAC", "shade": "Ruby Woo"},
            {"name": "Lip Liner", "brand": "MAC", "shade": "Cherry"}
        ]
    },
    {
        "title": "Dramatic Cat Eye",
        "category": "bold",
        "image": "https://images.unsplash.com/photo-1516975080664-ed2fc6a32937?w=800&q=80",
        "description": "Bold winged eyeliner look",
        "products": [
            {"name": "Liquid Eyeliner", "brand": "Stila", "shade": "Intense Black"},
            {"name": "Mascara", "brand": "Benefit", "shade": "They're Real"}
        ]
    },
    {
        "title": "Editorial High Fashion",
        "category": "editorial",
        "image": "https://images.unsplash.com/photo-1560869713-7d0a29430803?w=800&q=80",
        "description": "Avant-garde editorial makeup",
        "products": [
            {"name": "Artistic Palette", "brand": "Pat McGrath", "shade": "Mothership"},
            {"name": "Glitter", "brand": "MAC", "shade": "Reflects Pearl"}
        ]
    },
    {
        "title": "Runway Ready",
        "category": "editorial",
        "image": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=800&q=80",
        "description": "High fashion runway makeup",
        "products": [
            {"name": "Foundation", "brand": "Giorgio Armani", "shade": "Luminous Silk"},
            {"name": "Highlighter", "brand": "Becca", "shade": "Champagne Pop"}
        ]
    }
]

result = db['beauty_looks'].insert_many(new_looks)
print(f'✓ Added {len(new_looks)} new beauty looks')

print('\n📊 Updated beauty counts:')
for cat in ['natural', 'glam', 'smokey', 'bold', 'editorial']:
    count = db['beauty_looks'].count_documents({'category': cat})
    print(f'  {cat}: {count} looks')
