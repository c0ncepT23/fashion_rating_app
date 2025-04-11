# app/utils/constants.py
"""
Constants for fashion rating application including style mappings,
feedback templates, and other fixed values
"""

# Style definitions
STYLE_CATEGORIES = {
    "casual": "Relaxed, everyday wear focused on comfort",
    "formal": "Elegant, structured attire for professional or dressy occasions",
    "streetwear": "Urban-inspired, trendy style with casual elements",
    "bohemian": "Free-spirited style with artistic, earthy elements",
    "vintage": "Styles inspired by past decades and retro aesthetics",
    "minimalist": "Clean, simple designs with few embellishments",
    "sporty": "Athletic-inspired pieces with functional elements",
    "edgy_chic": "Bold, fashion-forward style with unconventional elements",
    "preppy": "Clean-cut, collegiate-inspired classic style",
    "romantic": "Feminine, soft styles with delicate details",
    "fashion_week_off_duty": "Effortless yet deliberate styling seen on fashion industry insiders",
    "avant_garde": "Experimental, boundary-pushing fashion",
    "business_casual": "Professional yet relaxed office-appropriate attire"
}

# Clothing categories
CLOTHING_CATEGORIES = {
    "top": ["short sleeve top", "long sleeve top", "shirt", "tank_top", "hoodie"],
    "bottom": ["shorts", "trousers", "skirt", "leggings", "jeans"],
    "dress": ["short sleeve dress", "long sleeve dress", "vest dress", "sling dress", "gown"],
    "outerwear": ["short sleeve outwear", "long sleeve outwear", "vest", "jacket", "blazer", "cardigan"],
    "footwear": ["sneakers", "boots", "sandals", "heels", "loafers", "flats", "oxfords"],
    "accessory": ["hat", "bag", "scarf", "belt", "jewelry", "sunglasses", "watch"]
}

# Color harmony types
COLOR_HARMONY_TYPES = {
    "monochromatic": "Different shades of a single color",
    "complementary": "Colors opposite each other on the color wheel",
    "analogous": "Colors adjacent to each other on the color wheel",
    "triadic": "Three colors equally spaced on the color wheel",
    "split_complementary": "One color plus two adjacent to its complement",
    "tetradic": "Four colors arranged in two complementary pairs",
    "neutral": "Primarily neutral colors with minimal saturation",
    "bright": "High value colors that create a vibrant palette",
    "dark": "Low value colors that create a deep, rich palette",
    "balanced": "Mix of colors with balanced characteristics"
}

# Seasonal color palettes
SEASONAL_PALETTES = {
    "spring": ["coral", "mint", "yellow", "light_blue", "peach"],
    "summer": ["pastel_blue", "lavender", "soft_pink", "mint", "light_gray"],
    "autumn": ["olive", "rust", "mustard", "burgundy", "terracotta"],
    "winter": ["black", "white", "navy", "red", "emerald"]
}

# ====== FEEDBACK DETAIL OPTIONS ======

# Fit feedback details
FIT_DETAILS = {
    "excellent": [
        "creating a balanced silhouette with strong shoulders and a leggy look",
        "the proportions are perfectly balanced for your body type",
        "the combination creates visual interest while maintaining harmony",
        "the structured elements balance nicely with the more relaxed pieces",
        "the varying lengths create a sophisticated layered effect"
    ],
    "good": [
        "creating a nice balance between the upper and lower body",
        "the proportions work well for your frame",
        "the silhouette shows good understanding of shape and balance",
        "the combination of fitted and loose elements adds visual interest",
        "the layering creates depth without overwhelming the look"
    ],
    "average": [
        "adjusting the proportions to create more balance",
        "selecting pieces that enhance your natural body shape",
        "hemming or tailoring for a more precise fit",
        "considering the balance between loose and fitted elements",
        "paying attention to where pieces hit on your body"
    ],
    "poor": [
        "choosing pieces that better complement your body type",
        "adjusting the proportions to create more harmony",
        "reconsidering the balance between loose and fitted elements",
        "selecting items that create a more cohesive silhouette",
        "opting for better tailoring or fit in key pieces"
    ]
}

# Color feedback details
COLOR_DETAILS = {
    "excellent": [
        "demonstrating confident styling and creating visual impact",
        "showing sophisticated understanding of color theory",
        "the combination creates depth and visual interest",
        "the palette feels intentional and cohesive",
        "the color balance creates a harmonious overall look"
    ],
    "good": [
        "creating a pleasing visual harmony",
        "showing good understanding of coordinating colors",
        "the palette works well together throughout the outfit",
        "maintaining consistency in color temperature",
        "using color effectively to highlight key pieces"
    ],
    "average": [
        "adding a complementary accent color for more depth",
        "considering a more cohesive color strategy",
        "balancing the warm and cool tones more evenly",
        "using color more intentionally to create focus",
        "reducing the number of competing colors"
    ],
    "poor": [
        "focusing on a more harmonious color scheme",
        "reducing the number of competing colors",
        "choosing a dominant color with complementary accents",
        "considering color temperature and how colors interact",
        "selecting a more cohesive palette with better balance"
    ]
}

# Footwear feedback details
FOOTWEAR_DETAILS = {
    "excellent": [
        "they elongate the leg line while maintaining the outfit's style",
        "they coordinate perfectly with the color story while adding visual interest",
        "their proportions complement the rest of the outfit beautifully",
        "they add sophistication while maintaining comfort and practicality",
        "they serve as the perfect punctuation mark to complete the look"
    ],
    "good": [
        "they complement the outfit's style while providing proper proportion",
        "they coordinate with the color palette effectively",
        "they balance comfort with style appropriately",
        "they maintain the overall aesthetic of the outfit",
        "they support the visual flow from top to bottom"
    ],
    "average": [
        "choosing a style that better complements the outfit's aesthetic",
        "selecting a color that ties in more with the overall palette",
        "adjusting the proportion to better balance the outfit",
        "finding footwear that enhances rather than competes with the look",
        "selecting a more current style that still works with your outfit"
    ],
    "poor": [
        "selecting footwear that better aligns with the outfit's style",
        "choosing a color that coordinates with the overall palette",
        "finding a shape or style that better balances the proportions",
        "selecting footwear that enhances rather than detracts from the look",
        "considering the visual weight and how it impacts the overall silhouette"
    ]
}

# Accessories feedback details
ACCESSORIES_DETAILS = {
    "excellent": [
        "they add personality while maintaining the outfit's cohesion",
        "they create strategic points of interest without overwhelming",
        "they complement the outfit's style while adding dimension",
        "they demonstrate thoughtful curation and attention to detail",
        "they enhance the look without competing with key elements"
    ],
    "good": [
        "they complement the outfit without overwhelming it",
        "they add personal flair while maintaining cohesion",
        "they enhance the overall look with thoughtful details",
        "they create interest while staying true to the style",
        "they show good understanding of balance and proportion"
    ],
    "average": [
        "being more strategic with placement and visual weight",
        "selecting pieces that tie more directly to the color palette",
        "considering scale and proportion more carefully",
        "adding one statement piece rather than several competing elements",
        "choosing accessories that reinforce the outfit's style direction"
    ],
    "poor": [
        "simplifying to create more focus and cohesion",
        "selecting pieces that better complement the outfit's style",
        "choosing accessories with more intentional color coordination",
        "considering scale and proportion more carefully",
        "selecting fewer, more impactful pieces"
    ],
    "none": [
        "a statement necklace or earrings to add visual interest",
        "a belt to define the waistline and add structure",
        "a coordinating bag to add functionality and style",
        "simple jewelry to add a personal touch",
        "a scarf or hat to add dimension and personality"
    ]
}

# Style feedback details by style category
STYLE_DETAILS = {
    "casual": {
        "excellent": [
            "the relaxed elements blend with put-together pieces for an effortless look",
            "the outfit shows sophistication while maintaining everyday wearability",
            "there's thoughtful coordination without appearing too studied"
        ],
        "good": [
            "the pieces work well together for a cohesive casual look",
            "comfort and style are well-balanced throughout",
            "the outfit maintains interest while staying appropriate for everyday wear"
        ],
        "average": [
            "adding a few more intentional elements to elevate the casual look",
            "considering how to make comfortable pieces look more put-together",
            "focusing on fit to make casual pieces appear more polished"
        ],
        "poor": [
            "focusing on pieces that work better together while staying casual",
            "paying more attention to fit and proportion for a more polished look",
            "selecting pieces that show more intentionality while remaining comfortable"
        ]
    },
    "edgy_chic": {
        "excellent": [
            "the outfit successfully combines formal and casual elements for a look that's both edgy and feminine",
            "the contrast between structured and unconventional pieces creates compelling visual tension",
            "the outfit demonstrates confident understanding of proportion and strategic color use"
        ],
        "good": [
            "the outfit effectively balances edgy elements with more refined pieces",
            "there's good tension between structured and unexpected components",
            "the styling shows personality while maintaining wearability"
        ],
        "average": [
            "adding more contrast between edgy and classic elements",
            "incorporating one unexpected piece to create more visual interest",
            "playing more with proportion to enhance the edgy aesthetic"
        ],
        "poor": [
            "focusing on balance between classic and unconventional elements",
            "considering how to incorporate edge in more cohesive ways",
            "selecting statement pieces that work better with the overall look"
        ]
    },
    "fashion_week_off_duty": {
        "excellent": [
            "effortlessly combines high-end pieces with casual elements for that perfect 'off duty' look",
            "shows a confident understanding of proportion and subtle statement-making",
            "demonstrates insider fashion knowledge through thoughtful, unexpected styling choices"
        ],
        "good": [
            "balances statement pieces with wearable elements effectively",
            "shows a good grasp of current trends while maintaining personal style",
            "creates interest through thoughtful combination of silhouettes"
        ],
        "average": [
            "focusing more on creating an effortless appearance despite the deliberate styling",
            "incorporating more unexpected element combinations for fashion-forward impact",
            "refining the balance between statement and supporting pieces"
        ],
        "poor": [
            "reconsidering the balance between statement pieces and basics",
            "focusing on creating a more cohesive overall vision",
            "selecting pieces that work together more harmoniously"
        ]
    },
    # Default style details for fallback
    "default": {
        "excellent": [
            "the outfit shows thoughtful coordination and attention to detail",
            "there's excellent balance between different elements",
            "the overall look is cohesive while showing personal style"
        ],
        "good": [
            "the outfit demonstrates good understanding of the style",
            "the pieces work well together to create a cohesive look",
            "there's good balance between statement and supporting elements"
        ],
        "average": [
            "focusing more on the core elements of this style",
            "considering how to make the outfit more cohesive",
            "selecting pieces that more clearly align with the style's aesthetic"
        ],
        "poor": [
            "revisiting the key elements that define this style",
            "focusing on creating more cohesion between pieces",
            "selecting pieces that better represent this style's aesthetic"
        ]
    }
}