# config.py
# Configuration settings for the luxury live session optimizer project

# Success score weights (adjusted for luxury market)
# w1: Weight for conversion rate (purchase_comments / total_comments)
# w2: Weight for sales per viewer (sales_amount / views)
# w3: Weight for engagement per viewer (total_comments + reactions + shares / views)
SUCCESS_WEIGHTS = {
    'w1': 0.4,  # Prioritize conversion rate moderately
    'w2': 0.5,  # Prioritize sales per viewer heavily (luxury market focuses on high AOV)
    'w3': 0.1   # Engagement is less critical but still relevant
}

# Luxury brand tiers for categorizing brands
# Ultra-luxury: High-end, exclusive brands with the highest price points
# Mid-tier luxury: Well-known luxury brands, more accessible
# Premium: High-quality but more accessible brands
LUXURY_TIERS = {
    'ultra_luxury': [
        'CHANEL', 'LOUIS VUITTON', 'GUCCI', 'PRADA', 'HERMES', 'DIOR', 'BALENCIAGA',
        'GIVENCHY', 'YSL', 'BOTTEGA', 'CELINE', 'ALEXANDER MC QUEEN', 'FERRAGAMO'
    ],
    'mid_tier_luxury': [
        'COACH', 'MICHAEL KORS', 'KATE SPADE', 'MARC JACOBS', 'TORY BURCH', 'FURLA',
        'LONGCHAMP', 'BURBERRY', 'MOSCHINO', 'LOVE MOSCHINO', 'VERSACE', 'TRUSSARDI',
        'COCCINELLE', 'PINKO', 'CHIARA FERRAGNI', 'MCM', 'VIVIENNE WESTWOOD', 'KENZO',
        'RADLEY LONDON', 'TED BAKER', 'DIESEL', 'SUPERDRY', 'EVISU', 'MANDARINA DUCK',
        'LIU JO', 'TWINSET', 'GAELLE', 'MARIO VALENTINO', 'JOHN RICHMOND'
    ],
    'premium': [
        'CALVIN KLEIN', 'DKNY', 'TOMMY HILFIGER', 'TRUE RELIGION', 'ARMANI EXCHANGE',
        'GUESS', 'KARL LAGERFELD', 'KIPLING', 'FOSSIL', 'SWAROVSKI', 'THE NORTH FACE',
        'ADIDAS', 'PUMA', 'FILA', 'NEW ERA', 'CHAMPION', 'ABERCROMBIE & FITCH',
        'LULULEMON', 'STEVE MADDEN', 'COLE HAAN', 'ASICS', 'NEW BALANCE', 'ECCO',
        'REEBOK', 'HURLEY', 'FOREVER21', 'SAMSONITE', 'HERSCHEL', 'VANS', 'UNDEFEATED',
        'AMBLER', 'OBAG', 'LOCKNLOCK', 'LUCKY BRAND', 'NOOOF'
    ]
}

# Region-specific holidays for capturing increased viewership or purchasing behavior
HOLIDAYS = {
    'sg': ['2025-01-29', '2025-01-30', '2025-08-09'],  # Chinese New Year, National Day
    'us': ['2025-11-27', '2025-11-28'],  # Thanksgiving, Black Friday
    'kr': ['2025-09-16', '2025-01-29']   # Chuseok, Lunar New Year
}

# Minimum success score threshold for suggestions
SUCCESS_SCORE_MINIMAL = 50