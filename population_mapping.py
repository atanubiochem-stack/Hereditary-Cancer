"""
GenomeIndia Population Code to Full Ethnic Name Mapping Module
Decodes standardized GenomeIndia 4-tier community designations:
<EthnoLinguisticFamily>_<GeographicRegion>_<SocialCategory>_<GroupIndex>
"""

FAMILY_MAP = {
    'IE': 'Indo-European',
    'DR': 'Dravidian',
    'TB': 'Tibeto-Burman',
    'AA': 'Austroasiatic',
    'AN': 'Andamanese'
}

REGION_MAP = {
    'NGH': 'Nilgiri Hills',
    'WGH': 'Western Ghats',
    'EGH': 'Eastern Ghats',
    'CHR': 'Central Highlands',
    'NCH': 'North Central Highlands',
    'SCH': 'South Central Highlands',
    'BPV': 'Brahmaputra Valley',
    'NER': 'North Eastern Range',
    'WHR': 'Western Himalayan Range',
    'EHR': 'Eastern Himalayan Range',
    'WCP': 'Western Coastal Plains',
    'ECP': 'Eastern Coastal Plains',
    'WPL': 'Western Plains',
    'EPL': 'Eastern Plains',
    'NRP': 'Northern River Plains',
    'ERP': 'Eastern River Plains',
    'NDN': 'Northern Deccan',
    'SDN': 'Southern Deccan'
}

CATEGORY_MAP = {
    '1': 'Tribal',
    '2': 'Non-Tribal'
}

def get_full_ethnic_name(code, style='descriptive'):
    if not code or pd_isna(code) or str(code).lower() in ['nan', 'none', '.']:
        return 'Not Specified'
    parts = str(code).split('_')
    if len(parts) >= 4:
        fam = FAMILY_MAP.get(parts[0], parts[0])
        reg = REGION_MAP.get(parts[1], parts[1])
        cat = CATEGORY_MAP.get(parts[2], 'Community')
        idx = parts[3]
        if style == 'descriptive':
            return f"{fam} {cat} Group, {reg} (Community {idx})"
        elif style == 'plot':
            return f"{fam} {cat}, {reg} {idx}"
        elif style == 'combined':
            return f"{fam} {cat} Group, {reg} ({code})"
        elif style == 'short':
            return f"{fam} {cat}, {reg}"
    return str(code)

def pd_isna(val):
    return val is None or val != val
