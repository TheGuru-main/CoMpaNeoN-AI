"""
Domain knowledge for common categories.
Used by intent analyzer, symbols, and prompt enrichment.
Production-ready with multi-country support.
"""

COUNTRIES = {
    "NG": "Nigeria",
    "GH": "Ghana",
    "US": "United States",
    "IN": "India",
    "GB": "United Kingdom",
    "FR": "France",
    "DE": "Germany",
    "SA": "Saudi Arabia",
    "AE": "United Arab Emirates",
    "ZA": "South Africa",
    "KE": "Kenya",
    "TZ": "Tanzania",
    "UG": "Uganda",
    "EG": "Egypt",
    "MA": "Morocco",
    "DZ": "Algeria",
    "SN": "Senegal",
    "CI": "Ivory Coast",
    "CM": "Cameroon",
    "ET": "Ethiopia",
    "RW": "Rwanda",
    "SD": "Sudan",
    "SS": "South Sudan",
    "AO": "Angola",
    "MZ": "Mozambique",
    "ZM": "Zambia",
    "ZW": "Zimbabwe",
    "BW": "Botswana",
    "MW": "Malawi",
    "NA": "Namibia",
    "LS": "Lesotho",
    "SZ": "Eswatini",
    "CD": "Democratic Republic of the Congo",
    "CG": "Republic of the Congo",
    "GA": "Gabon",
    "BJ": "Benin",
    "TG": "Togo",
    "BF": "Burkina Faso",
    "ML": "Mali",
    "NE": "Niger",
    "TD": "Chad",
    "MR": "Mauritania",
    "LY": "Libya",
    "TN": "Tunisia",
    "ER": "Eritrea",
    "DJ": "Djibouti",
    "SO": "Somalia",
    "KM": "Comoros",
    "MG": "Madagascar",
    "SC": "Seychelles",
    "MU": "Mauritius",
    "CV": "Cape Verde",
    "ST": "São Tomé and Príncipe",
    "GQ": "Equatorial Guinea",
    "GW": "Guinea-Bissau",
    "GN": "Guinea",
    "LR": "Liberia",
    "SL": "Sierra Leone",
    "GM": "Gambia",
}

# States / regions for major African countries (sample, expand as needed)
STATES_BY_COUNTRY = {
    "NG": [
        "Lagos", "Abuja", "Rivers", "Kano", "Oyo", "Ogun", "Enugu",
        "Anambra", "Kaduna", "Delta", "Edo", "Benue", "Kwara", "Ondo",
        "Ekiti", "Osun", "Bauchi", "Gombe", "Jigawa", "Katsina",
        "Kebbi", "Niger", "Plateau", "Sokoto", "Taraba", "Yobe", "Zamfara",
    ],
    "GH": [
        "Greater Accra", "Ashanti", "Western", "Eastern", "Central",
        "Volta", "Northern", "Upper East", "Upper West", "Bono",
        "Bono East", "Ahafo", "Savannah", "North East", "Oti",
    ],
    "KE": [
        "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Uasin Gishu",
        "Kiambu", "Kajiado", "Machakos", "Meru", "Nyeri",
    ],
    "TZ": [
        "Dar es Salaam", "Dodoma", "Arusha", "Mwanza", "Mbeya",
        "Morogoro", "Tanga", "Kigoma", "Tabora", "Zanzibar Urban/West",
    ],
    "UG": [
        "Kampala", "Wakiso", "Mbarara", "Gulu", "Jinja",
        "Mbale", "Masaka", "Fort Portal", "Arua", "Lira",
    ],
    "ZA": [
        "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
        "Free State", "Limpopo", "Mpumalanga", "North West",
        "Northern Cape",
    ],
    "EG": [
        "Cairo", "Alexandria", "Giza", "Luxor", "Aswan",
        "Asyut", "Beheira", "Dakahlia", "Gharbia", "Minya",
    ],
    "MA": [
        "Casablanca-Settat", "Rabat-Salé-Kénitra", "Marrakech-Safi",
        "Fès-Meknès", "Tangier-Tétouan-Al Hoceïma", "Oriental",
        "Béni Mellal-Khénifra", "Drâa-Tafilalet", "Souss-Massa",
        "Guelmim-Oued Noun",
    ],
    "DZ": [
        "Algiers", "Oran", "Constantine", "Annaba", "Blida",
        "Batna", "Sétif", "Tlemcen", "Béjaïa", "Tizi Ouzou",
    ],
    "SN": [
        "Dakar", "Thiès", "Saint-Louis", "Ziguinchor", "Kaolack",
        "Louga", "Tambacounda", "Kolda", "Matam", "Fatick",
    ],
    "CI": [
        "Abidjan", "Yamoussoukro", "Bouaké", "San-Pédro", "Daloa",
        "Korhogo", "Man", "Gagnoa", "Divo", "Abengourou",
    ],
    "CM": [
        "Centre", "Littoral", "West", "North", "Far North",
        "East", "South", "Adamawa", "Northwest", "Southwest",
    ],
    "ET": [
        "Addis Ababa", "Oromia", "Amhara", "Tigray", "Somali",
        "Afar", "Benishangul-Gumuz", "Gambela", "Harari", "SNNPR",
    ],
    "RW": [
        "Kigali", "Eastern", "Western", "Northern", "Southern",
    ],
    "SD": [
        "Khartoum", "Omdurman", "North Kordofan", "South Kordofan",
        "Darfur", "Red Sea", "River Nile", "Gezira", "Kassala",
    ],
    "SS": [
        "Central Equatoria", "Eastern Equatoria", "Western Equatoria",
        "Jonglei", "Upper Nile", "Unity", "Lakes", "Warrap",
        "Northern Bahr el Ghazal", "Western Bahr el Ghazal",
    ],
    "AO": [
        "Luanda", "Benguela", "Huambo", "Lubango", "Malanje",
        "Cabinda", "Uíge", "Namibe", "Cunene", "Moxico",
    ],
    "MZ": [
        "Maputo", "Sofala", "Zambezia", "Nampula", "Tete",
        "Manica", "Gaza", "Inhambane", "Cabo Delgado", "Niassa",
    ],
    "ZM": [
        "Lusaka", "Copperbelt", "Central", "Eastern", "Luapula",
        "Muchinga", "Northern", "North-Western", "Southern", "Western",
    ],
    "ZW": [
        "Harare", "Bulawayo", "Manicaland", "Mashonaland Central",
        "Mashonaland East", "Mashonaland West", "Masvingo",
        "Matabeleland North", "Matabeleland South", "Midlands",
    ],
}

# People names (multi-cultural, not exhaustive)
PEOPLE_NAMES = [
    "Idris Akeem", "Chinedu Okafor", "Amina Bello", "John Smith",
    "Fatima Yusuf", "Emeka Obi", "Grace Adeyemi", "Musa Ibrahim",
    "Ngozi Eze", "David Johnson", "Maryam Abubakar", "Tunde Bakare",
    "Kwame Mensah", "Ama Serwaa", "Jean-Pierre Dubois", "Sophie Martin",
    "Ahmed Hassan", "Layla Ali", "Raj Patel", "Priya Sharma",
    "Li Wei", "Zhang Min", "Carlos Mendoza", "Isabella García",
    "Omar Farouk", "Nadia Khaled", "Thabo Mbeki", "Lindiwe Zulu",
    "Samuel Okonkwo", "Esther Adeleke",
]

ANIMALS = [
    "dog", "cat", "lion", "tiger", "elephant", "goat", "sheep",
    "cow", "horse", "chicken", "fish", "snake", "monkey",
    "eagle", "parrot", "camel", "rabbit", "crocodile", "hippopotamus",
    "giraffe", "zebra", "antelope", "leopard", "cheetah", "hyena",
    "jackal", "warthog", "buffalo", "rhinoceros", "gorilla",
    "chimpanzee", "baboon", "ostrich", "flamingo", "pelican",
    "duck", "goose", "turkey", "pig", "donkey", "mule",
]

THINGS = [
    "phone", "computer", "car", "bicycle", "television", "radio",
    "chair", "table", "book", "pen", "shoe", "shirt", "laptop",
    "camera", "refrigerator", "air conditioner", "fan", "watch",
    "microwave", "blender", "kettle", "iron", "washing machine",
    "stove", "oven", "toaster", "vacuum cleaner", "hair dryer",
    "electric fan", "generator", "solar panel", "battery",
    "light bulb", "door", "window", "mirror", "clock", "calendar",
]

def get_domain_entities(domain: str) -> list:
    """Return a list of entities for a given domain."""
    domain_lower = domain.lower()
    if domain_lower == "country":
        return list(COUNTRIES.values())
    elif domain_lower == "state":
        # Flatten all states from all countries
        states = []
        for country_states in STATES_BY_COUNTRY.values():
            states.extend(country_states)
        return states
    elif domain_lower == "people":
        return PEOPLE_NAMES
    elif domain_lower == "animal":
        return ANIMALS
    elif domain_lower == "thing":
        return THINGS
    return []