MAIN_GENRES = {
    "pop", "rock", "hiphop", "rnb", "electronic",
    "jazz", "metal", "classical", "blues",
    "country", "folk", "reggae", "world", "soundtrack",
    "alternative", "punk", "industrial", "lounge"  # Only added these 4
}

SUBGENRE_MAP = {
    # ===== Your Original Mappings (Preserved) =====
    # Hip-Hop/Rap
    "rap": "hiphop",
    "trap": "hiphop",
    "triphop": "hiphop",
    "breakbeat": "hiphop",

    # R&B/Soul
    "funk": "rnb",
    "soul": "rnb",
    "jazzfunk": "rnb",
    "acidjazz": "jazz",

    # Electronic
    "house": "electronic",
    "techno": "electronic",
    "trance": "electronic",
    "dubstep": "electronic",
    "ambient": "electronic",
    "drumnbass": "electronic",
    "idm": "electronic",
    "synthwave": "electronic",
    "chillout": "electronic",
    "experimental": "electronic",
    "lofi": "electronic",
    "technoindustrial": "industrial",  # Mapped to new main genre
    "darkambient": "electronic",
    "electronica": "electronic",
    "deephouse": "electronic",
    "downtempo": "electronic",
    "eurodance": "electronic",
    "minimal": "electronic",
    "minimalism": "electronic",
    "noise": "electronic",
    "dance": "electronic",
    "atmospheric": "electronic",
    "ambient": "electronic",
    
    # Pop
    "easylistening": "pop", 
    "electropop": "pop",
    "synthpop": "pop",
    "poprock": "pop",
    "powerpop": "pop",
    "instrumentalpop": "pop",
    "disco": "pop",

    # Rock
    "alternativerock": "alternative",  # Mapped to new main genre
    "postrock": "rock",
    "progressive": "rock",
    "progressiverock": "rock",
    "hardrock": "rock",
    "classicrock": "rock",
    "rocknroll": "rock",
    "garage": "rock",
    "instrumentalrock": "rock",

    # New: Punk
    "punkrock": "punk",
    "hardcore": "punk",
    "popunk": "punk",

    # New: Alternative/Indie
    "indie": "alternative",
    "indierock": "alternative",
    "grunge": "alternative",
    "indiepop": "alternative",
    "newwave": "alternative",
    "darkwave": "alternative",
    "psychedelic": "alternative",
    "gothic": "alternative",

    # Jazz
    "jazzfusion": "jazz",

    # Classical/Orchestral
    "neoclassical": "classical",
    "baroque": "classical",
    "orchestral": "classical",
    "symphonic": "classical",

    # World/Ethnic
    "african": "world",
    "latin": "world",
    "ethno": "world",
    "ethnicrock": "world",
    "celtic": "world",
    "asian": "world",
    "flamenco": "world",
    "chanson": "world",
    "chansonfrancaise": "world",

    # Metal
    "heavymetal": "metal",
    "deathmetal": "metal", 
    "blackmetal": "metal",
    "metalcore": "metal",
    "hardcore": "metal",

    # Other
    "newage": "newage",
    "industrial": "industrial",
    "lounge": "lounge",
    "singersongwriter": "folk",
    "popfolk": "folk",
    "ska": "reggae",
    "reggaeton": "reggae",
    "avantgarde": "experimental",
    "soundtrack": "soundtrack",

    # Era-based
    "60s": "rock",
    "70s": "rock", 
    "80s": "pop",
    "90s": "alternative",


}