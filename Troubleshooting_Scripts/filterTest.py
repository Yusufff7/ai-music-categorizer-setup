import pandas as pd
from utils import infer_main_genres, MAIN_GENRES, SUBGENRE_MAP  

def run_test_case(input_genres, case_name):
    """Tests your EXACT infer_main_genres() function with visualization"""
    print(f"\n{'='*60}\nTEST CASE: {case_name}\n{'='*60}")
    print(f"Input: {input_genres}")
    print(f"MAIN_GENRES: {MAIN_GENRES}")
    print(f"SUBGENRE_MAP: {SUBGENRE_MAP}")
    
    # Create a mock DataFrame row
    row = pd.Series({
        'genres': input_genres,
        'TAGS': [f"genre---{g}" for g in input_genres] if input_genres else []
    })
    
    # Run your ACTUAL function
    main_genres, unmapped = infer_main_genres(row)
    is_valid = len(main_genres) > 0
    
    # Visualize results
    print("\nPROCESSING STEPS:")
    print(f"1. Raw genres       → {input_genres}")
    print(f"2. Mapped genres    → {main_genres} ({'VALID' if is_valid else 'INVALID'})")
    print(f"3. Unmapped genres  → {unmapped}")
    print(f"4. Final decision   → {'KEEP TRACK' if is_valid else 'DISCARD TRACK'}")
    
    if is_valid:
        print(f"   Final genres: {main_genres}")
    
    print("\nDEBUG BREAKDOWN:")
    for genre in input_genres:
        status = (
            "MAIN GENRE" if genre in MAIN_GENRES else
            f"MAPPED TO: {SUBGENRE_MAP.get(genre)}" if genre in SUBGENRE_MAP else
            "UNMAPPED"
        )
        print(f"- {genre:15} → {status}")

# ===== Test Scenarios =====
test_cases = [
    (["rock", "pop"], "Multiple main genres"),
    (["punkrock"], "Mappable subgenre"),
    (["rock", "invalid_subgenre"], "Valid + invalid mix"),
    (["unknown_genre"], "Completely unmapped genre"),
    ([], "Empty genre list"),
    (["techno", "dubstep", "pop"], "Mixed electronic and pop"),
    (["invalid1", "invalid2"], "All invalid genres"),
    (["rock", "rock"], "Duplicate genres")
]

# Execute tests
print(f"\n{'#'*30} STARTING TESTS {'#'*30}\n")
for genres, description in test_cases:
    run_test_case(genres, description)

print(f"\n{'#'*30} TEST SUMMARY {'#'*30}")
print(f"Tested {len(test_cases)} scenarios with your ACTUAL functions")
print("Key behaviors verified:")
print("- Uses your exact MAIN_GENRES and SUBGENRE_MAP")
print("- Preserves your original logic unmodified")
print("- Shows step-by-step genre processing")