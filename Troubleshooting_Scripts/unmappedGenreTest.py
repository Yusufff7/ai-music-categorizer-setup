import os
import pandas as pd
from collections import Counter
from utils import MAIN_GENRES, SUBGENRE_MAP

def analyze_genre_stats(tsv_path):
    # Load the TSV file
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    
    # Initialize stats trackers
    results = {
        'no_genre': [],
        'unmapped_only': [],
        'unmapped_subgenres': Counter(),
        'total_tracks': 0,
        'main_genre_counts': Counter(),
        'genre_combinations': Counter(),
        'all_main_genres': set()
    }
    
    for line in lines:
        if len(line) < 5:
            continue  # Skip malformed lines
            
        results['total_tracks'] += 1
        metadata = line[:5]
        tags = line[5:]
        path = metadata[3]
        
        # Extract and map genre tags
        genre_tags = []
        main_genres = set()
        
        for tag in tags:
            if tag.startswith('genre---'):
                genre = tag.replace('genre---', '')
                genre_tags.append(genre)
                
                # Map to main genre
                if genre in MAIN_GENRES:
                    main_genres.add(genre)
                else:
                    mapped = SUBGENRE_MAP.get(genre, None)
                    if mapped in MAIN_GENRES:
                        main_genres.add(mapped)
        
        # Case 1: No genre tags at all
        if not genre_tags:
            results['no_genre'].append(path)
            continue
            
        # Case 2: Only unmapped genres (no valid main genres)
        if not main_genres:
            results['unmapped_only'].append(path)
            for g in genre_tags:
                if g not in MAIN_GENRES and g not in SUBGENRE_MAP:
                    results['unmapped_subgenres'][g] += 1
            continue
            
        # Track genre statistics
        results['all_main_genres'].update(main_genres)
        
        # Count individual genres
        for genre in main_genres:
            results['main_genre_counts'][genre] += 1
            
        # Count genre combinations (sorted to avoid dupes like rock+pop vs pop+rock)
        if len(main_genres) > 1:
            sorted_combo = tuple(sorted(main_genres))
            results['genre_combinations'][sorted_combo] += 1
    
    return results

def save_analysis_report(results, output_dir='data/distribution'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save genre statistics
    with open(os.path.join(output_dir, 'genre_stats.csv'), 'w') as f:
        f.write("genre,count,percentage\n")
        total_mapped = sum(results['main_genre_counts'].values())
        for genre, count in results['main_genre_counts'].most_common():
            percentage = count / total_mapped
            f.write(f"{genre},{count},{percentage:.2%}\n")
    
    # Save genre combinations
    with open(os.path.join(output_dir, 'genre_combinations.csv'), 'w') as f:
        f.write("combination,count,percentage\n")
        total_combos = sum(results['genre_combinations'].values())
        for combo, count in results['genre_combinations'].most_common():
            percentage = count / total_combos
            combo_str = "+".join(combo)
            f.write(f"{combo_str},{count},{percentage:.2%}\n")
    
    # Save unmapped data (existing functionality)
    with open(os.path.join(output_dir, 'no_genre_tracks.txt'), 'w') as f:
        f.write("\n".join(results['no_genre']))
    
    with open(os.path.join(output_dir, 'unmapped_only_tracks.txt'), 'w') as f:
        f.write("\n".join(results['unmapped_only']))
    
    with open(os.path.join(output_dir, 'unmapped_subgenres.csv'), 'w') as f:
        f.write("subgenre,count\n")
        for genre, count in results['unmapped_subgenres'].most_common():
            f.write(f"{genre},{count}\n")
    
    print(f"Analysis reports saved to {output_dir} directory")

if __name__ == "__main__":
    tsv_path = os.path.join('data', 'raw_30s_cleantags.tsv')
    
    print(f"Analyzing genre statistics in {tsv_path}...")
    results = analyze_genre_stats(tsv_path)
    
    # Print summary
    print("\n=== Genre Statistics ===")
    print(f"Total tracks processed: {results['total_tracks']}")
    print(f"Tracks with no genre tags: {len(results['no_genre'])} ({len(results['no_genre'])/results['total_tracks']:.1%})")
    print(f"Tracks with only unmapped genres: {len(results['unmapped_only'])} ({len(results['unmapped_only'])/results['total_tracks']:.1%})")
    
    # Print main genre counts
    print("\nTop 20 main genres:")
    for genre, count in results['main_genre_counts'].most_common(20):
        percentage = count / sum(results['main_genre_counts'].values())
        print(f"{genre}: {count} ({percentage:.1%})")
    
    # Print top genre combinations
    print("\nTop 20 genre combinations:")
    for combo, count in results['genre_combinations'].most_common(20):
        combo_str = "+".join(combo)
        print(f"{combo_str}: {count}")
    
    # Print unmapped subgenres
    print("\nTop 20 unmapped subgenres:")
    for genre, count in results['unmapped_subgenres'].most_common(20):
        print(f"{genre}: {count}")
    
    # Save detailed reports
    save_analysis_report(results)