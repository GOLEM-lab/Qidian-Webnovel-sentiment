import csv
import os
import re
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Optional


def split_by_comment_numbers(text: str) -> List[str]:
    """
    Split text by comment number indicators.
    Priority: Find all "punctuation + number" patterns and split there.
    Numbers are removed from the output.
    
    Split rules:
    - Split at: 。！？" + number (sentence end punctuation)
    - Check for attribution after 。！？" (keep together if found)
    - Never split at: " + number (dialogue start)
    - Remove all comment numbers from output
    
    Args:
        text: Chinese text with embedded comment numbers
        
    Returns:
        List of sections split by comment numbers (with numbers removed)
    """
    if not text or not text.strip():
        return []
    
    # Step 1: Remove very long numbers (likely IDs like 20361055.0)
    text_cleaned = re.sub(r'\d{5,}\.?\d*', '', text)
    
    # Step 2: Find all comment number patterns (punctuation + 1-4 digit number)
    # Now includes: 。！？" (closing quote) and " (opening quote)
    pattern = r'([。！？""])\s*(\d{1,4})\s*'
    
    # Collect valid split positions
    split_positions = []
    
    matches = list(re.finditer(pattern, text_cleaned))
    
    for match in matches:
        punctuation = match.group(1)
        
        # Rule 1: Skip opening quotes "
        if punctuation == '"':
            continue
        
        # Rule 2: For 。！？", check if followed by attribution
        # Examples: 
        #   "你在干什么？"他问。 → keep together
        #   他很生气！他摔门而出。 → keep together
        if punctuation in ['。', '！', '？', '"']:
            next_start = match.end()
            next_chunk = text_cleaned[next_start:next_start + 20] if next_start < len(text_cleaned) else ""
            
            # Attribution indicators (words describing how something was said/done)
            attribution_words = ['说', '道', '喊', '问', '答', '叫', '叹', '回', '笑', '骂', '叹', '哭', '想', '念', '看', '摔', '转', '走', '跑']
            
            # Check if attribution follows
            has_attribution = any(word in next_chunk[:8] for word in attribution_words)
            
            if has_attribution:
                continue
        
        # Valid split point: store positions
        split_positions.append({
            'punct_end': match.start() + 1,
            'number_end': match.end()
        })
    
    # Step 3: Extract sections and remove comment numbers
    sections = []
    last_end = 0
    
    for pos in split_positions:
        # Extract content from last position up to punctuation
        content = text_cleaned[last_end:pos['punct_end']].strip()
        
        if content:
            sections.append(content)
        
        # Next section starts after the comment number
        last_end = pos['number_end']
    
    # Add remaining text
    if last_end < len(text_cleaned):
        remaining = text_cleaned[last_end:].strip()
        if remaining:
            # Remove any trailing numbers
            remaining = re.sub(r'\s*\d{1,4}\s*', '', remaining).strip()
            if remaining:
                sections.append(remaining)
    
    return sections


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences based on Chinese punctuation marks.
    Handles dialogue quotes properly.
    
    Args:
        text: Chinese text to split
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    sentence_markers = ["。", "！", "？", "..."]
    sentences = []
    start = 0
    i = 0
    
    while i < len(text):
        char = text[i]
        
        # Handle dialogue: find matching quotes
        if char == '"':
            # Find the closing quote
            quote_end = i
            for j in range(i + 1, len(text)):
                if text[j] == '"':
                    quote_end = j
                    break
            
            # If we found a closing quote, treat the dialogue as one unit
            if quote_end > i:
                # Check if there's a sentence marker before the closing quote
                dialogue_text = text[i:quote_end + 1]
                # Look for sentence markers within dialogue
                has_marker = any(marker in dialogue_text for marker in sentence_markers)
                
                if has_marker:
                    # Split at the marker within dialogue
                    for marker in sentence_markers:
                        marker_pos = dialogue_text.find(marker)
                        if marker_pos != -1:
                            sentence = text[start:quote_end + 1].strip()
                            if sentence:
                                sentences.append(sentence)
                            start = quote_end + 1
                            i = quote_end + 1
                            break
                    else:
                        i += 1
                else:
                    i = quote_end + 1
            else:
                i += 1
        elif char in sentence_markers:
            sentence = text[start:i + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = i + 1
            i += 1
        else:
            i += 1
    
    # Add remaining text if any
    if start < len(text):
        remaining = text[start:].strip()
        if remaining:
            sentences.append(remaining)
    
    return sentences


def smart_segment_by_punctuation(sentences: List[str]) -> List[str]:
    """
    Apply punctuation-based rules to group sentences into paragraphs.
    
    Rules:
    1. Dialogue + action/thought continuation: "..." + "。" (non-quote) → merge
    2. Emotional continuation: consecutive "？" or "！" → merge
    3. Interrupted dialogue: "。！？"" + non-"。" + "..." → merge 3 sentences
    4. Default: sentences ending with "。！？" → new paragraph
    
    Args:
        sentences: List of sentences to group
        
    Returns:
        List of paragraphs
    """
    if not sentences:
        return []
    
    paragraphs = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i]
        
        # Rule 1: Dialogue + action/thought continuation
        if (i + 1 < len(sentences) and 
            current.endswith('"') and 
            sentences[i + 1].endswith('。') and 
            not sentences[i + 1].startswith('"')):
            paragraphs.append(current + sentences[i + 1])
            i += 2
        
        # Rule 2: Emotional continuation
        elif (i + 1 < len(sentences) and 
              current.endswith(('？', '！')) and 
              sentences[i + 1].endswith(('？', '！'))):
            paragraphs.append(current + sentences[i + 1])
            i += 2
        
        # Rule 3: Interrupted dialogue continuation
        elif (i + 2 < len(sentences) and 
              current.endswith(('。', '！', '？', '"')) and 
              not sentences[i + 1].endswith('。') and 
              sentences[i + 2].startswith('"')):
            paragraphs.append(current + sentences[i + 1] + sentences[i + 2])
            i += 3
        
        # Rule 4: Default rules
        else:
            if current.endswith(('。', '！', '？')):
                paragraphs.append(current)
                i += 1
            elif current.endswith('"'):
                if i + 1 < len(sentences) and sentences[i + 1].startswith('"'):
                    paragraphs.append(current)
                    i += 1
                else:
                    if i + 1 < len(sentences):
                        paragraphs.append(current + sentences[i + 1])
                        i += 2
                    else:
                        paragraphs.append(current)
                        i += 1
            else:
                if i + 1 < len(sentences):
                    paragraphs.append(current + sentences[i + 1])
                    i += 2
                else:
                    paragraphs.append(current)
                    i += 1
    
    return paragraphs


def adjust_to_target_count(paragraphs: List[str], target_count: int) -> List[str]:
    """
    Adjust paragraph list to match exact target count by merging or splitting.
    
    Args:
        paragraphs: Current list of paragraphs
        target_count: Required number of paragraphs
        
    Returns:
        Adjusted list with exactly target_count paragraphs
    """
    current_count = len(paragraphs)
    
    if current_count == target_count:
        return paragraphs
    
    # Need to merge paragraphs
    if current_count > target_count:
        while len(paragraphs) > target_count:
            # Find shortest adjacent pair to merge
            min_len = float('inf')
            merge_idx = 0
            
            for i in range(len(paragraphs) - 1):
                combined_len = len(paragraphs[i]) + len(paragraphs[i + 1])
                if combined_len < min_len:
                    min_len = combined_len
                    merge_idx = i
            
            # Merge the pair
            paragraphs[merge_idx] = paragraphs[merge_idx] + paragraphs[merge_idx + 1]
            paragraphs.pop(merge_idx + 1)
    
    # Need to split paragraphs
    elif current_count < target_count:
        while len(paragraphs) < target_count:
            # Find longest paragraph to split
            max_len = 0
            split_idx = 0
            
            for i, para in enumerate(paragraphs):
                if len(para) > max_len:
                    max_len = len(para)
                    split_idx = i
            
            # Split the longest paragraph
            para_to_split = paragraphs[split_idx]
            sentences = split_into_sentences(para_to_split)
            
            if len(sentences) > 1:
                # Split roughly in half
                mid = len(sentences) // 2
                first_half = ''.join(sentences[:mid])
                second_half = ''.join(sentences[mid:])
                
                paragraphs[split_idx] = first_half
                paragraphs.insert(split_idx + 1, second_half)
            else:
                # Can't split further by sentences, split by character count
                mid = len(para_to_split) // 2
                paragraphs[split_idx] = para_to_split[:mid]
                paragraphs.insert(split_idx + 1, para_to_split[mid:])
    
    return paragraphs


def segment_chinese_chapter(chapter_text: str, target_paragraph_count: int, 
                            use_punctuation_rules: bool = True) -> Tuple[List[str], int, str]:
    """
    Main segmentation function with mandatory target count.
    
    Priority Flow:
    1. Split by comment numbers (punctuation + number patterns)
    2. If comment numbers found but count doesn't match target, adjust to target
    3. If no comment numbers, use punctuation rules then adjust to target
    4. ALWAYS return exactly target_paragraph_count paragraphs
    
    Args:
        chapter_text: Chapter text to segment
        target_paragraph_count: REQUIRED target paragraph count from master file
        use_punctuation_rules: Whether to use punctuation-based rules
        
    Returns:
        Tuple of (paragraphs, sentence_count, method_used)
    """
    if not chapter_text or not chapter_text.strip():
        return [chapter_text] if chapter_text else [], 0, "empty"
    
    # STEP 1: Split by comment numbers first (highest priority)
    sections_by_comments = split_by_comment_numbers(chapter_text)
    
    if len(sections_by_comments) > 1:
        # Successfully found comment-number boundaries
        
        # Check if comment count matches target exactly
        if len(sections_by_comments) == target_paragraph_count:
            # Perfect match!
            total_sentences = sum(len(split_into_sentences(s)) for s in sections_by_comments)
            return sections_by_comments, total_sentences, "comment_numbers_exact"
        
        # Comment count doesn't match - need to adjust
        elif len(sections_by_comments) < target_paragraph_count:
            # Have fewer segments than needed - further split using punctuation
            final_paragraphs = []
            
            for section in sections_by_comments:
                section_sentences = split_into_sentences(section)
                
                if len(section_sentences) > 1:
                    # Apply punctuation rules to further split this section
                    sub_paragraphs = smart_segment_by_punctuation(section_sentences)
                    final_paragraphs.extend(sub_paragraphs)
                else:
                    final_paragraphs.append(section)
            
            # Adjust to exact target count
            final_paragraphs = adjust_to_target_count(final_paragraphs, target_paragraph_count)
            total_sentences = sum(len(split_into_sentences(p)) for p in final_paragraphs)
            return final_paragraphs, total_sentences, "comment_numbers_adjusted_up"
        
        else:
            # Have more segments than needed - merge some
            adjusted = adjust_to_target_count(sections_by_comments, target_paragraph_count)
            total_sentences = sum(len(split_into_sentences(p)) for p in adjusted)
            return adjusted, total_sentences, "comment_numbers_adjusted_down"
    
    # No comment numbers found - use punctuation rules and adjust
    sentences = split_into_sentences(chapter_text)
    sentence_count = len(sentences)
    
    if sentence_count == 0:
        # No sentences detected, split text into target_paragraph_count chunks
        chunk_size = max(1, len(chapter_text) // target_paragraph_count)
        paragraphs = []
        for i in range(0, len(chapter_text), chunk_size):
            paragraphs.append(chapter_text[i:i + chunk_size])
        paragraphs = adjust_to_target_count(paragraphs, target_paragraph_count)
        return paragraphs, 0, "no_sentences_adjusted"
    
    # Use punctuation rules if enabled
    if use_punctuation_rules:
        paragraphs = smart_segment_by_punctuation(sentences)
    else:
        # Simple sentence-based segmentation
        sentences_per_paragraph = max(1, sentence_count // target_paragraph_count)
        paragraphs = []
        for i in range(0, len(sentences), sentences_per_paragraph):
            paragraph = ''.join(sentences[i:i+sentences_per_paragraph])
            paragraphs.append(paragraph)
    
    # Always adjust to exact target count
    paragraphs = adjust_to_target_count(paragraphs, target_paragraph_count)
    
    return paragraphs, sentence_count, "punctuation_rules_adjusted"


def load_master_alignment_file(csv_path: str) -> dict:
    """
    Load master alignment file with paragraph counts.
    Simple approach: read line by line, skip header, parse data directly.
    
    Args:
        csv_path: Path to master alignment CSV file
        
    Returns:
        Dictionary mapping (book_id, chapter_id) -> paragraph_count
    """
    paragraph_counts = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Read {len(lines)} lines from master file")
        
        # Skip first line (header)
        data_lines = lines[1:]
        
        loaded_count = 0
        error_count = 0
        
        for line_num, line in enumerate(data_lines, start=2):
            line = line.strip()
            if not line:
                continue
            
            # Split by semicolon
            parts = line.split(';')
            
            # We need at least 3 columns: book_id, chapter_id, num_paragraphs
            # Assuming format: book_id;chapter_id;...other columns...;num_paragraphs
            if len(parts) < 3:
                error_count += 1
                continue
            
            try:
                # First column: book_id
                book_id = str(int(float(parts[0])))
                # Second column: chapter_id  
                chapter_id = str(int(float(parts[1])))
                # Last column: num_paragraphs (assuming it's the last one)
                para_count = int(float(parts[-1]))
                
                key = f"{book_id}_{chapter_id}"
                paragraph_counts[key] = para_count
                loaded_count += 1
                
            except (ValueError, IndexError) as e:
                error_count += 1
                continue
        
        print(f"Successfully loaded {loaded_count} chapter mappings")
        print(f"Skipped {error_count} lines due to parsing errors")
        
    except FileNotFoundError:
        print(f"ERROR: Master alignment file not found: {csv_path}")
    except Exception as e:
        print(f"ERROR loading master alignment file: {e}")
        import traceback
        traceback.print_exc()
    
    return paragraph_counts


def process_chapters(input_folder: str, output_folder: str, 
                    master_alignment_csv: str,
                    use_punctuation_rules: bool = True,
                    debug: bool = False) -> dict:
    """
    Process all chapter CSV files and segment them using master alignment file.
    
    Args:
        input_folder: Folder containing input CSV files
        output_folder: Folder to save output CSV files
        master_alignment_csv: Path to master alignment CSV (REQUIRED)
        use_punctuation_rules: Whether to use punctuation-based rules
        debug: Whether to print debug information
        
    Returns:
        Statistics dictionary
    """
    # Load master alignment file (REQUIRED)
    if not os.path.exists(master_alignment_csv):
        print(f"ERROR: Master alignment file not found: {master_alignment_csv}")
        print("This file is REQUIRED for processing. Cannot continue.")
        return {"error": "Master alignment file not found"}
    
    paragraph_counts = load_master_alignment_file(master_alignment_csv)
    
    if not paragraph_counts:
        print("ERROR: No paragraph counts loaded from master file. Cannot continue.")
        return {"error": "No paragraph counts loaded"}
    
    # Initialize statistics
    stats = {
        "total_chapters": 0,
        "comment_numbers_exact": 0,
        "comment_numbers_adjusted_up": 0,
        "comment_numbers_adjusted_down": 0,
        "punctuation_rules_adjusted": 0,
        "no_sentences_adjusted": 0,
        "missing_target_count": 0,
        "errors": 0
    }
    
    os.makedirs(output_folder, exist_ok=True)
    book_data = defaultdict(list)
    
    # Check folder exists
    print(f"\nLooking in folder: {os.path.abspath(input_folder)}")
    print(f"Folder exists: {os.path.exists(input_folder)}")
    
    if not os.path.exists(input_folder):
        print(f"ERROR: Folder does not exist: {input_folder}")
        return stats
    
    # Process CSV files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    if len(csv_files) == 0:
        all_files = os.listdir(input_folder)[:10]
        print(f"Sample files in folder: {all_files}")
    
    for filename in csv_files:
        # Extract book_id from filename
        book_id = os.path.splitext(filename)[0]
        
        input_file = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")
        print(f"  Book ID: {book_id}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(input_file)
            
            # Identify columns
            chapter_index_col = None
            chapter_id_col = None
            content_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'index' in col_lower and 'chapter' in col_lower:
                    chapter_index_col = col
                elif 'id' in col_lower and 'chapter' in col_lower:
                    chapter_id_col = col
                elif 'content' in col_lower:
                    content_col = col
            
            if not content_col:
                print(f"  ERROR: Could not find content column in {filename}")
                print(f"  Available columns: {df.columns.tolist()}")
                stats["errors"] += 1
                continue
            
            if debug:
                print(f"  Using columns: Content={content_col}, ChapterID={chapter_id_col}")
            
            # Process each chapter in the CSV
            for _, row in df.iterrows():
                chapter_id = str(int(row[chapter_id_col])) if chapter_id_col else "unknown"
                chapter_text = str(row[content_col])
                
                if not chapter_text or chapter_text == 'nan' or not chapter_text.strip():
                    print(f"  Empty content for chapter {chapter_id}, skipping")
                    continue
                
                # Get target paragraph count from master file (REQUIRED)
                key = f"{book_id}_{chapter_id}"
                
                if key not in paragraph_counts:
                    print(f"  WARNING: No target count for book {book_id}, chapter {chapter_id}")
                    stats["missing_target_count"] += 1
                    continue
                
                target_count = paragraph_counts[key]
                
                if debug:
                    print(f"  Chapter {chapter_id} - Target paragraphs: {target_count}")
                    print(f"  Content length: {len(chapter_text)} chars")
                
                # Segment the chapter (MUST match target count)
                segments, sentence_count, method = segment_chinese_chapter(
                    chapter_text, 
                    target_count, 
                    use_punctuation_rules
                )
                
                # Verify we got exactly the right count
                if len(segments) != target_count:
                    print(f"  ERROR: Got {len(segments)} segments but target was {target_count}")
                    stats["errors"] += 1
                    continue
                
                # Update statistics
                stats["total_chapters"] += 1
                if method in stats:
                    stats[method] += 1
                
                print(f"  ✓ Chapter {chapter_id} - Method: {method}, Paragraphs: {len(segments)} (target: {target_count})")
                
                # Add segments to book data
                for seg_idx, segment in enumerate(segments):
                    segment_data = {
                        'bookId': book_id,
                        'chapterId': chapter_id,
                        'text': segment,
                        'segmentIndex': seg_idx + 1
                    }
                    book_data[book_id].append(segment_data)
                    
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            stats["errors"] += 1
            continue
    
    # Save results
    print(f"\nSaving results...")
    for book_id, segments in book_data.items():
        output_file = os.path.join(output_folder, f"{book_id}.csv")
        book_df = pd.DataFrame(segments)
        book_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  Saved {len(segments)} segments for book {book_id}")
    
    return stats


def main():
    """Main execution function"""
    # Configuration
    input_folder = "data/qidianFreeChapters"
    output_folder = "qidian_freechapter_byparagraph_aligned"
    master_alignment_csv = "scratch/p315327/qidian_webnovel_chapters_merge_meta_aligned_with_paragraphs.csv"
    use_punctuation_rules = True
    debug = False
    
    print("=" * 80)
    print("Chinese Chapter Segmentation - With Master Alignment File")
    print("=" * 80)
    print(f"Master alignment file: {master_alignment_csv}")
    print()
    
    # Process chapters
    stats = process_chapters(
        input_folder=input_folder,
        output_folder=output_folder,
        master_alignment_csv=master_alignment_csv,
        use_punctuation_rules=use_punctuation_rules,
        debug=debug
    )
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Segmentation Statistics:")
    print("=" * 80)
    
    # Check if there was an error
    if 'error' in stats:
        print(f"ERROR: {stats['error']}")
        return
    
    print(f"Total chapters processed: {stats['total_chapters']}")
    print(f"Chapters skipped (no target count): {stats['missing_target_count']}")
    
    if stats['total_chapters'] > 0:
        print(f"\nMethod breakdown:")
        print(f"  Comment numbers (exact match): {stats['comment_numbers_exact']} ({stats['comment_numbers_exact']/stats['total_chapters']*100:.1f}%)")
        print(f"  Comment numbers (adjusted up): {stats['comment_numbers_adjusted_up']} ({stats['comment_numbers_adjusted_up']/stats['total_chapters']*100:.1f}%)")
        print(f"  Comment numbers (adjusted down): {stats['comment_numbers_adjusted_down']} ({stats['comment_numbers_adjusted_down']/stats['total_chapters']*100:.1f}%)")
        print(f"  Punctuation rules (adjusted): {stats['punctuation_rules_adjusted']} ({stats['punctuation_rules_adjusted']/stats['total_chapters']*100:.1f}%)")
        print(f"  No sentences (adjusted): {stats['no_sentences_adjusted']} ({stats['no_sentences_adjusted']/stats['total_chapters']*100:.1f}%)")
        print(f"  Errors: {stats['errors']}")
    else:
        print("No chapters were processed.")
    
    print("\n" + "=" * 80)
    print("Segmentation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()