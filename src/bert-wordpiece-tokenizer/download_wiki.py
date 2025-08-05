import bz2
import re
import urllib.request
import os
from pathlib import Path

def download_wikipedia_dump():
    """Download March 2024 Wikipedia dump with progress tracking."""
    url = "https://dumps.wikimedia.org/enwiki/20240301/enwiki-20240301-pages-articles-multistream.xml.bz2"
    filename = "enwiki-20240301-pages-articles-multistream.xml.bz2"
    
    print(f"Downloading {filename} from {url}")
    print("This is a 25GB file - it will take some time...")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = round((block_num * block_size / total_size) * 100, 2)
            mb_downloaded = round((block_num * block_size) / (1024 * 1024), 2)
            mb_total = round(total_size / (1024 * 1024), 2)
            print(f"\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\nDownload completed: {filename}")
        return filename
    except Exception as e:
        print(f"\nDownload failed: {e}")
        # Try fallback mirror or alternative approach
        print("Trying alternative download method...")
        return None

def stream_and_clean_wikipedia(bz2_file, output_file="wiki_clean.txt", max_size_mb=100):
    """Stream decompress and clean Wikipedia XML, output plain sentences."""
    print(f"Processing {bz2_file} -> {output_file}")
    
    # XML tag removal regex
    xml_tag_regex = re.compile(r'<[^>]*>')
    
    max_bytes = max_size_mb * 1024 * 1024
    bytes_written = 0
    
    with bz2.open(bz2_file, 'rt', encoding='utf-8', errors='ignore') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            buffer = ""
            
            for line in infile:
                if bytes_written >= max_bytes:
                    break
                    
                # Remove XML tags
                clean_line = xml_tag_regex.sub('', line).strip()
                
                if clean_line and len(clean_line) > 10:  # Skip very short lines
                    # Simple sentence splitting on periods and newlines
                    sentences = re.split(r'[.\n]+', clean_line)
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        # Basic filtering: must have at least 5 characters and some letters
                        if len(sentence) > 5 and re.search(r'[a-zA-Z]', sentence):
                            # Remove non-ASCII characters if desired
                            # sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
                            
                            if sentence:
                                line_to_write = sentence + '\n'
                                outfile.write(line_to_write)
                                bytes_written += len(line_to_write.encode('utf-8'))
                                
                                if bytes_written >= max_bytes:
                                    break
                
                # Progress update every 1000 lines
                if infile.tell() % 100000 == 0:
                    mb_processed = infile.tell() / (1024 * 1024)
                    print(f"\rProcessed: {mb_processed:.1f} MB, Output: {bytes_written/(1024*1024):.1f} MB", end="")
    
    print(f"\nCleaning completed. Output file: {output_file} ({bytes_written/(1024*1024):.1f} MB)")
    return output_file

if __name__ == "__main__":
    # Download the dump
    dump_file = download_wikipedia_dump()
    
    if dump_file and os.path.exists(dump_file):
        # Process and clean
        clean_file = stream_and_clean_wikipedia(dump_file)
        print(f"Ready for tokenizer training: {clean_file}")
    else:
        print("Download failed. You may need to download manually or try a different mirror.")