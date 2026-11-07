import os
import pandas as pd
import shutil

def clean_article_text(text, title):
    # 1. Remove the footer starting from "预览时标签不可点"
    if '预览时标签不可点' in text:
        text = text.split('预览时标签不可点')[0]
    
    # 2. Split into lines to remove the header
    lines = text.split('\n')
    header_patterns = {
        '北京市重点站区管理委员会',
        '在小说阅读器读本章',
        '去阅读',
        '在小说阅读器中沉浸阅读',
        title.strip(),
    }
    t_clean = title.strip()
    header_patterns.add(t_clean.replace('_', '|'))
    header_patterns.add(t_clean.replace('|', '_'))
    
    start_idx = 0
    for i, line in enumerate(lines):
        val = line.strip()
        if not val:
            continue
        if val in header_patterns:
            continue
        start_idx = i
        break
        
    cleaned_lines = lines[start_idx:]
    cleaned_text = '\n'.join(cleaned_lines).strip()
    return cleaned_text

def find_matching_file(row, files):
    time_str = str(row['时间']).strip()
    prefix = time_str.replace(':', '-')
    title = str(row['标题']).strip()
    
    matches = [f for f in files if f.startswith(prefix) and f.endswith('.txt')]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        clean_title = ''.join(c for c in title if c.isalnum())
        for m in matches:
            clean_m = ''.join(c for c in m if c.isalnum())
            if clean_title in clean_m or clean_m in clean_title:
                return m
            # Fallback check
            if len(clean_title) > 5 and clean_title[:5] in clean_m:
                return m
            if len(clean_title) > 5 and clean_title[-5:] in clean_m:
                return m
    return None

def main():
    excel_path = '/Users/clsljt/Documents/hhgh/station/gongzhonghao/article_summary.xlsx'
    txt_dir = '/Users/clsljt/Documents/hhgh/station/gongzhonghao/article_contents'
    
    print("Loading excel...")
    df = pd.read_excel(excel_path)
    
    # List files in the directory
    txt_files = os.listdir(txt_dir)
    
    cleaned_data = []
    print("Processing articles...")
    for idx, row in df.iterrows():
        title = str(row['标题'])
        raw_text = str(row['文章正文'])
        
        # Clean text
        cleaned_text = clean_article_text(raw_text, title)
        df.at[idx, '文章正文'] = cleaned_text
        
        # Find matching txt file
        txt_file = find_matching_file(row, txt_files)
        if txt_file:
            txt_path = os.path.join(txt_dir, txt_file)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
        else:
            print(f"Warning: Could not match text file for row {idx} ({title})")
            
    print("Saving cleaned excel file...")
    df.to_excel(excel_path, index=False)
    print("Processing completed successfully!")

if __name__ == '__main__':
    main()
