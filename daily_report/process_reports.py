import os
import re
import json
import docx
import fitz
import pdfplumber
import pandas as pd

def get_file_date(filename):
    m = re.search(r'[（(](\d+)\.(\d+)[）)]', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d{2})(\d{2})\.(?:docx|pdf)$', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)月(\d+)日', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)\.(\d+)', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0

def parse_prefix(prefix):
    # Extract date
    date_match = re.search(r'\d+\s*月\s*\d+\s*日(?:至\d+日)?', prefix)
    date = date_match.group(0).replace(' ', '') if date_match else ''
    
    # Extract site
    sites_list = ['微博', '小红书', '抖音', '今日头条', '微信', 'X平台', 'X 平台', '快手', '即刻', '新浪微博', '百度百家号', '网易', '腾讯', '搜狐', '知乎']
    matched_sites = []
    for s in sites_list:
        if s in prefix:
            matched_sites.append(s.replace(' ', ''))
    site = '/'.join(matched_sites) if matched_sites else '网络'
    
    # Extract user
    user = '网友'
    quotes_match = re.search(r'“([^“”]+?)”', prefix)
    if quotes_match:
        user = quotes_match.group(1).strip()
    else:
        dy_match = re.search(r'抖音号为：([A-Za-z0-9_]+)', prefix)
        if dy_match:
            user = dy_match.group(1).strip()
        elif 'X平台' in site or 'X 平台' in prefix:
            user = 'X平台账号'
        elif '有网友' in prefix or '有网民' in prefix:
            user = '网友'
            
    return site, user, date

def extract_tashan_docx(docx_path):
    doc = docx.Document(docx_path)
    text_paragraphs = []
    started = False
    for p in doc.paragraphs:
        line_str = p.text.strip()
        if not started:
            if '他山之石' in line_str and any(h in line_str for h in ['一、', '二、', '三、', '四、', '五、']):
                started = True
                continue
        if started:
            if any(end_marker in line_str for end_marker in ['责编：', '复核：', '报：', '送：', '工作提示：', '北京市重点站区管理委员会宣传处']):
                started = False
                break
            if any(line_str.startswith(h) for h in ['一、', '二、', '三、', '四、', '五、']):
                started = False
                break
            text_paragraphs.append(line_str)
    cleaned = [l for l in text_paragraphs if l]
    return '\n'.join(cleaned)

def extract_tashan_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_lines = []
    started = False
    for idx_page, page in enumerate(doc):
        page_text = page.get_text('text')
        lines = page_text.split('\n')
        for line in lines:
            line_str = line.strip()
            if not started:
                if '他山之石' in line_str and (any(h in line_str for h in ['一、', '二、', '三、', '四、', '五、']) or (line_str == '他山之石' and idx_page > 0)):
                    started = True
                    continue
            if started:
                if any(end_marker in line_str for end_marker in ['责编：', '复核：', '报：', '送：', '工作提示：', '北京市重点站区管理委员会宣传处']):
                    started = False
                    break
                if any(line_str.startswith(h) for h in ['一、', '二、', '三、', '四、', '五、']):
                    started = False
                    break
                text_lines.append(line_str)
        if not started and text_lines:
            break
            
    cleaned = []
    for l in text_lines:
        if not l:
            continue
        if l.startswith('—') and l.endswith('—'):
            continue
        if l.startswith('-') and l.endswith('-'):
            continue
        if l.isdigit():
            continue
        cleaned.append(l)
    return '\n'.join(cleaned)

def extract_netizen_feedbacks_docx(docx_path, filename):
    doc = docx.Document(docx_path)
    capture = False
    paragraphs = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if any(k in t for k in ['网民反映', '不实信息', '敏感信息']) and any(h in t for h in ['一、', '二、', '三、', '四、', '五、']):
            capture = True
            continue
        if capture:
            if t.startswith('二、') or t.startswith('三、') or t.startswith('四、') or t.startswith('工作提示：') or t.startswith('责编') or t.startswith('报：') or t.startswith('送：'):
                capture = False
                break
            if t:
                paragraphs.append(t)
                
    results = []
    pattern_std = re.compile(
        r'([A-Za-z0-9\u4e00-\u9fa5]+?)(?:用户|大V|博主|账号|自媒体)?“([^“”]+?)”\s*([^“”]{0,100}?)(?:发文称|发视频称|发帖称|发声称|发布视频称|发文表示|发视频表示|说)[\s，,：:](.*)',
        re.DOTALL
    )
    
    for p in paragraphs:
        if '宣传处已将上述' in p or '流转至国铁北京局' in p or '宣传处监测到' in p:
            continue
        if '监测到涉' in p or '共监测到' in p or '网民反映分布' in p or '网民发文主要涉及' in p:
            continue
        if len(p) < 25:
            continue
        if '针对《舆情日报》' in p and ('回应：' in p or '反馈：' in p or '回应称：' in p):
            continue
            
        # Try standard pattern first
        m_std = pattern_std.search(p)
        if m_std:
            site = m_std.group(1).strip()
            user = m_std.group(2).strip()
            middle = m_std.group(3).strip()
            content = m_std.group(4).strip()
            
            date_match = re.search(r'\d+\s*月\s*\d+\s*日(?:至\d+日)?', middle)
            if not date_match:
                date_match = re.search(r'\d+\s*月\s*\d+\s*日(?:至\d+日)?', p[:30])
            date = date_match.group(0).replace(' ', '') if date_match else ''
            
            admin_patterns = [
                r'宣传处已将上述.*',
                r'工作提示：.*',
                r'截至.*时，该舆情.*'
            ]
            for ap in admin_patterns:
                content = re.sub(ap, '', content).strip()
                
            other_info_parts = [f"发布站点: {site}", f"用户ID: {user}"]
            if date:
                other_info_parts.append(f"发布时间: {date}")
            other_info = ", ".join(other_info_parts)
            
            results.append({
                '日报名称': filename,
                '反馈内容': content,
                '其他信息（发布站点等）': other_info
            })
            continue
            
        # Try prefix-split logic (requiring date)
        matched_verb = None
        for v in ['反映', '反馈', '发文称', '发视频称', '发帖称', '发声称', '说']:
            if v in p:
                if p.startswith('（') and v in p[:15]:
                    continue
                matched_verb = v
                break
        
        if matched_verb:
            parts = p.split(matched_verb, 1)
            prefix = parts[0].strip()
            content = parts[1].strip()
            
            date_match = re.search(r'\d+\s*月\s*\d+\s*日(?:至\d+日)?', prefix)
            if not date_match:
                date_match = re.search(r'\d+\s*月\s*\d+\s*日(?:至\d+日)?', p[:30])
                
            if date_match:
                date = date_match.group(0).replace(' ', '')
                site, user, _ = parse_prefix(prefix)
                
                content = re.sub(r'^[，,：:\s]+', '', content)
                admin_patterns = [
                    r'宣传处已将上述.*',
                    r'工作提示：.*',
                    r'截至.*时，该舆情.*'
                ]
                for ap in admin_patterns:
                    content = re.sub(ap, '', content).strip()
                    
                other_info_parts = [f"发布站点: {site}", f"用户ID: {user}"]
                if date:
                    other_info_parts.append(f"发布时间: {date}")
                other_info = ", ".join(other_info_parts)
                
                results.append({
                    '日报名称': filename,
                    '反馈内容': content,
                    '其他信息（发布站点等）': other_info
                })
    return results

def extract_feedbacks_table_pdf(pdf_path, filename):
    all_rows = []
    current_type = None # '4col' or '5col'
    last_station = ""
    last_domain = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if not tables:
                continue
            for table in tables:
                if not table or len(table[0]) not in [4, 5]:
                    continue
                
                row0 = [str(cell).strip() if cell is not None else "" for cell in table[0]]
                is_header = False
                if len(row0) == 4 and any(k in row0[0] or k in row0[1] or k in row0[3] for k in ['涉及场站', '场站', '发布站点', '摘要', '内容']):
                    is_header = True
                    current_type = '4col'
                elif len(row0) == 5 and any(k in row0[0] or k in row0[1] or k in row0[4] for k in ['问题领域', '发布站点', '作者', '摘要', '内容']):
                    is_header = True
                    current_type = '5col'
                
                if not is_header and current_type is not None:
                    if len(row0) == 4 and current_type == '4col':
                        pass
                    elif len(row0) == 5 and current_type == '5col':
                        pass
                    else:
                        continue
                elif not is_header and current_type is None:
                    if len(row0) == 4:
                        current_type = '4col'
                    elif len(row0) == 5:
                        current_type = '5col'
                    else:
                        continue
                
                start_row_idx = 1 if is_header else 0
                for row in table[start_row_idx:]:
                    if not row or len(row) != (4 if current_type == '4col' else 5):
                        continue
                    
                    # Clean row cells
                    # Replace \n with empty string for metadata, but space for content
                    cells = []
                    for idx, cell in enumerate(row):
                        if cell is None:
                            cells.append("")
                        else:
                            val = str(cell).strip()
                            if idx == (3 if current_type == '4col' else 4):
                                # content column
                                cells.append(val.replace('\n', ' '))
                            else:
                                # metadata column
                                cells.append(val.replace('\n', ''))
                    
                    if not any(cells):
                        continue
                        
                    if current_type == '4col':
                        station, site, time_val, summary = cells
                        if station:
                            last_station = station
                        else:
                            station = last_station
                        
                        if not summary or summary == '摘要':
                            continue
                            
                        other_info_parts = []
                        if station:
                            other_info_parts.append(f"涉及场站: {station}")
                        if site:
                            other_info_parts.append(f"发布站点: {site}")
                        if time_val:
                            other_info_parts.append(f"发布时间: {time_val}")
                        other_info = ", ".join(other_info_parts)
                        
                        all_rows.append({
                            '日报名称': filename,
                            '反馈内容': summary,
                            '其他信息（发布站点等）': other_info
                        })
                    elif current_type == '5col':
                        domain, site, author, time_val, summary = cells
                        if domain:
                            last_domain = domain
                        else:
                            domain = last_domain
                            
                        if not summary or summary == '摘要':
                            continue
                            
                        other_info_parts = []
                        if domain:
                            other_info_parts.append(f"问题领域: {domain}")
                        if site:
                            other_info_parts.append(f"发布站点: {site}")
                        if author:
                            other_info_parts.append(f"作者: {author}")
                        if time_val:
                            other_info_parts.append(f"发布时间: {time_val}")
                        other_info = ", ".join(other_info_parts)
                        
                        all_rows.append({
                            '日报名称': filename,
                            '反馈内容': summary,
                            '其他信息（发布站点等）': other_info
                        })
    return all_rows

def extract_netizen_feedbacks_pdf(pdf_path, filename):
    # First check if there are tables
    table_feedbacks = extract_feedbacks_table_pdf(pdf_path, filename)
    if table_feedbacks:
        return table_feedbacks
        
    # Otherwise, extract paragraph-based feedbacks
    doc = fitz.open(pdf_path)
    started = False
    section_lines = []
    
    for idx_page, page in enumerate(doc):
        page_text = page.get_text('text')
        lines = page_text.split('\n')
        for line in lines:
            line_str = line.strip()
            if any(k in line_str for k in ['网民反映', '不实信息', '敏感信息']) and any(h in line_str for h in ['一、', '二、', '三、', '四、', '五、']):
                started = True
                continue
            if started:
                if any(end_marker in line_str for end_marker in ['责编：', '复核：', '报：', '送：', '工作提示：', '北京市重点站区管理委员会宣传处']):
                    started = False
                    break
                if any(line_str.startswith(h) for h in ['一、', '二、', '三、', '四、', '五、']):
                    started = False
                    break
                section_lines.append(line_str)
        if not started and section_lines:
            break
            
    cleaned_lines = []
    for l in section_lines:
        if not l:
            continue
        if l.startswith('—') and l.endswith('—'):
            continue
        if l.startswith('-') and l.endswith('-'):
            continue
        if l.isdigit():
            continue
        cleaned_lines.append(l)
        
    full_text = ' '.join(cleaned_lines)
    full_text = re.sub(r'\s+', ' ', full_text)
    
    results = []
    pattern_std = re.compile(
        r'([A-Za-z0-9\u4e00-\u9fa5]+?)(?:用户|大V|博主|账号|自媒体)?“([^“”]+?)”\s*([^“”]{0,100}?)(?:发文称|发视频称|发帖称|发声称|发布视频称|发文表示|发视频表示|说)[\s，,：:]',
        re.DOTALL
    )
    
    matches_std = list(pattern_std.finditer(full_text))
    
    for i, m in enumerate(matches_std):
        site = m.group(1).strip()
        user = m.group(2).strip()
        middle = m.group(3).strip()
        
        content_start = m.end()
        content_end = matches_std[i+1].start() if i + 1 < len(matches_std) else len(full_text)
        content = full_text[content_start:content_end].strip()
        
        admin_patterns = [
            r'宣传处已将上述.*',
            r'工作提示：.*',
            r'截至.*时，该舆情.*'
        ]
        for ap in admin_patterns:
            content = re.sub(ap, '', content).strip()
            
        date_match = re.search(r'\d+\s*月\s*\d+\s*日(?:至\d+日)?', middle)
        if not date_match:
            date_match = re.search(r'\d+\s*月\s*\d+\s*日(?:至\d+日)?', full_text[max(0, m.start()-50):m.start()])
        date = date_match.group(0).replace(' ', '') if date_match else ''
        
        other_info_parts = [f"发布站点: {site}", f"用户ID: {user}"]
        if date:
            other_info_parts.append(f"发布时间: {date}")
        other_info = ", ".join(other_info_parts)
        
        results.append({
            '日报名称': filename,
            '反馈内容': content,
            '其他信息（发布站点等）': other_info
        })
        
    # Check for prefix-split fallback
    verbs_list = ['反映', '反馈', '发文称', '发视频称', '发帖称', '发声称', '说']
    sentences = re.split(r'(?<=。)|(?<=\；)', full_text)
    for s in sentences:
        s = s.strip()
        if len(s) < 25:
            continue
        if '宣传处已将上述' in s or '流转至国铁北京局' in s or '宣传处监测到' in s:
            continue
        if '监测到涉' in s or '共监测到' in s or '网民反映分布' in s or '网民发文主要涉及' in s:
            continue
        if '针对《舆情日报》' in s and ('回应：' in s or '反馈：' in s or '回应称：' in s):
            continue
            
        # If already matched by std, skip
        already_matched = False
        for r in results:
            if r['反馈内容'][:15] in s:
                already_matched = True
                break
        if already_matched:
            continue
            
        matched_verb = None
        for v in verbs_list:
            if v in s:
                if s.startswith('（') and v in s[:15]:
                    continue
                matched_verb = v
                break
                
        if matched_verb:
            parts = s.split(matched_verb, 1)
            prefix = parts[0].strip()
            content = parts[1].strip()
            
            # Require date
            date_match = re.search(r'\d+\s*月\s*\d+\s*日(?:至\d+日)?', prefix)
            if not date_match:
                date_match = re.search(r'\d+\s*月\s*\d+\s*日(?:至\d+日)?', s[:30])
                
            if date_match:
                date = date_match.group(0).replace(' ', '')
                site, user, _ = parse_prefix(prefix)
                
                content = re.sub(r'^[，,：:\s]+', '', content)
                admin_patterns = [
                    r'宣传处已将上述.*',
                    r'工作提示：.*',
                    r'截至.*时，该舆情.*'
                ]
                for ap in admin_patterns:
                    content = re.sub(ap, '', content).strip()
                    
                other_info_parts = [f"发布站点: {site}", f"用户ID: {user}"]
                if date:
                    other_info_parts.append(f"发布时间: {date}")
                other_info = ", ".join(other_info_parts)
                
                results.append({
                    '日报名称': filename,
                    '反馈内容': content,
                    '其他信息（发布站点等）': other_info
                })
                
    return results

def main():
    base_dir = '/Users/clsljt/Documents/hhgh/station/dailyReport'
    data_dir = os.path.join(base_dir, 'data')
    
    # 1. Load existing data
    tashan_json_path = os.path.join(base_dir, '他山之石.json')
    wangmin_json_path = os.path.join(base_dir, '网民反映.json')
    
    existing_tashan = []
    if os.path.exists(tashan_json_path):
        with open(tashan_json_path, 'r', encoding='utf-8') as f:
            existing_tashan = json.load(f)
            
    existing_wangmin = []
    if os.path.exists(wangmin_json_path):
        with open(wangmin_json_path, 'r', encoding='utf-8') as f:
            existing_wangmin = json.load(f)
            
    # Filter existing to keep ONLY May (month 5) data
    filtered_tashan = []
    for item in existing_tashan:
        m, d = get_file_date(item['日报名称'])
        if m == 5:
            filtered_tashan.append(item)
            
    filtered_wangmin = []
    for item in existing_wangmin:
        m, d = get_file_date(item['日报名称'])
        if m == 5:
            filtered_wangmin.append(item)
            
    # Set of already processed files (now only contains May files, so all other months will be re-processed)
    existing_tashan_files = set(item['日报名称'] for item in filtered_tashan)
    existing_wangmin_files = set(item['日报名称'] for item in filtered_wangmin)
    
    new_tashan = []
    new_wangmin = []
    
    # Folders to process
    months = ['1月', '2月', '3月', '4月', '舆情日报6月至23日']
    
    for m in months:
        month_path = os.path.join(data_dir, m)
        if not os.path.exists(month_path):
            print(f"Warning: directory {m} does not exist. Skipping.")
            continue
            
        files = sorted(os.listdir(month_path))
        print(f"Processing folder: {m} ({len(files)} files)")
        for f in files:
            if f.startswith('.'):
                continue
            fpath = os.path.join(month_path, f)
            if not os.path.isfile(fpath):
                continue
                
            # Process 他山之石
            if f not in existing_tashan_files:
                content_tashan = ""
                if f.endswith('.docx'):
                    content_tashan = extract_tashan_docx(fpath)
                elif f.endswith('.pdf'):
                    content_tashan = extract_tashan_pdf(fpath)
                
                if content_tashan:
                    new_tashan.append({
                        '日报名称': f,
                        '举措详细内容': content_tashan
                    })
                    
            # Process 网民反映
            if f not in existing_wangmin_files:
                feedbacks = []
                if f.endswith('.docx'):
                    feedbacks = extract_netizen_feedbacks_docx(fpath, f)
                elif f.endswith('.pdf'):
                    feedbacks = extract_netizen_feedbacks_pdf(fpath, f)
                    
                if feedbacks:
                    new_wangmin.extend(feedbacks)
                    
    # Combine filtered May data and new data
    combined_tashan = filtered_tashan + new_tashan
    combined_wangmin = filtered_wangmin + new_wangmin
    
    # Sort chronologically
    def sort_key(item):
        filename = item['日报名称']
        month, day = get_file_date(filename)
        return (month, day, filename)
        
    combined_tashan.sort(key=sort_key)
    combined_wangmin.sort(key=sort_key)
    
    print(f"Extraction completed!")
    print(f"他山之石: pre-existing (May) {len(filtered_tashan)}, newly-extracted {len(new_tashan)}, total {len(combined_tashan)}")
    print(f"网民反映: pre-existing (May) {len(filtered_wangmin)}, newly-extracted {len(new_wangmin)}, total {len(combined_wangmin)}")
    
    # Save JSON files
    with open(tashan_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_tashan, f, ensure_ascii=False, indent=4)
        
    with open(wangmin_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_wangmin, f, ensure_ascii=False, indent=4)
        
    # Save XLSX files
    tashan_xlsx_path = os.path.join(base_dir, '他山之石.xlsx')
    wangmin_xlsx_path = os.path.join(base_dir, '网民反映.xlsx')
    
    df_tashan = pd.DataFrame(combined_tashan)
    df_tashan.to_excel(tashan_xlsx_path, index=False)
    
    df_wangmin = pd.DataFrame(combined_wangmin)
    df_wangmin.to_excel(wangmin_xlsx_path, index=False)
    
    print("Files successfully saved!")

if __name__ == '__main__':
    main()
