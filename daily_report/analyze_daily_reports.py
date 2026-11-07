import os
import glob
import re
import docx
import fitz
import pandas as pd

def reconstruct_pdf_paragraphs(lines):
    paragraphs = []
    current_para = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.startswith('—') and line_str.endswith('—'):
            continue
        if line_str.isdigit():
            continue
            
        is_new_para = False
        if any(line_str.startswith(h) for h in ['一、', '二、', '三、', '四、', '五、', '六、', '●', '工作提示：', '工作建议：']):
            is_new_para = True
        elif re.match(r'^\d+\.', line_str) or re.match(r'^（[一二三四五六七八九十]）', line_str) or re.match(r'^\d+月\d+日', line_str):
            is_new_para = True
        elif line.startswith('   ') or line.startswith('\t'):
            is_new_para = True
            
        if is_new_para and current_para:
            paragraphs.append(''.join(current_para))
            current_para = []
            
        current_para.append(line_str)
        
    if current_para:
        paragraphs.append(''.join(current_para))
    return paragraphs

def parse_filename_info(filename):
    issue_match = re.search(r'第\s*(\d+)\s*期', filename)
    issue_no = int(issue_match.group(1)) if issue_match else None
    
    month, day = 0, 0
    m = re.search(r'[（(](\d+)\.(\d+)[）)]', filename)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r'（(\d+)月(\d+)日）', filename)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
        else:
            m = re.search(r'(\d+)\.(\d+)', filename)
            if m:
                month, day = int(m.group(1)), int(m.group(2))
                
    if month > 0 and day > 0:
        date_str = f"2026-{month:02d}-{day:02d}"
    else:
        date_str = "未知日期"
        
    return date_str, issue_no

def extract_measures_from_file(path):
    if path.endswith('.docx'):
        try:
            doc = docx.Document(path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        except Exception as e:
            print(f"Error reading docx {path}: {e}")
            return None
    else:
        try:
            doc = fitz.open(path)
            lines = []
            for page in doc:
                lines.extend([l for l in page.get_text('text').split('\n')])
            paragraphs = reconstruct_pdf_paragraphs(lines)
        except Exception as e:
            print(f"Error reading pdf {path}: {e}")
            return None
            
    sections = []
    current_section = {'heading': 'Header', 'paragraphs': []}
    for p in paragraphs:
        if any(p.startswith(h) for h in ['一、', '二、', '三、', '四、', '五、', '六、']):
            if current_section['paragraphs'] or current_section['heading'] != 'Header':
                sections.append(current_section)
            current_section = {'heading': p, 'paragraphs': []}
        else:
            current_section['paragraphs'].append(p)
    if current_section['paragraphs'] or current_section['heading'] != 'Header':
        sections.append(current_section)
        
    lead_inst = []
    impl_feed = []
    inst_rect = []
    tashan_sug = []
    
    for sec in sections:
        h = sec['heading']
        paras = sec['paragraphs']
        
        if '领导批示' in h:
            lead_inst.append('\n'.join(paras))
            continue
            
        if any(k in h for k in ['落实', '反馈', '整改', '应对', '处置']):
            impl_feed.append('\n'.join(paras))
            continue
            
        if '他山之石' in h:
            tashan_sug.append('\n'.join(paras))
            continue
            
        # Parse individual paragraphs for action patterns
        netizen_measures = []
        for p in paras:
            if any(k in p for k in ['工作提示：', '工作建议：', '提示：', '建议：', '建议各', '工作提示', '工作建议']):
                netizen_measures.append(p)
            elif any(k in p for k in ['已将上述', '转办', '接宣传处', '迅速行动', '高度重视', '第一时间', '回应称', '反馈称', '回应：', '反馈：']):
                netizen_measures.append(p)
            elif '他山之石' not in h and any(k in p for k in ['采取了', '进行整改', '开展整改', '制定了', '推出', '优化', '提升', '解决']):
                if len(p) > 30 and any(kw in p for kw in ['站区', '管委会', '办公室', '站办', '宣传处', '交通委', '分局', '大队']):
                    netizen_measures.append(p)
        if netizen_measures:
            inst_rect.append('\n'.join(netizen_measures))
            
    lead_inst_str = '\n\n'.join(lead_inst).strip()
    impl_feed_str = '\n\n'.join(impl_feed).strip()
    inst_rect_str = '\n\n'.join(inst_rect).strip()
    tashan_sug_str = '\n\n'.join(tashan_sug).strip()
    
    # Compile a clean overall summary
    summary_parts = []
    if lead_inst_str:
        summary_parts.append(f"【领导批示】\n{lead_inst_str}")
    if impl_feed_str:
        summary_parts.append(f"【落实与反馈】\n{impl_feed_str}")
    if inst_rect_str:
        summary_parts.append(f"【即时整改与处置】\n{inst_rect_str}")
    if tashan_sug_str:
        summary_parts.append(f"【他山之石与建议】\n{tashan_sug_str}")
        
    if summary_parts:
        overall_summary = '\n\n'.join(summary_parts)
    else:
        overall_summary = "今日主要为日常场站新闻与网民点赞/反映，未记录明确的舆情处置整改措施或工作建议。"
        
    return {
        '领导批示': lead_inst_str,
        '落实与反馈': impl_feed_str,
        '即时整改与处置': inst_rect_str,
        '他山之石与工作建议': tashan_sug_str,
        '主要整改措施与建议汇总': overall_summary
    }

def main():
    data_dir = '/Users/clsljt/Documents/hhgh/station/dailyReport/data'
    output_excel = '/Users/clsljt/Documents/hhgh/station/dailyReport/舆情日报整改措施与建议汇总.xlsx'
    
    # Find all daily report files
    all_files = sorted(glob.glob(os.path.join(data_dir, '**/舆情日报*.docx'), recursive=True) + 
                       glob.glob(os.path.join(data_dir, '**/舆情日报*.pdf'), recursive=True))
    
    print(f"Found {len(all_files)} daily report files.")
    
    records = []
    for f in all_files:
        filename = os.path.basename(f)
        date_str, issue_no = parse_filename_info(filename)
        
        # Extract content
        res = extract_measures_from_file(f)
        if res is None:
            continue
            
        record = {
            '日期': date_str,
            '期数': f"第{issue_no}期" if issue_no else "",
            '日报名称': filename,
            '领导批示': res['领导批示'],
            '落实与反馈': res['落实与反馈'],
            '即时整改与处置': res['即时整改与处置'],
            '他山之石与工作建议': res['他山之石与工作建议'],
            '主要整改措施与建议汇总': res['主要整改措施与建议汇总']
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    
    # Sort chronologically by date
    df.sort_values(by='日期', inplace=True)
    
    # Write to Excel
    print(f"Writing {len(df)} records to {output_excel}...")
    df.to_excel(output_excel, index=False)
    print("Completed!")

if __name__ == '__main__':
    main()
