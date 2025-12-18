#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 内容提取工具
"""

import pdfplumber
import sys
import os

def extract_pdf_content(pdf_path, output_path=None):
    """
    从PDF文件中提取文本内容
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = []
            
            print(f"PDF 总页数: {len(pdf.pages)}")
            
            # 提取每一页的内容
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"处理第 {page_num} 页...")
                
                # 提取文本
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"\n=== 第 {page_num} 页 ===\n")
                    text_content.append(page_text)
                    text_content.append("\n")
            
            # 合并所有内容
            full_text = "".join(text_content)
            
            # 保存到文件
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                print(f"内容已保存到: {output_path}")
            else:
                print("=== PDF 内容摘要 ===")
                # 只打印前2000个字符作为预览
                preview = full_text[:2000]
                print(preview)
                if len(full_text) > 2000:
                    print(f"\n... (还有 {len(full_text) - 2000} 个字符)")
                    
            return full_text
            
    except Exception as e:
        print(f"读取PDF时出错: {e}")
        return None

if __name__ == "__main__":
    pdf_file = "OmniFuse.pdf"
    output_file = "OmniFuse_content.txt"
    
    if os.path.exists(pdf_file):
        content = extract_pdf_content(pdf_file, output_file)
        
        # 简单的关键词搜索来了解论文内容
        if content:
            print("\n=== 关键信息搜索 ===")
            keywords = ["abstract", "introduction", "method", "architecture", "fusion", "module", "experiment"]
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    print(f"✓ 找到关键词: {keyword}")
                else:
                    print(f"✗ 未找到关键词: {keyword}")
    else:
        print(f"文件不存在: {pdf_file}")