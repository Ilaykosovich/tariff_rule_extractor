import sys
import os
from pdf_pipeline import run_ocr_pipeline, summarize_pages

def main(argv=None):
    # Получаем первый PDF файл из папки pdf_data
    pdf_data_dir = "pdf_data"
    pdf_files = [f for f in os.listdir(pdf_data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("Ошибка: PDF файлы не найдены в папке pdf_data")
        return
    
    pdf_file = os.path.join(pdf_data_dir, pdf_files[0])
    print(f"Обработка файла: {pdf_file}")
    pages = run_ocr_pipeline(pdf_file)
    print(f"Extracted pages: {len(pages)}")
    summary_result = summarize_pages(pdf_file, pages)
    summaries = {
        int(summary["page_number"]): summary.get("plain_summary", "")
        for summary in summary_result.get("summaries", [])
    }
    for sam in summaries.values():
        print(sam)
    print(f"Summarized pages: {len(summaries)}")
    
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        print("Arguments:", argv)
    else:
        print("Hello from main.py")

if __name__ == "__main__":
    main()
