import os
import subprocess
import shutil
import fitz

def build_all():
    env = os.environ.copy()
    paths = env.get("PATH", "").split(";")
    clean_paths = [p for p in paths if "anaconda" not in p.lower() and "conda" not in p.lower()]
    env["PATH"] = ";".join(clean_paths)

    thesis_dir = r"c:\RTM\IIT KGP\sem 7\BTP1_repo\thesis"
    miktex_bin = r"C:\Users\DELL\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
    pdflatex = os.path.join(miktex_bin, "pdflatex.exe")
    bibtex = os.path.join(miktex_bin, "bibtex.exe")

    # 1. Build main.tex
    print("Building main.pdf...")
    subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"], cwd=thesis_dir, env=env, capture_output=True)
    subprocess.run([bibtex, "main"], cwd=thesis_dir, env=env, capture_output=True)
    subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"], cwd=thesis_dir, env=env, capture_output=True)
    res = subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"], cwd=thesis_dir, env=env, capture_output=True, text=True)

    main_pdf = os.path.join(thesis_dir, "main.pdf")
    final_pdf = os.path.join(thesis_dir, "BTP1_Report_Final.pdf")
    
    if os.path.exists(main_pdf):
        doc = fitz.open(main_pdf)
        print(f"SUCCESS: Built main.pdf with {len(doc)} pages.")
        # Synchronize BTP1_Report_Final.pdf
        shutil.copyfile(main_pdf, final_pdf)
        print(f"SUCCESS: Synchronized {final_pdf} ({os.path.getsize(final_pdf)} bytes).")
        return len(doc)
    else:
        print("FAILED to build main.pdf. Output:")
        print(res.stdout[-1500:])
        return 0

if __name__ == "__main__":
    build_all()
