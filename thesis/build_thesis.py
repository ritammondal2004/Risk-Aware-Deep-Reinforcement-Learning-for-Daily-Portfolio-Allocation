import os
import subprocess
import shutil
import fitz

def build_all():
    thesis_dir = r"c:\RTM\IIT KGP\sem 7\BTP1_repo\thesis"
    miktex_bin = r"C:\Users\DELL\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
    pdflatex = os.path.join(miktex_bin, "pdflatex.exe")
    bibtex = os.path.join(miktex_bin, "bibtex.exe")

    # Terminate any running Acrobat instance that could lock PDF files
    try:
        subprocess.run(["taskkill", "/F", "/IM", "Acrobat.exe"], capture_output=True)
    except Exception:
        pass

    env = os.environ.copy()
    paths = env.get("PATH", "").split(";")
    clean_paths = [p for p in paths if "anaconda" not in p.lower() and "conda" not in p.lower()]
    clean_paths.insert(0, miktex_bin)
    env["PATH"] = ";".join(clean_paths)

    # 1. Build main.tex with full BibTeX resolution
    print("Building main.pdf...")
    subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"], cwd=thesis_dir, env=env, capture_output=True)
    b_res = subprocess.run([bibtex, "main"], cwd=thesis_dir, env=env, capture_output=True, text=True)
    if b_res.returncode != 0:
        print("BibTeX error:", b_res.stdout, b_res.stderr)
    subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"], cwd=thesis_dir, env=env, capture_output=True)
    res = subprocess.run([pdflatex, "-interaction=nonstopmode", "main.tex"], cwd=thesis_dir, env=env, capture_output=True, text=True)

    main_pdf = os.path.join(thesis_dir, "main.pdf")
    final_pdf = os.path.join(thesis_dir, "BTP1_Report_Final.pdf")
    
    if res.returncode == 0 and os.path.exists(main_pdf):
        doc = fitz.open(main_pdf)
        print(f"SUCCESS: Built main.pdf with {len(doc)} pages.")
        # Synchronize BTP1_Report_Final.pdf
        try:
            shutil.copyfile(main_pdf, final_pdf)
            print(f"SUCCESS: Synchronized {final_pdf} ({os.path.getsize(final_pdf)} bytes).")
        except Exception as e:
            print(f"NOTE: Could not sync {final_pdf} ({e}). main.pdf was successfully built ({len(doc)} pages).")
        return len(doc)
    else:
        print(f"FAILED to build main.pdf (exit code {res.returncode}). Output:")
        print(res.stdout[-2500:] if res.stdout else "No stdout")
        return 0

if __name__ == "__main__":
    build_all()
