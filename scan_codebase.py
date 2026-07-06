import os
from pathlib import Path

def get_size(path):
    size_bytes = os.path.getsize(path)
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def scan_directory():
    root_dir = Path(os.getcwd())
    output_file = "codebase_scan_report.txt"
    
    ignore_dirs = {'.git', '__pycache__', '.vscode', '.ipynb_checkpoints', 'env', 'venv'}
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"CODEBASE SCAN REPORT\n")
        f.write(f"Root: {root_dir}\n")
        f.write("="*60 + "\n\n")
        
        f.write("📁 DIRECTORY TREE & FILES:\n")
        f.write("-" * 30 + "\n")
        
        python_files = []
        data_files = []
        
        for root, dirs, files in os.walk(root_dir):
            # Remove ignored directories from traversal
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            level = root.replace(str(root_dir), '').count(os.sep)
            indent = ' ' * 4 * (level)
            f.write(f"{indent}📂 {os.path.basename(root)}/\n")
            
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                size_str = get_size(file_path)
                f.write(f"{sub_indent}📄 {file} ({size_str})\n")
                
                # Categorize files for later
                if file.endswith('.py'):
                    python_files.append(file_path)
                elif file.endswith(('.csv', '.pkl', '.pt', '.npz', '.dat', '.json')):
                    data_files.append((file_path, size_str))

        f.write("\n\n" + "="*60 + "\n")
        f.write("🐍 PYTHON SCRIPTS TO REFACTOR:\n")
        f.write("-" * 30 + "\n")
        for py_file in python_files:
            rel_path = os.path.relpath(py_file, root_dir)
            f.write(f"- {rel_path}\n")
            
        f.write("\n\n" + "="*60 + "\n")
        f.write("📦 DATA & MODEL WEIGHTS (Likely Zenodo Candidates):\n")
        f.write("-" * 30 + "\n")
        for data_file, size in data_files:
            rel_path = os.path.relpath(data_file, root_dir)
            f.write(f"- {rel_path}  |  Size: {size}\n")

    print(f"\n✅ Scan complete! Report saved to '{output_file}'.")
    print("Please open the file and paste its contents to the AI.")

if __name__ == "__main__":
    scan_directory()