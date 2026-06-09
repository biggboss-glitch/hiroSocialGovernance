import os
import re

directories = [
    r"c:\Users\win11\Downloads\MetaHackathonAgent\hiro-social-governance\MiroFish\frontend\src",
    r"c:\Users\win11\Downloads\MetaHackathonAgent\hiro-social-governance\MiroFish\backend"
]

translations = {
    "图谱构建": "Graph Build",
    "环境搭建": "Env Setup",
    "开始模拟": "Start Simulation",
    "报告生成": "Generate Report",
    "顶部导航栏": "Top Navigation",
    "中间步骤指示器": "Step Indicator",
    "主内容区": "Main Content Area"
}

def clean_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {filepath} due to read error: {e}")
        return

    original_content = content

    # Translate specific UI texts
    for zh, en in translations.items():
        content = content.replace(zh, en)

    # Remove single line comments that have Chinese (Python, JS, Vue)
    lines = content.split('\n')
    new_lines = []
    
    # Regex to match a line that is entirely a comment and contains Chinese
    # Python: ^\s*#.*[\u4e00-\u9fff]
    # JS/Vue: ^\s*//.*[\u4e00-\u9fff]
    # HTML/Vue: ^\s*<!--.*[\u4e00-\u9fff].*-->\s*$
    
    python_comment_re = re.compile(r'^\s*#.*[\u4e00-\u9fff]')
    js_comment_re = re.compile(r'^\s*//.*[\u4e00-\u9fff]')
    html_comment_re = re.compile(r'^\s*<!--.*[\u4e00-\u9fff].*-->\s*$')

    for line in lines:
        if python_comment_re.search(line) or js_comment_re.search(line) or html_comment_re.search(line):
            continue  # Skip this line
        new_lines.append(line)

    content = '\n'.join(new_lines)

    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Cleaned {filepath}")
        except Exception as e:
            print(f"Failed to write {filepath}: {e}")

for d in directories:
    for root, dirs, files in os.walk(d):
        if 'node_modules' in root or '__pycache__' in root or '.git' in root or 'venv' in root:
            continue
        for file in files:
            if file.endswith(('.py', '.js', '.vue', '.html', '.ts')):
                clean_file(os.path.join(root, file))

print("Cleanup complete.")
