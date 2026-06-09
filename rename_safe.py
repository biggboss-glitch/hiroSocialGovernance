"""
Safe replacement script that avoids BOM issues.
Replaces MiroFish -> Hiro in all relevant non-Python source files.
"""
import os

ROOT = r"c:\Users\win11\Downloads\MetaHackathonAgent\hiro-social-governance\MiroFish"

# Files to process (relative to ROOT)
targets = [
    "frontend/index.html",
    "frontend/src/App.vue",
    "frontend/src/main.js",
    "frontend/src/views/Home.vue",
    "frontend/src/views/InteractionView.vue",
    "frontend/src/views/MainView.vue",
    "frontend/src/views/Process.vue",
    "frontend/src/views/ReportView.vue",
    "frontend/src/views/SimulationRunView.vue",
    "frontend/src/views/SimulationView.vue",
    "frontend/src/components/GraphPanel.vue",
    "frontend/src/components/HistoryDatabase.vue",
    "frontend/src/components/LanguageSwitcher.vue",
    "frontend/src/components/Step1GraphBuild.vue",
    "frontend/src/components/Step2EnvSetup.vue",
    "frontend/src/components/Step3Simulation.vue",
    "frontend/src/components/Step4Report.vue",
    "frontend/src/components/Step5Interaction.vue",
    "frontend/vite.config.js",
    "docker-compose.yml",
    ".env",
    ".env.example",
]

# Also find all .vue, .js, .html files in frontend/src
for dp, dn, fn in os.walk(os.path.join(ROOT, "frontend", "src")):
    dn[:] = [d for d in dn if d != 'node_modules']
    for f in fn:
        if f.endswith(('.vue', '.js', '.html')):
            rel = os.path.relpath(os.path.join(dp, f), ROOT)
            if rel not in targets:
                targets.append(rel)

count = 0
for rel in targets:
    fp = os.path.join(ROOT, rel)
    if not os.path.exists(fp):
        continue
    with open(fp, 'r', encoding='utf-8') as fh:
        content = fh.read()
    if 'MiroFish' in content or 'mirofish' in content.lower():
        new_content = content.replace('MiroFish', 'Hiro')
        # Don't replace "mirofish_" prefixed IDs (graph IDs etc) - those are system identifiers
        with open(fp, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(new_content)
        count += 1
        print(f"Updated: {rel}")

print(f"\nTotal files updated: {count}")
