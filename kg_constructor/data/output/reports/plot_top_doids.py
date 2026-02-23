import matplotlib.pyplot as plt
import numpy as np
import os

# Dados
data = [
    ["DOID:4", "disease", 800], ["DOID:10763", "hypertension", 271],
    ["DOID:225", "syndrome", 263], ["DOID:162", "cancer", 245],
    ["DOID:848", "arthritis", 197], ["DOID:7148", "rheumatoid arthritis", 164],
    ["DOID:1596", "depression", 158], ["DOID:9351", "diabetes mellitus", 152],
    ["DOID:8893", "psoriasis", 107], ["DOID:5844", "myocardial infarction", 105],
    ["DOID:0060180", "lung cancer", 101], ["DOID:557", "asthma", 92],
    ["DOID:5419", "schizophrenia", 89], ["DOID:0051061", "multiple sclerosis", 88],
    ["DOID:8577", "ulcerative colitis", 88], ["DOID:2841", "asthma", 80],
    ["DOID:2030", "atopic dermatitis", 78], ["DOID:552", "malaria", 78],
    ["DOID:3083", "COPD", 74], ["DOID:2723", "glaucoma", 73]
]

# Separação de dados
ids = [d[0] for d in data]
names = [d[1].capitalize() for d in data] # Capitalizar para aspeto formal
counts = [d[2] for d in data]
blacklist = {"DOID:4", "DOID:225", "DOID:162", "DOID:848"}

# Configuração de Cores
color_line = '#BDC3C7'      
color_normal = '#2E86C1'    
color_blacklist = '#CB4335' 
color_text = '#2C3E50'      

plt.figure(figsize=(14, 12), facecolor='white')
y_pos = np.arange(len(names))[::-1] 

# Desenho do Gráfico
for i, (d_id, name, count) in enumerate(zip(ids, names, counts)):
    is_blacklisted = d_id in blacklist
    pt_color = color_blacklist if is_blacklisted else color_normal
    
    # 1. Linha horizontal (Lollipop stick) - do eixo Y até ao ponto
    plt.hlines(y=y_pos[i], xmin=0, xmax=count, color=color_line, 
               linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)
    
    # 2. Marcador (Ponto)
    plt.plot(count, y_pos[i], marker='o', markersize=12, 
             markerfacecolor=pt_color, markeredgecolor='white', 
             markeredgewidth=1.5, zorder=3)
    
    # 3. Label à frente do ponto: (ID) Valor
    label_text = f"({d_id}) {int(count)}"
    plt.text(count + 12, y_pos[i], label_text, va='center', 
             fontsize=10.5, fontweight='bold', color=pt_color)

# Configuração dos Eixos (Nomes em BOLD no Y)
plt.yticks(y_pos, names, fontsize=11, fontweight='bold', color=color_text)
plt.xticks(fontsize=11, color=color_text)

plt.xlabel("Frequency (Occurrences)", fontsize=13, fontweight='bold', labelpad=15, color=color_text)
plt.title("Top 20 Disease Entities", 
          fontsize=18, fontweight='bold', pad=30, color=color_text)

# Refinamentos finais
plt.subplots_adjust(left=0.25) # Ajustado pois nomes ocupam menos espaço que ID+Nome
plt.grid(axis='x', linestyle=':', alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color(color_line)
plt.gca().spines['bottom'].set_color(color_line)

# Guardar o plot na pasta de output
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.abspath(os.path.join(script_dir, "..", "plots"))
os.makedirs(plots_dir, exist_ok=True)
output_path = os.path.join(plots_dir, "top_20_doids.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()