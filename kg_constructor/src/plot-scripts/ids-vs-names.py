import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Dados
categories = ['Drugs', 'Ingredients']
names_counts = [3359, 2552]  
ids_counts = [584, 2021]     

# Cores do nosso padrão
color_raw = '#BDC3C7'    # Cinza (Raw/Bruto)
color_clean = '#2E86C1'  # Azul (Clean/ID Único)
color_text = '#2C3E50'

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

# Barras com cantos ligeiramente arredondados (simulado por linewidth)
rects1 = ax.bar(x - width/2, names_counts, width, label='By Name', color=color_raw, edgecolor='white', linewidth=1)
rects2 = ax.bar(x + width/2, ids_counts, width, label='By ID', color=color_clean, edgecolor='white', linewidth=1)

# Estética de Eixos
ax.set_ylabel(' Total of Entities', fontsize=12, fontweight='bold', color=color_text)
ax.set_title('Quantity vs Quality', fontsize=16, fontweight='bold', pad=25, color=color_text)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color=color_text)
ax.legend(fontsize=11, frameon=False, loc='upper right')

# Adicionar valores e a "Métrica de Impacto"
for i, rect in enumerate(rects1):
    # Valor em cima da barra bruta
    ax.annotate(f'{names_counts[i]}', xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold')
    
    # Valor em cima da barra limpa
    clean_rect = rects2[i]
    ax.annotate(f'{ids_counts[i]}', xy=(clean_rect.get_x() + clean_rect.get_width()/2, clean_rect.get_height()),
                xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold', color=color_clean)

# Limpeza Final
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(axis='y', linestyle=':', alpha=0.3)

plt.tight_layout()
# Guardar o plot na pasta plots
plots_dir = Path(__file__).parent.parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
output_path = plots_dir / "ids_vs_names.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot guardado em: {output_path}")